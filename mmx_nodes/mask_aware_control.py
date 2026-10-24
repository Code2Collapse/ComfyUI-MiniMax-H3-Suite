"""H3 ControlNet that respects the latent mask instead of fighting it.

The defect and the fix are documented in mmx_utils/mask_aware_control.py, with
the ComfyUI lines they live on. In short: the core H3 Fun ControlNet adds its
residual to every video row, while latent masking tells some of those rows they
are already finished. Two forces, one set of rows, opposite directions.

This node wraps ComfyUI's own control patch rather than reimplementing it, so
every improvement to the core control path is inherited and only the residual
application changes. If ComfyUI's internals move, this fails with a sentence
naming what it could not find, instead of silently going back to the old
behaviour.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import torch
from comfy_api.latest import io

GATE_EVENT = "minimax_h3_gate_report"


def _send_gate_report(node_id: str, report: str, hint: str = "") -> None:
    """C2C L9.39: the gate is only known once sampling starts (the denoise mask arrives with the sampler), after
    this node has returned its text output - so the measured report goes to the node's panel as an event (it was
    computed and never shown). Never raises: a report must not stop a render."""
    if not node_id:
        return
    try:
        import server
        server.PromptServer.instance.send_sync(GATE_EVENT, {"node_id": node_id, "report": report, "hint": hint})
    except Exception:  # noqa: BLE001
        pass

logger = logging.getLogger(__name__)

_PKG = Path(__file__).resolve().parents[1]
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from mmx_utils.mask_aware_control import (  # noqa: E402
    SCHEDULES,
    apply_region,
    boxes_from_bboxes,
    control_gate,
    crop_by_boxes,
    describe_gate,
    row_values_from_mask,
    schedule_scale,
)
from mmx_utils.fun_control_hint import (  # noqa: E402
    assemble_hint,
    describe_hint,
    frame_indices,
    looks_like_a_latent_mask,
    visibility_from_mask,
)

# The H3 control machinery lives in ComfyUI's own extras. Import defensively:
# `comfy_extras.nodes_minimax_h3` pulls in the H3 model modules, which on a
# build/environment version skew raise AttributeError, not ImportError - that
# skew has already cost this pack six nodes once. The schema must stay
# constructible either way, so the failure is deferred to execute().
_CORE_IMPORT_ERROR: Exception | None = None
try:
    import torch.nn.functional as F
    import comfy.model_management
    import comfy.utils
    from comfy_extras.nodes_minimax_h3 import MiniMaxH3FunControlPatch
except Exception as _e:  # noqa: BLE001
    _CORE_IMPORT_ERROR = _e
    MiniMaxH3FunControlPatch = object  # type: ignore[assignment,misc]


def _require_core() -> None:
    if _CORE_IMPORT_ERROR is not None:
        raise RuntimeError(
            "Mask-Aware ControlNet needs ComfyUI's own MiniMax H3 control nodes, "
            f"which failed to import here ({type(_CORE_IMPORT_ERROR).__name__}: "
            f"{_CORE_IMPORT_ERROR}). That usually means this ComfyUI build has no "
            "MiniMax H3 support, or the running ComfyUI and its installed "
            "comfy_kitchen are at different versions. Update ComfyUI."
        )


class MaskAwareControlPatch(MiniMaxH3FunControlPatch):  # type: ignore[misc,valid-type]
    """The core patch, with the residual gated by the sampler's denoise mask."""

    def __init__(self, *args, preserved_strength: float = 0.0,
                 boundary_softness: float = 0.0, affect_text_rows: bool = False,
                 inpaint: bool = False, control_region=None, schedule: str = "constant",
                 crop_boxes=None, node_id: str = "", **kwargs):
        super().__init__(*args, **kwargs)
        self.node_id = str(node_id or "")
        self.control_region = control_region
        self.schedule = schedule
        self.crop_boxes = crop_boxes
        self._region_rows = None
        self._sched = 1.0
        self.inpaint = bool(inpaint)
        self.hint_report = "not built yet"
        self.preserved_strength = float(preserved_strength)
        self.boundary_softness = float(boundary_softness)
        self.affect_text_rows = bool(affect_text_rows)
        self._denoise_mask = None
        self._gate = None
        self._gate_key = None
        self.last_report = "not run yet"

    # ── build the whole 49-channel hint ─────────────────────────────────────
    def prepare_control_latent(self, target_shape):
        """The full hint, with neutral values where ComfyUI leaves zeros.

        See mmx_utils/fun_control_hint.py for why the visibility channel's
        neutral value is ONES and not zeros. Short version: visibility 0 means
        "this is a hole", so a zero-padded channel asks the Union model to
        inpaint the entire frame.
        """
        target_shape = tuple(target_shape)
        if self.control_latent is not None and self.control_latent_shape == target_shape:
            return

        latent_frames, lat_h, lat_w = target_shape[2:]
        frame_count = max((latent_frames - 2) // 5, 0) * 17 + 5
        sc = self.vae.spacial_compression_encode()
        width, height = lat_w * sc, lat_h * sc
        loaded = comfy.model_management.loaded_models(only_currently_used=True)

        def fit(frames, mode="bilinear"):
            # A16: with Subject Crop boxes, plate-sized frames take the source's own crop + stretch, so they land on
            # the cropped target exactly; without them, core's cover-crop (centre) is kept.
            if self.crop_boxes is not None and tuple(frames.shape[-2:]) != (height, width):
                idx = torch.arange(frame_count, device=frames.device).clamp(max=frames.shape[0] - 1)
                return crop_by_boxes(frames[idx], self.crop_boxes, width, height, mode)
            return self._fit_frames(frames, frame_count, width, height)

        try:
            control_latent = None
            if self.control_video is not None:
                control_latent = self._encode(fit(self.control_video), target_shape)

            visibility_latent = masked_latent = None
            inpainting = self.inpaint and self.mask is not None
            if inpainting:
                mask_hw = (int(self.mask.shape[-2]), int(self.mask.shape[-1]))
                if looks_like_a_latent_mask(mask_hw, (lat_h, lat_w), (height, width)):
                    raise ValueError(
                        f"the mask wired into this node is {mask_hw[0]}x{mask_hw[1]}, "
                        f"which is LATENT resolution - the latent here is "
                        f"{lat_h}x{lat_w} and the picture is {height}x{width}. "
                        "That is the mask for Set Latent Noise Mask, not for the "
                        "ControlNet. Wire the PIXEL-space mask here (the one that "
                        "goes into Mask To Latent Space), and keep the latent one "
                        "on the sampler. Feeding the latent mask here stretches a "
                        "handful of rows across the whole clip."
                    )
                visibility = visibility_from_mask(self.mask, frame_count, inpaint=True)
                if self.crop_boxes is not None and tuple(visibility.shape[-2:]) != (height, width):
                    visibility = crop_by_boxes(visibility, self.crop_boxes, width, height)
                else:
                    visibility = comfy.utils.common_upscale(
                        visibility, width, height, "bilinear", "center")
                visibility = (visibility > 0.5).to(torch.float32)
                source = (torch.zeros(frame_count, 3, height, width,
                                      dtype=visibility.dtype, device=visibility.device)
                          if self.source_video is None else
                          fit(self.source_video))
                masked_latent = self._encode(
                    source * visibility.to(source.device), target_shape)
                visibility_latent = F.interpolate(
                    visibility.squeeze(1)[None, None],
                    size=(latent_frames, lat_h, lat_w),
                    mode="trilinear", align_corners=False)

            reference = control_latent if control_latent is not None else masked_latent
            device = reference.device if reference is not None else None
            dtype = reference.dtype if reference is not None else torch.float32
            hint = assemble_hint(
                control_latent, visibility_latent, masked_latent,
                latent_shape=target_shape, device=device, dtype=dtype)
            self.hint_report = describe_hint(
                control_latent is not None, inpainting, frame_count, latent_frames)
            self._region_rows = self._build_region_rows(frame_count, width, height, latent_frames, lat_h, lat_w)
            logger.info("[MiniMaxSuite] %s", self.hint_report)
        finally:
            comfy.model_management.load_models_gpu(loaded)

        self.control_latent = hint
        self.control_latent_shape = target_shape

    def _build_region_rows(self, frame_count, width, height, latent_frames, lat_h, lat_w):
        """control_region (pixel, 1 = this control may steer) -> one value per generated latent row, reduced on
        H3's own frame grouping (H3 Mask To Latent Space) and pooled per 2x2 patch like the denoise mask."""
        if self.control_region is None:
            return None
        from mmx_nodes.mask_to_latent import MiniMaxH3_MaskToLatentSpace
        region = self.control_region.to(torch.float32)
        if region.ndim == 2:
            region = region[None]
        idx = torch.arange(frame_count, device=region.device).clamp(max=region.shape[0] - 1)
        region = region[idx][:, None]
        if self.crop_boxes is not None and tuple(region.shape[-2:]) != (height, width):
            region = crop_by_boxes(region, self.crop_boxes, width, height)
        elif tuple(region.shape[-2:]) != (height, width):
            region = comfy.utils.common_upscale(region, width, height, "bilinear", "center")
        out = MiniMaxH3_MaskToLatentSpace.execute(masks=region[:, 0], compression={"compression": "auto"},
                                                  spatial_method="max", temporal_method="max",
                                                  grow_spatial=0, grow_temporal=0, vae=self.vae)
        lat = (out.args[0] if hasattr(out, "args") else out[0]).to(torch.float32)
        if tuple(lat.shape) != (int(latent_frames), int(lat_h), int(lat_w)):
            raise ValueError(f"control_region reduced to {tuple(lat.shape)} but the latent is "
                             f"{(int(latent_frames), int(lat_h), int(lat_w))}: give one region per frame of the "
                             "video being generated (or the plate with crop_bboxes).")
        return row_values_from_mask(lat, int(latent_frames), int(lat_h), int(lat_w))

    # ── capture the mask ────────────────────────────────────────────────────
    def diffusion_model_wrapper(self, executor, x, timestep, context,
                                transformer_options, **kwargs):
        # This is the only place the denoise mask is visible: the block patches
        # downstream receive the layout but not the mask.
        self._denoise_mask = kwargs.get("denoise_mask")
        self._gate = None
        self._gate_key = None
        sigmas = transformer_options.get("sigmas")
        sigma = float(sigmas[0]) if sigmas is not None else float(timestep.flatten()[0]) / 1000.0
        self._sched = schedule_scale(self.schedule, sigma, self.sigma_start, self.sigma_end)
        try:
            return super().diffusion_model_wrapper(
                executor, x, timestep, context, transformer_options, **kwargs)
        finally:
            self._denoise_mask = None

    # ── gate the residual ───────────────────────────────────────────────────
    def _gate_for(self, layout, latent_shape, device):
        mask = self._denoise_mask
        latent_t, lat_h, lat_w = latent_shape[2:]
        key = (id(mask), int(latent_t), int(lat_h), int(lat_w))
        if self._gate is not None and self._gate_key == key:
            return self._gate

        mask3 = None
        if mask is not None:
            # [B,C,T,H,W] as the sampler hands it over; the model reads [0,0].
            mask3 = mask[0, 0] if mask.ndim == 5 else mask
        gate = control_gate(
            mask3, layout.img_update, int(latent_t), int(lat_h), int(lat_w),
            preserved_strength=self.preserved_strength,
            boundary_softness=self.boundary_softness,
        )
        gate = apply_region(gate, layout.img_update, self._region_rows)
        self._gate = gate.to(device)
        self._gate_key = key
        self.last_report = describe_gate(gate, layout.img_update)
        _send_gate_report(self.node_id, self.last_report, getattr(self, "hint_report", ""))
        return self._gate

    def after_block(self, block_index, args, out):
        if not self.active:
            return out
        control_index = self.model_patch.model.injection_layers.index(block_index)
        if control_index == 0:
            self.control_latent = self.control_latent.to(out["img"].device)
            self.control_stream = self.model_patch.model.init_stream(
                self.pristine_stream, self.control_latent, args["layout"], args["t_emb"])
            self.pristine_stream = None
        self.control_stream, skip = self.model_patch.model.step(
            control_index, self.control_stream, args["t_emb"], args["mod_segments"],
            args["rope_freqs"], transformer_options=args["transformer_options"])

        layout = args["layout"]
        skip[layout.audio_pos.to(skip.device)] = 0

        # THE CHANGE. Core adds `skip` to every image row; here each row is
        # scaled by how much the sampler is actually generating it.
        # A16: no silent fallback to ungated control. A gate that cannot be built is an error the user sees,
        # not a quietly different operation (the old code ran core's ungated behaviour and only logged it).
        gate = self._gate_for(layout, self.control_latent.shape, skip.device)
        img_pos = layout.img_pos.to(skip.device)
        skip[img_pos] = skip[img_pos] * gate.to(skip.dtype).unsqueeze(-1)

        if not self.affect_text_rows:
            # Core also leaks the residual onto the TEXT rows, because the
            # control stream starts as a clone of h and only the image rows are
            # overwritten. Off by default here: the prompt is not something a
            # depth map should be editing.
            # The layout has no text_pos, but `segments` is a list of
            # (start, end, kind) and names the text span directly.
            for a, b, kind in layout.segments:
                if kind == "text":
                    skip[a:b] = 0

        out["img"].add_(skip, alpha=self.strength * self._sched)
        return out


class MiniMaxH3_MaskAwareControlNet(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MiniMaxH3_MaskAwareControlNet",
            display_name="H3 Mask-Aware ControlNet",
            category="MiniMax H3/Conditioning",
            description=(
                "H3 Fun ControlNet-Union that works alongside a latent noise "
                "mask. Two fixes over ComfyUI's own node. (1) It builds all 49 "
                "control channels. ComfyUI only builds the visibility and "
                "masked-latent channels when a mask is connected and zero-pads "
                "otherwise - but visibility 0 means 'this is a hole', so a "
                "control-video-only graph is silently asking the Union model to "
                "inpaint the entire frame. Here the neutral value is ONES. "
                "(2) Its residual is scaled per latent row by the sampler's own "
                "denoise mask, so the control steers only what is actually "
                "being generated instead of pushing rows the sampler is holding "
                "still. mask/source_video here are PIXEL space, not the latent "
                "mask that feeds Set Latent Noise Mask."
            ),
            inputs=[
                io.Model.Input("model", tooltip="The H3 model to patch."),
                io.ModelPatch.Input(
                    "control_net",
                    tooltip="A MiniMax H3 Fun ControlNet, loaded with ComfyUI's "
                            "ModelPatchLoader."),
                io.Vae.Input("vae", tooltip="The H3 VAE, used to encode the hint."),
                io.Float.Input(
                    "strength", default=1.0, min=0.0, max=10.0, step=0.01,
                    tooltip="Overall control strength, as in the core node."),
                io.Float.Input(
                    "preserved_strength", default=0.0, min=0.0, max=1.0, step=0.05,
                    tooltip="How much control still reaches rows the latent mask "
                            "asked to preserve. 0 keeps the control out of them "
                            "entirely, which is the point of this node. 1 restores "
                            "ComfyUI's behaviour of ignoring the mask."),
                io.Float.Input(
                    "boundary_softness", default=1.0, min=0.0, max=16.0, step=0.5,
                    tooltip="Blur the gate at the mask edge, in latent patches. A "
                            "hard gate leaves a visible seam where the control "
                            "stops; 1-2 patches is usually enough to hide it."),
                io.Boolean.Input(
                    "affect_text_rows", default=False, optional=True,
                    tooltip="ComfyUI's control stream also adds a residual to the "
                            "TEXT rows, because it starts as a copy of the whole "
                            "sequence. Off here: a depth map should not be editing "
                            "the prompt. Turn it on to match core exactly."),
                io.Image.Input(
                    "control_video", optional=True,
                    tooltip="The hint - depth, pose, edges. Optional if you are "
                            "only inpainting with mask + source_video."),
                io.Boolean.Input(
                    "inpaint", default=False, optional=True,
                    tooltip="Off: structural control only - the visibility "
                            "channels are filled with ones, meaning nothing is a "
                            "hole. On: also drive the Union model's inpainting "
                            "branch from mask + source_video. The upstream model "
                            "card treats these as two separate modes and ships a "
                            "different script for each, so this is a switch, not "
                            "something to infer."),
                io.Mask.Input(
                    "mask", optional=True,
                    tooltip="PIXEL-space inpaint mask, only used when inpaint is "
                            "on. This is NOT the latent mask that feeds Set "
                            "Latent Noise Mask - wire the same pixel mask you "
                            "send into Mask To Latent Space. A latent mask here "
                            "is refused by name rather than stretched across the "
                            "clip. The sampler's own denoise mask is read "
                            "automatically and needs no wire."),
                io.Image.Input(
                    "source_video", optional=True,
                    tooltip="The plate the mask is cut from. REQUIRED when "
                            "inpaint is on: these become the masked-latent "
                            "channels, which are the plate with the hole cut "
                            "out of it, and ComfyUI's own node substitutes a "
                            "BLACK frame when it is missing. In a cropped "
                            "workflow this is the CROPPED video - the same one "
                            "that is VAE-encoded for the sampler."),
                io.Float.Input(
                    "sigma_start", default=1.0, min=0.0, max=1.0, step=0.01,
                    optional=True, advanced=True),
                io.Float.Input(
                    "sigma_end", default=0.0, min=0.0, max=1.0, step=0.01,
                    optional=True, advanced=True),
                # A16 Part III, appended: where this control applies, how its strength runs, crop alignment
                io.Mask.Input(
                    "control_region", optional=True,
                    tooltip="Where THIS control may steer (1) and where it is ignored (0), e.g. pose only on the "
                            "person. Combined with the sampler's denoise mask: a row is steered only where both "
                            "allow it. Pixel space, the target's size (or the plate's with crop_bboxes)."),
                io.Combo.Input(
                    "schedule", options=list(SCHEDULES), default=SCHEDULES[0], optional=True,
                    tooltip="Strength over the steps inside the sigma window. constant: as before. fade out: full "
                            "at the start, nothing at the end - the control sets the layout and leaves the detail "
                            "to the model. fade in: the reverse. smooth fade out: eased."),
                io.BoundingBox.Input(
                    "crop_bboxes", optional=True, force_input=True,
                    tooltip="Subject Crop's boxes. With them, a PLATE-sized control video / source video / mask / "
                            "region is cropped with the same boxes and stretched like the source, so it lands "
                            "exactly on the cropped target. Without them, ComfyUI's centre cover-crop is used - a "
                            "plate-sized control on a cropped target is then misaligned."),
            ],
            outputs=[
                io.Model.Output("model"),
                io.String.Output(
                    "report",
                    tooltip="What was configured, and what to look for if the "
                            "control still fights the mask."),
            ],
            hidden=[io.Hidden.unique_id],
        )

    @classmethod
    def execute(cls, model, control_net, vae, strength, preserved_strength,
                boundary_softness, affect_text_rows=False, inpaint=False,
                control_video=None, mask=None, source_video=None,
                sigma_start=1.0, sigma_end=0.0, control_region=None,
                schedule="constant", crop_bboxes=None):
        # Wiring first, environment second. A user who connected nothing gets
        # told that on any machine; the environment message is only useful to
        # someone whose wiring is already right.
        if control_video is None and mask is None:
            raise ValueError(
                "Mask-Aware ControlNet needs something to steer with: connect a "
                "control_video (depth, pose, edges), or a mask plus source_video "
                "for inpainting. With neither, the ControlNet has no hint."
            )
        if inpaint and mask is None:
            raise ValueError(
                "inpaint is on but no mask is connected. Either wire the "
                "PIXEL-space mask (the one that also feeds Mask To Latent "
                "Space), or turn inpaint off for structural control only."
            )
        if inpaint and source_video is None:
            # Core silently substitutes a black plate here. The visibility
            # channel then reads "everything outside the hole is visible" while
            # the masked-latent channels read "and it is solid black" - two
            # statements the model has no way to reconcile, and the reason a
            # mask-plus-control graph can come back looking destroyed.
            raise ValueError(
                "inpaint is on with a mask but no source_video. The Union "
                "model's masked-latent channels are the PLATE with the hole "
                "cut out of it, so without a plate they encode a black frame "
                "while the visibility channel still claims that area is "
                "visible. Connect the video the mask was drawn on - in a "
                "cropped workflow, the cropped video that is also VAE-encoded "
                "for the sampler - or turn inpaint off."
            )
        _require_core()

        patched = model.clone()
        patch = MaskAwareControlPatch(
            control_net, vae, control_video, mask, source_video, strength,
            sigma_start, sigma_end,
            preserved_strength=preserved_strength,
            boundary_softness=boundary_softness,
            affect_text_rows=affect_text_rows,
            inpaint=inpaint,
            control_region=control_region,
            schedule=schedule,
            crop_boxes=boxes_from_bboxes(crop_bboxes),
            node_id=str(getattr(cls.hidden, "unique_id", "") or ""),
        )
        patch.register(patched)

        lines = [
            f"Control strength {strength:.2f}, gated by the sampler's denoise mask.",
            ("Inpaint mode: visibility and masked-latent channels are built "
             "from the connected mask."
             if inpaint else
             "Structural control only: visibility is ONES, so nothing is "
             "flagged as a hole. ComfyUI's own node leaves those channels at "
             "zero, which asks the model to inpaint the whole frame."),
        ]
        if preserved_strength <= 0.0:
            lines.append(
                "Preserved rows get NO control - the mask holds them and nothing "
                "pushes back.")
        elif preserved_strength >= 1.0:
            lines.append(
                "preserved_strength is 1.0, so the mask is ignored and this "
                "behaves exactly like ComfyUI's own control node, contradiction "
                "included. Lower it to get the fix.")
        else:
            lines.append(
                f"Preserved rows keep {preserved_strength:.0%} of the control - a "
                "partial hold, not a full one.")
        lines.append(
            f"Gate edge softened over {boundary_softness:.1f} latent patches."
            if boundary_softness > 0 else
            "Hard gate edge: if you see a seam where the control stops, raise "
            "boundary_softness.")
        if affect_text_rows:
            lines.append(
                "affect_text_rows is on: the residual also reaches the prompt "
                "rows, matching core.")
        if control_region is not None:
            lines.append("control_region: the control only steers rows that both the region and the denoise mask allow.")
        if schedule != "constant":
            lines.append(f"Strength schedule: {schedule} over the sigma window {sigma_start:.2f} -> {sigma_end:.2f}.")
        lines.append("crop_bboxes: plate-sized control / source / mask are cropped with the source's own boxes."
                     if crop_bboxes is not None else
                     "No crop_bboxes: a control video with another aspect than the target is centre cover-cropped "
                     "(ComfyUI's rule) - for a cropped target, wire Subject Crop's bboxes.")
        lines.append(
            "Per-row detail is logged at sample time; the counts depend on the "
            "mask you actually sample with.")
        return io.NodeOutput(patched, "\n".join(lines))

NODE_LIST = [MiniMaxH3_MaskAwareControlNet]
