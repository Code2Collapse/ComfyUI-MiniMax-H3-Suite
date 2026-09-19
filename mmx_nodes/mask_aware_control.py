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

logger = logging.getLogger(__name__)

_PKG = Path(__file__).resolve().parents[1]
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from mmx_utils.mask_aware_control import control_gate, describe_gate  # noqa: E402
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
                 inpaint: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
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

        try:
            control_latent = None
            if self.control_video is not None:
                control_latent = self._encode(
                    self._fit_frames(self.control_video, frame_count, width, height),
                    target_shape)

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
                visibility = comfy.utils.common_upscale(
                    visibility, width, height, "bilinear", "center")
                visibility = (visibility > 0.5).to(torch.float32)
                source = (torch.zeros(frame_count, 3, height, width,
                                      dtype=visibility.dtype, device=visibility.device)
                          if self.source_video is None else
                          self._fit_frames(self.source_video, frame_count, width, height))
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
            logger.info("[MiniMaxSuite] %s", self.hint_report)
        finally:
            comfy.model_management.load_models_gpu(loaded)

        self.control_latent = hint
        self.control_latent_shape = target_shape

    # ── capture the mask ────────────────────────────────────────────────────
    def diffusion_model_wrapper(self, executor, x, timestep, context,
                                transformer_options, **kwargs):
        # This is the only place the denoise mask is visible: the block patches
        # downstream receive the layout but not the mask.
        self._denoise_mask = kwargs.get("denoise_mask")
        self._gate = None
        self._gate_key = None
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
        self._gate = gate.to(device)
        self._gate_key = key
        self.last_report = describe_gate(gate, layout.img_update)
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
        try:
            gate = self._gate_for(layout, self.control_latent.shape, skip.device)
            img_pos = layout.img_pos.to(skip.device)
            skip[img_pos] = skip[img_pos] * gate.to(skip.dtype).unsqueeze(-1)
        except Exception as exc:  # noqa: BLE001
            # A gate that cannot be built must not take the render down. Fall
            # back to core behaviour and say so once, loudly enough to find.
            logger.warning(
                "[MiniMaxSuite] mask-aware control gate unavailable, falling "
                "back to ungated core behaviour: %s", exc)
            self.last_report = f"gate unavailable, ran ungated: {exc}"

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

        out["img"].add_(skip, alpha=self.strength)
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
                    tooltip="The plate the mask is cut from, for inpainting. In "
                            "a cropped workflow this is the CROPPED video, the "
                            "same one that is VAE-encoded for the sampler."),
                io.Float.Input(
                    "sigma_start", default=1.0, min=0.0, max=1.0, step=0.01,
                    optional=True, advanced=True),
                io.Float.Input(
                    "sigma_end", default=0.0, min=0.0, max=1.0, step=0.01,
                    optional=True, advanced=True),
            ],
            outputs=[
                io.Model.Output("model"),
                io.String.Output(
                    "report",
                    tooltip="What was configured, and what to look for if the "
                            "control still fights the mask."),
            ],
        )

    @classmethod
    def execute(cls, model, control_net, vae, strength, preserved_strength,
                boundary_softness, affect_text_rows=False, inpaint=False,
                control_video=None, mask=None, source_video=None,
                sigma_start=1.0, sigma_end=0.0):
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
        _require_core()

        patched = model.clone()
        patch = MaskAwareControlPatch(
            control_net, vae, control_video, mask, source_video, strength,
            sigma_start, sigma_end,
            preserved_strength=preserved_strength,
            boundary_softness=boundary_softness,
            affect_text_rows=affect_text_rows,
            inpaint=inpaint,
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
        lines.append(
            "Per-row detail is logged at sample time; the counts depend on the "
            "mask you actually sample with.")
        return io.NodeOutput(patched, "\n".join(lines))

NODE_LIST = [MiniMaxH3_MaskAwareControlNet]
