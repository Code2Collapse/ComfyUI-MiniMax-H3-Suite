"""H3 Open Regen: one node for both Open-H3 Regenerate recipes (owner 2026-11-25).

draft frames -> 24 fps, 17k+5 frames, draft canvas -> H3 VAE encode -> 3D latent upscale onto the planned output grid
(any aspect: keep the draft's, a preset or custom, or exact width x height with fit) -> refine latent (audio pinned)
-> model patched for the recipe -> core sampling (BasicGuider, Euler, CFG 1, explicit sigmas) -> optional refined
latent saved before decoding -> decode (core H3 VAE or NVIDIA's fast decoder) -> real frames, soundtrack cut ->
low-frequency colour lock to the draft (no clamping).

Regenerate = rdh9616/ComfyUI-Open-H3-Regenerate (Apache-2.0): merged refiner + Turbo + LMS Sharp LoRA and the Sharp
curve branches on an FL2VA model, text-only conditioning, shift-6 grid from start_sigma (0.8) or the release grid.
IC = slmonker/ComfyUI-OpenH3Regenerate2KIC (Apache-2.0): Ref2VA model, speed -> LMS Sharp -> IC LoRA, core
MiniMax H3 Reference to Video with the draft as <Video 1> and the official IC task prompt, SigmaShift 12 / 3, the
up3 / up4 / hf8 grids. Both recipes run through the Suite's ported, tested pieces (mmx_nodes/open_regen.py,
mmx_utils/open_regen_ic.py); this file only orders them. Distributed as part of a GPL-3.0 work.
"""
from __future__ import annotations

import logging
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from comfy_api.latest import io

_PKG = Path(__file__).resolve().parents[1]
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from mmx_nodes import latent_upscale_model as LU  # noqa: E402
from mmx_nodes import open_regen as OR  # noqa: E402
from mmx_utils import colour_lock as CL  # noqa: E402
from mmx_utils import h3_aspect as A  # noqa: E402
from mmx_utils import open_regen_core as core  # noqa: E402
from mmx_utils import open_regen_ic as R  # noqa: E402

LOG = logging.getLogger("MiniMaxSuite.open_regen_all")
CATEGORY = "MiniMax H3/Regenerate"
MODE_REGEN, MODE_IC = "Regenerate (rdh9616)", "IC (2K IC)"
DECODERS = ("core H3 VAE (best quality)", "fast decoder (StreamRender-H3)")
DRAFT_AREA = 1344 * 768            # the draft canvas both recipes were trained on (768p); smaller drafts stay native
ONE_PASS_AREA = 2560 * 1440        # above this the upscaler runs two passes (its trained range, as upstream does 4K)
IC_SHIFT = (12.0, 3.0)             # slmonker's ModelSamplingMiniMaxH3 values
REGEN_SEED = 1766580932            # rdh9616's refine seed


def _out(node_output):
    return node_output.args if hasattr(node_output, "args") else node_output


def _resize_frames(frames, w, h, chunk=16):
    """[N, H, W, C] -> [N, h, w, 3]: cover-scale (bilinear, antialiased when shrinking) + centre crop."""
    src_h, src_w = int(frames.shape[1]), int(frames.shape[2])
    if (src_w, src_h) == (w, h):
        return frames[..., :3].contiguous()
    sw, sh, x0, y0 = core.cover_crop_box(src_w, src_h, w, h)
    out = []
    for s in range(0, frames.shape[0], chunk):
        part = frames[s:s + chunk, :, :, :3].movedim(-1, 1).to(torch.float32)
        part = F.interpolate(part, size=(sh, sw), mode="bilinear", align_corners=False,
                             antialias=sw < src_w or sh < src_h)
        out.append(part[:, :, y0:y0 + h, x0:x0 + w].clamp(0, 1).movedim(1, -1))
    return torch.cat(out).contiguous()


def draft_canvas(src_w, src_h, max_area=DRAFT_AREA):
    """(w, h) on the 32 px grid closest to the draft's own aspect, at most `max_area` pixels, never larger than the
    draft beyond grid snapping. Searches the nearby 32 px pairs instead of rounding each side alone (1920x1080 ->
    1312x736, 0.3 % off, not 1344x768, 1.6 % off)."""
    ratio = src_w / float(src_h)
    s = min(1.0, math.sqrt(max_area / float(src_w * src_h)))
    h0 = src_h * s
    best = None
    for h in range(max(32, int(h0 // 32) * 32 - 96), int(math.ceil(h0 / 32)) * 32 + 64, 32):
        w = max(32, int(round(h * ratio / 32)) * 32)
        if w * h > max(max_area, 32 * 32) or w > src_w + 31 or h > src_h + 31:
            continue
        key = (round(abs((w / h) / ratio - 1.0), 4), -(w * h))
        if best is None or key < best[0]:
            best = (key, (w, h))
    if best is None:
        return max(32, int(round(src_w * s / 32)) * 32), max(32, int(round(src_h * s / 32)) * 32)
    return best[1]


def _intermediate_cells(content_cells, src_cells):
    """Two-pass upscale for targets above the trained range: first to ~2K area at the target's aspect."""
    ch, cw = content_cells
    sh, sw = src_cells
    if ch * cw * 256 <= ONE_PASS_AREA * 1.05 or sh * sw * 256 >= ONE_PASS_AREA:
        return None
    k = math.sqrt(ONE_PASS_AREA / float(ch * cw * 256))
    return max(2, int(round(ch * k / 2)) * 2), max(2, int(round(cw * k / 2)) * 2)


def _reference_for_lock(draft, plan, real):
    """The draft placed like the output (same crop / pad), for the colour lock."""
    ref = draft[:real]
    ch, cw = plan.content_cells
    ref = _resize_frames(ref, cw * 16, ch * 16) if plan.mode in ("crop", "pad") else ref
    if plan.mode == "crop":
        (oh, ow), (oy, ox) = plan.out_cells, plan.offset_cells
        ref = ref[:, oy * 16:(oy + oh) * 16, ox * 16:(ox + ow) * 16]
    elif plan.mode == "pad":
        (oh, ow), (oy, ox) = plan.out_cells, plan.offset_cells
        x = ref.movedim(-1, 1)
        pads = (ox * 16, (ow - cw - ox) * 16, oy * 16, (oh - ch - oy) * 16)
        ref = F.pad(x, pads, mode="replicate").movedim(1, -1)
    return ref


def _decode(samples, decoder, fast_vae_name, vae):
    video, _ = OR._h3_video(samples, "H3 Open Regen decode")
    if decoder == DECODERS[1]:
        images, rep = OR.fast_decode(samples, OR.load_fast_vae(OR._full_path("vae", fast_vae_name)), vae=vae)
        return images, rep
    images = vae.decode(video)
    if images.ndim == 5:
        images = images.reshape(-1, *images.shape[-3:])
    return images, f"core H3 VAE decode: {tuple(video.shape)} latent -> {images.shape[0]} frames"


def run(model, clip, vae, audio_vae, images, fps, prompt, mode, size_by, width, height, megapixels, aspect,
        custom_ratio, fit, anchor, upscaler, max_frames=0, seed=REGEN_SEED, decoder=DECODERS[0],
        fast_vae_name="", colour_lock=1.0, upscale_precision="bf16", save_latent=False, audio=None,
        lora_dtype="bf16", sampler_fn=None, colour_method="global"):
    """The whole pipeline; returns (images, audio, latent, report). `sampler_fn` is injectable for tests."""
    choice = mode.get("mode") if isinstance(mode, dict) else mode
    if choice not in (MODE_REGEN, MODE_IC):
        raise ValueError(f"H3 Open Regen: mode must be {MODE_REGEN!r} or {MODE_IC!r}")
    opts = mode if isinstance(mode, dict) else {}
    ic = choice == MODE_IC
    need = "ic_lora" if ic else "lora_name"
    if not opts.get(need) or str(opts.get(need)).startswith("("):
        raise ValueError(f"H3 Open Regen: pick the {'IC' if ic else 'Open-H3-Regenerate'} LoRA ({need}); put it in "
                         "models/loras first")
    lines = [f"mode: {choice}"]
    if not isinstance(images, torch.Tensor) or images.ndim != 4 or images.shape[0] == 0:
        raise ValueError("H3 Open Regen: no frames - connect the draft's IMAGE batch")

    # 1. draft: 24 fps, own aspect on the 32 px grid at the 768p area, 17k+5 frames
    cw, chh = draft_canvas(int(images.shape[2]), int(images.shape[1]))
    frames = _resize_frames(images, cw, chh)
    min_real = R.MIN_REAL_FRAMES if ic else 0
    draft, real, dw, dh, rep = OR.prepare_draft(frames, "keep (snap to 32)", int(max_frames), float(fps),
                                                fps_map=OR.FPS_MAPS[1] if ic else OR.FPS_MAPS[0],
                                                min_real_frames=min_real)
    padded = int(draft.shape[0])
    lines += [f"draft canvas {dw}x{dh} px (source {images.shape[2]}x{images.shape[1]})", rep]

    # 2. encode + plan + upscale
    latent = {"samples": vae.encode(draft[:, :, :, :3])}
    src_cells = tuple(int(v) for v in OR._h3_video(latent, "H3 Open Regen")[0].shape[-2:])
    if aspect == "source":   # plan from the draft's TRUE ratio, not the 32 px canvas's rounded one
        aspect, custom_ratio = "custom", f"{int(images.shape[2])}:{int(images.shape[1])}"
    try:
        plan = A.plan(src_cells, size_by, int(width), int(height), 2.0, float(megapixels), aspect, custom_ratio,
                      fit, anchor)
    except A.SizingError as exc:
        raise ValueError(f"H3 Open Regen: {exc}") from None
    lines.append(plan.report())
    mid = _intermediate_cells(plan.content_cells, src_cells)
    if mid is not None:
        latent, r1 = LU.upscale(latent, "model", upscaler, 0, 0, "auto", upscale_precision, 0, 4, True, cells=mid)
        lines.append(f"upscale pass 1 -> {mid[1] * 16}x{mid[0] * 16} px\n  " + r1.replace("\n", "\n  "))
    latent, r2 = LU.upscale(latent, "model", upscaler, 0, 0, "auto", upscale_precision, 0, 4, True,
                            cells=plan.content_cells)
    lines.append("upscale -> target\n  " + r2.replace("\n", "\n  "))
    latent, note = LU.frame(latent, plan, "edge")
    if note:
        lines.append(note)
    out_w, out_h = plan.out_px

    # 3. conditioning, target latent, model, sigmas
    if ic:
        from comfy_extras.nodes_minimax_h3 import MiniMaxH3ReferenceToVideo, MiniMaxH3SigmaShift
        cond, ref_latent = _out(MiniMaxH3ReferenceToVideo.execute(
            clip=clip, prompt=R.build_prompt(prompt), width=out_w, height=out_h, length=padded,
            ref_image_size="match", vae=vae, audio_vae=audio_vae, ref_videos={"ref_video_0": draft}))[:2]
        target, frames_n, rrep = OR.refine_latent(latent, OR.AUDIO_MODES[0], audio_vae, None,
                                                   reference_latent=ref_latent)
        m, lrep = OR.apply_lora_recipe(model, opts.get("ic_lora"), 1.0,
                                       torch.bfloat16 if lora_dtype == "bf16" else torch.float32,
                                       opts.get("speed_lora", "none"), 1.0, opts.get("sharp_lora", "none"), 1.0)
        m = _out(MiniMaxH3SigmaShift.execute(m, *IC_SHIFT))[0]
        sig, _ = R.build_sigmas(str(opts.get("ic_grid", "up3")), "")
        lrep.append(f"SigmaShift video {IC_SHIFT[0]:g} / audio {IC_SHIFT[1]:g}")
    else:
        tokens = clip.tokenize(prompt or "")
        cond = clip.encode_from_tokens_scheduled(tokens)
        target, frames_n, rrep = OR.refine_latent(latent, OR.AUDIO_MODES[0], audio_vae, None)
        m, lrep = OR.apply_lora_recipe(model, opts.get("lora_name"), 1.0,
                                       torch.bfloat16 if lora_dtype == "bf16" else torch.float32)
        curve = opts.get("curve_name", OR.NO_CURVE)
        if curve and curve != OR.NO_CURVE:
            table_choice = opts.get("curve_table", OR.OWN_TABLE)
            table = None if table_choice == OR.OWN_TABLE else OR.read_table(OR._full_path("diffusion_models",
                                                                                            table_choice))
            lrep.append(OR.apply_curves(m, OR._full_path("loras", curve), table, 1.0))
        else:
            lrep.append("Sharp curve branches: off")
        sig = core.refine_sigmas(int(opts.get("steps", 3)), float(opts.get("start_sigma", 0.8)))
    sigmas = torch.tensor(sig, dtype=torch.float32)
    lines += lrep + [rrep, "sigmas: " + ", ".join(f"{v:.6f}" for v in sig) + f" (Euler, CFG 1, seed {int(seed)})"]

    # 4. sample with core's own SamplerCustomAdvanced path
    if sampler_fn is None:
        import comfy.samplers
        from comfy_extras.nodes_custom_sampler import Guider_Basic, Noise_RandomNoise, SamplerCustomAdvanced
        guider = Guider_Basic(m)
        guider.set_conds(cond)
        refined = _out(SamplerCustomAdvanced.execute(Noise_RandomNoise(int(seed)), guider,
                                                     comfy.samplers.sampler_object("euler"), sigmas, target))[0]
    else:
        refined = sampler_fn(m, cond, sigmas, target, int(seed))
    if save_latent:
        OR.MiniMaxH3_OpenRegenSaveLatent.execute(refined, real, "open_h3_regen", audio)
        lines.append("refined latent saved to output/open_h3_latents/ before decoding")

    # 5. decode, finish, colour lock
    decoded, drep = _decode(refined, decoder, fast_vae_name, vae)
    lines.append(drep)
    kept, sound, frep = OR.finish_frames(decoded, real, "none", audio)
    lines.append(frep)
    if float(colour_lock) > 0:
        ref = _reference_for_lock(draft, plan, real)
        before = CL.mean_colour_error(kept, F.interpolate(ref[..., :3].movedim(-1, 1), size=kept.shape[1:3],
                                                          mode="bilinear", align_corners=False).movedim(1, -1))
        kept = CL.lock(kept, ref, float(colour_lock), method=colour_method)
        after = CL.mean_colour_error(kept, F.interpolate(ref[..., :3].movedim(-1, 1), size=kept.shape[1:3],
                                                         mode="bilinear", align_corners=False).movedim(1, -1))
        lines.append(f"colour lock {float(colour_lock):g} ({colour_method}): mean colour difference to the draft {before:.4f} -> "
                     f"{after:.4f} (low frequencies only; not clamped)")
    lines.append(f"output: {kept.shape[0]} frames {kept.shape[2]}x{kept.shape[1]} px at 24 fps")
    return kept, sound, refined, "\n".join(lines)


def _lora_list():
    names = OR._files("loras")
    return names or ["(put the LoRA files in models/loras)"]


class MiniMaxH3_OpenRegen(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        loras = _lora_list()
        tables = OR._files("diffusion_models")
        upscalers = LU._model_names()
        vaes = OR._files("vae") or ["(put fast_h3_vae_noncausal_24l.safetensors in models/vae)"]
        pick = OR._pick
        regen_inputs = [
            io.Combo.Input("lora_name", options=loras, default=pick(loras, "open_h3_regenerate_2k_lora",
                                                                     "open_h3_regenerate"),
                           tooltip="NVIDIA's merged Open-H3-Regenerate-2K LoRA (refiner + Turbo + LMS Sharp)."),
            io.Combo.Input("curve_name", options=[OR.NO_CURVE] + loras,
                           default=R.pick_default(loras, ("sharp_curve_branches",), OR.NO_CURVE),
                           tooltip="LMS Sharp's 50 AdaLN curve branches (part of the reference recipe)."),
            io.Combo.Input("curve_table", options=[OR.OWN_TABLE] + tables,
                           default=R.pick_default(tables, ("ref2va_pruned_int8_convrot", "ref2va_pruned"), OR.OWN_TABLE),
                           tooltip="adaln_t_table the curve branches read (Comfy-Org ref2va pruned checkpoint)."),
            io.Float.Input("start_sigma", default=0.8, min=0.0, max=1.0, step=0.001,
                           tooltip="0.8 stays close to the draft (upstream workflows). 0 = NVIDIA's release grid "
                                   "0.909 / 0.857 / 0.667, which redraws more."),
            io.Int.Input("steps", default=3, min=1, max=8, tooltip="Euler steps (3 = reference)."),
        ]
        ic_inputs = [
            io.Combo.Input("ic_lora", options=loras,
                           default=R.pick_default(loras, ("open_regenerate_2k_ic", "open_h3_regenerate_2k_ic", "regenerate_2k_ic"), loras[0]),
                           tooltip="The Open-H3-Regenerate-2K-IC LoRA (official Diffusers keys or ComfyUI keys)."),
            io.Combo.Input("speed_lora", options=["none"] + loras, default=R.pick_default(loras, ("hyperflow_8step",), "none"),
                           tooltip="Applied first: HyperFlow 8-step (DMAD 4-step accepted)."),
            io.Combo.Input("sharp_lora", options=["none"] + loras, default=R.pick_default(loras, ("minimax_h3_lms",), "none"),
                           tooltip="Applied second: LMS Sharp r64."),
            io.Combo.Input("ic_grid", options=list(R.SIGMAS), default="up3",
                           tooltip="up3: 3 steps from 0.835 (official default); up4; hf8: 8 steps from 1.0."),
        ]
        return io.Schema(
            node_id="MiniMaxH3_OpenRegen",
            has_intermediate_output=True,  # L9.290: core resends the report when the node is served from cache
            display_name="H3 Open Regen (Regenerate + IC)",
            category=CATEGORY,
            description=("Regenerates a draft video at any size and aspect ratio in one node: Open-H3 Regenerate "
                         "(rdh9616) or Regenerate 2K IC (slmonker). Draft in, finished frames and soundtrack out. "
                         "Load the base model with UNET Loader: FL2VA for Regenerate, Ref2VA for IC."),
            search_aliases=["open h3 regenerate", "regenerate 2k ic", "h3 upscale video", "open regen"],
            inputs=[
                io.Model.Input("model", tooltip="MiniMax H3 base model: FL2VA (Regenerate) or Ref2VA (IC)."),
                io.Clip.Input("clip", tooltip="The MiniMax H3 text encoder."),
                io.Vae.Input("vae", tooltip="MiniMax H3 video VAE (encode, and decode with the core decoder)."),
                io.Vae.Input("audio_vae", tooltip="MiniMax H3 audio VAE (the pinned silence stream)."),
                io.Image.Input("images", tooltip="The draft's frames."),
                io.Float.Input("fps", default=24.0, min=1.0, max=240.0, step=0.001,
                               tooltip="The draft's frame rate (Get Video Components fps); resampled to 24."),
                io.String.Input("prompt", multiline=True, default="",
                                tooltip="What the draft shows. IC adds the official task text in front."),
                io.DynamicCombo.Input("mode", options=[io.DynamicCombo.Option(MODE_REGEN, regen_inputs),
                                                       io.DynamicCombo.Option(MODE_IC, ic_inputs)],
                                      tooltip="Regenerate: rdh9616's recipe. IC: the 2K IC recipe (the draft is "
                                              "<Video 1> of a Reference to Video pass)."),
                io.Combo.Input("size_by", options=list(A.SIZE_BY), default="megapixels",
                               tooltip="megapixels: keep the chosen aspect at this pixel count (3.69 = 2560x1440). "
                                       "width / height: exact size. scale: the draft size times 'scale'... use "
                                       "megapixels for 'keep aspect'."),
                io.Float.Input("megapixels", default=3.69, min=0.3, max=33.0, step=0.01,
                               tooltip="Target pixel count in millions (aspect from 'aspect'). 2K = 3.69, "
                                       "1080p = 2.07, 4K = 8.29.",
                               extra_dict={"c2c_show_if": {"size_by": ["megapixels"]}}),
                io.Combo.Input("aspect", options=list(A.ASPECTS), default="source",
                               tooltip="source: the draft's own ratio. A preset, or custom (custom_ratio).",
                               extra_dict={"c2c_show_if": {"size_by": ["megapixels", "scale"]}}),
                io.String.Input("custom_ratio", default="2.39:1", tooltip="aspect = custom: W:H or a decimal.",
                                extra_dict={"c2c_show_if": {"size_by": ["megapixels", "scale"], "aspect": ["custom"]}}),
                io.Int.Input("width", default=2560, min=64, max=8192, step=32,
                             tooltip="size_by = width / height: output width (32 px grid).",
                             extra_dict={"c2c_show_if": {"size_by": ["width / height"]}}),
                io.Int.Input("height", default=1440, min=64, max=8192, step=32,
                             tooltip="size_by = width / height: output height (32 px grid).",
                             extra_dict={"c2c_show_if": {"size_by": ["width / height"]}}),
                io.Combo.Input("fit", options=list(A.FITS), default="contain",
                               tooltip="When the draft's ratio differs from the target: contain (whole picture), "
                                       "crop (fill, cut at anchor), pad (border regenerated), stretch, auto."),
                io.Combo.Input("anchor", options=list(A.ANCHORS), default="center", advanced=True,
                               tooltip="crop / pad: where the picture sits.",
                               extra_dict={"c2c_show_if": {"fit": ["crop", "pad"]}}),
                io.Combo.Input("upscaler", options=upscalers, default=pick(upscalers, "minimax_h3_latent_upscaler_3d"),
                               tooltip="The H3 3D latent upscaler (models/latent_upscale_models)."),
                io.Int.Input("max_frames", default=0, min=0, max=3600,
                             tooltip="Frames kept after resampling to 24 fps; 0 = all (H3: 5-15 s per pass)."),
                io.Int.Input("seed", default=REGEN_SEED, min=0, max=0xFFFFFFFFFFFFFFFF, control_after_generate=True,
                             tooltip="Refine noise seed."),
                io.Combo.Input("decoder", options=list(DECODERS), default=DECODERS[0],
                               tooltip="core H3 VAE: the released decoder (best fidelity). fast: NVIDIA's distilled "
                                       "24-layer decoder (faster, slightly lossy)."),
                io.Combo.Input("fast_vae_name", options=vaes, default=pick(vaes, "fast_h3_vae_noncausal"),
                               advanced=True, tooltip="decoder = fast: fast_h3_vae_noncausal_24l.safetensors.",
                               extra_dict={"c2c_show_if": {"decoder": [DECODERS[1]]}}),
                io.Float.Input("colour_lock", default=1.0, min=0.0, max=1.0, step=0.05,
                               tooltip="Puts the draft's large-scale colour and brightness back (fine detail kept, "
                                       "nothing clamped). 0 = off."),
                io.Combo.Input("colour_method", options=list(CL.METHODS), default="global", advanced=True,
                               tooltip="global: per-frame colour offset + broad contrast from the draft (no halos). "
                                       "local: also fixes regional drift; a tiny very bright highlight can get a "
                                       "soft halo."),
                io.Combo.Input("upscale_precision", options=["bf16", "fp16", "fp32"], default="bf16", advanced=True,
                               tooltip="Latent upscaler precision (reference: bf16)."),
                io.Combo.Input("lora_dtype", options=["bf16", "fp32"], default="bf16", advanced=True,
                               tooltip="RAM copy of the LoRA factors (bf16 6.2 GB, fp32 12.4 GB for the merged LoRA)."),
                io.Boolean.Input("save_latent", default=False, advanced=True,
                                 tooltip="Save the refined latent to output/open_h3_latents/ before decoding, so a "
                                         "failed decode never costs the refine."),
                io.Audio.Input("audio", optional=True, tooltip="The draft's soundtrack; cut to the kept frames."),
            ],
            outputs=[io.Image.Output("images", tooltip="Regenerated frames at 24 fps, real frames only."),
                     io.Audio.Output("audio", tooltip="The draft's soundtrack cut to the kept frames (None when "
                                                      "no audio is connected)."),
                     io.Latent.Output("latent", tooltip="The refined H3 AV latent (for re-decoding)."),
                     io.String.Output("report", tooltip="Every step's sizes, files, sigmas and the colour lock.")],
        )

    @classmethod
    def execute(cls, model, clip, vae, audio_vae, images, fps, prompt, mode, size_by, megapixels, aspect,
                custom_ratio, width, height, fit, upscaler, max_frames, seed, decoder, colour_lock,
                anchor="center", fast_vae_name="", upscale_precision="bf16", lora_dtype="bf16", save_latent=False,
                audio=None, colour_method="global") -> io.NodeOutput:
        imgs, sound, latent, report = run(model, clip, vae, audio_vae, images, fps, prompt, mode, size_by, width,
                                          height, megapixels, aspect, custom_ratio, fit, anchor, upscaler,
                                          max_frames, seed, decoder, fast_vae_name, colour_lock, upscale_precision,
                                          save_latent, audio, lora_dtype, colour_method=colour_method)
        LOG.info(report)
        return io.NodeOutput(imgs, sound, latent, report, ui={"text": [report]})


NODES = [MiniMaxH3_OpenRegen]
