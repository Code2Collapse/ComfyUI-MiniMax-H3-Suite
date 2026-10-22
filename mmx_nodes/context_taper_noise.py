# PORTED FROM: https://github.com/beijinren/ComfyUI-H3-Context-Noise@7e55312 nodes.py (MIT, Copyright (c) 2026 beijinren)
# Behaviour unchanged; V3 schema, suite node ids / category and plain-English errors added here.
"""MiniMaxH3_ContextTaperNoise / MiniMaxH3_ContextLatentTaperNoise - corrupt the carried context before a chained clip.

A clean context makes H3 copy the previous clip's exact texture and colour, which shows as chroma residue right after
the trim boundary. Corrupting the carried frames keeps the motion readable while the model has to re-synthesise the
appearance from the identity references. Two paths, the two ways Motion Context takes a context:

* frames:  IMAGE -> IMAGE, blocky chroma noise on the last frames (Motion Context ``context_frames``);
* latent:  LATENT -> LATENT, gaussian noise on the last latent steps, the audio stream untouched
           (Motion Context ``context_latent``, which wins over ``context_frames`` when both are wired).

Experimental: an empirical recipe validated upstream on character-swap chains, not on this suite's own workflows.
Not for H3 Continuation: its pinned head must come back unchanged (Commit checks that).
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import torch
from comfy_api.latest import io

_PKG = Path(__file__).resolve().parents[1]
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from mmx_utils.context_taper import (  # noqa: E402
    LATENT_SAFE_RUNS,
    POC_BLOCK,
    VIDEO_RUN_GRID,
    TaperError,
    alpha_for,
    format_schedule,
    nearest_index_map,
    new_rng,
    palette_grid,
    resolve_grid,
    steps_for_frames,
    validate,
)

_LOG = logging.getLogger("C2C.mmx.context_taper")
_MAX_SEED = 0xFFFFFFFFFFFFFFFF
PATTERNS = ("poc_chroma_blocks", "gaussian_rgb", "uniform_rgb")


def _streams(latent):
    """(parts, container) of an H3 AV latent; a plain single-stream latent is refused (unbind would split the batch)."""
    if not isinstance(latent, dict) or "samples" not in latent:
        raise ValueError("expected a LATENT with 'samples'")
    samples = latent["samples"]
    if isinstance(samples, torch.Tensor):
        raise ValueError("this is a plain single-stream latent, not an H3 video/audio pair. Wire Motion Context Load "
                         "Latent's output, or an H3 sampler's output.")
    if isinstance(samples, (list, tuple)):
        return list(samples), list
    if hasattr(samples, "unbind"):
        return list(samples.unbind()), type(samples)
    raise ValueError(f"unsupported latent payload {type(samples).__name__}")


def _repack(latent, parts, container):
    out = dict(latent)
    out["samples"] = list(parts) if container is list else container(parts)
    return out


def _block_noise(pattern, grid_w, grid_h, yi, xi, rng, gen):
    """One frame of block noise, float32 [H, W, 3] in 0..1, nearest-upscaled (no interpolation softens the blocks)."""
    if pattern == "poc_chroma_blocks":
        small = torch.tensor(palette_grid(rng, grid_w, grid_h), dtype=torch.float32).div_(255.0)
    elif pattern == "gaussian_rgb":
        small = torch.randn((grid_h, grid_w, 3), generator=gen, dtype=torch.float32).mul_(0.25).add_(0.5).clamp_(0, 1)
    elif pattern == "uniform_rgb":
        small = torch.rand((grid_h, grid_w, 3), generator=gen, dtype=torch.float32)
    else:
        raise ValueError(f"unknown pattern {pattern!r}")
    return small.index_select(0, yi).index_select(1, xi)


def taper_frames(images, tail_frames, alpha, alpha_end, ramp_frames, seed, pattern="poc_chroma_blocks",
                 grid_mode="poc_36x64", block_size=POC_BLOCK):
    if not isinstance(images, torch.Tensor) or images.ndim != 4 or int(images.shape[3]) < 3:
        raise ValueError(f"expected an IMAGE batch [B,H,W,3+], got {tuple(getattr(images, 'shape', ()))}")
    available = int(images.shape[0])
    n = min(int(tail_frames), available)
    ramp = min(int(ramp_frames), n)
    validate(n, float(alpha), float(alpha_end), ramp)
    height, width = int(images.shape[1]), int(images.shape[2])
    grid_w, grid_h = resolve_grid(width, height, grid_mode, block_size)
    rng = new_rng(seed)
    gen = torch.Generator(device="cpu").manual_seed(int(seed) & _MAX_SEED)
    yi = torch.tensor(nearest_index_map(height, grid_h), dtype=torch.long)
    xi = torch.tensor(nearest_index_map(width, grid_w), dtype=torch.long)
    out = images.clone()
    start = available - n
    alphas = []
    for position in range(n):
        amount = alpha_for(position, n, float(alpha), float(alpha_end), ramp)
        alphas.append(amount)
        noise = _block_noise(pattern, grid_w, grid_h, yi, xi, rng, gen).to(device=out.device, dtype=out.dtype)
        frame = out[start + position, :, :, :3]
        out[start + position, :, :, :3] = frame * (1.0 - amount) + noise * amount
    header = (f"H3 context taper noise (frames)\n  frames {start}..{available - 1} of {available} injected, "
              f"{start} clean before them\n  pattern={pattern} grid={grid_w}x{grid_h} "
              f"(~{width / grid_w:.1f}x{height / grid_h:.1f} px blocks) seed={int(seed)}")
    if n < int(tail_frames):
        header += f"\n  only {available} frames arrived: {n} injected instead of {int(tail_frames)}"
    if n not in VIDEO_RUN_GRID:
        header += (f"\n  note: {n} is not a Motion Context window ({', '.join(map(str, LATENT_SAFE_RUNS))}); fine as "
                   "long as it covers context_length")
    return out, format_schedule(alphas, unit="frame", header=header)


def taper_latent(context_latent, tail_frames, alpha, alpha_end, ramp_steps, seed, noise_scale="match_latent_std"):
    frames = int(tail_frames)
    steps = steps_for_frames(frames)
    if steps is None:
        raise ValueError(f"{frames} frames is not a whole number of latent steps. Use one of "
                         f"{', '.join(map(str, LATENT_SAFE_RUNS))}.")
    parts, container = _streams(context_latent)
    video = parts[0]
    if not isinstance(video, torch.Tensor):
        raise ValueError(f"the video stream is {type(video).__name__}, not a tensor")
    unbatched = video.ndim == 4
    if unbatched:
        video = video.unsqueeze(0)
    if video.ndim != 5:
        raise ValueError(f"expected a video latent [B,C,T,H,W], got {tuple(video.shape)}")
    total = int(video.shape[2])
    if steps > total:
        raise ValueError(f"asked for the last {steps} latent steps ({frames} frames) but the latent has only {total}.")
    start = total - steps
    ramp = min(int(ramp_steps), steps)
    validate(steps, float(alpha), float(alpha_end), ramp)
    out = video.clone()
    scale = 1.0
    if noise_scale == "match_latent_std":
        scale = float(out[:, :, start:].detach().float().std().item())
        if scale == 0.0 or scale != scale:
            scale = 1.0
    gen = torch.Generator(device="cpu").manual_seed(int(seed) & _MAX_SEED)
    alphas = []
    for k in range(steps):
        amount = alpha_for(k, steps, float(alpha), float(alpha_end), ramp)
        alphas.append(amount)
        block = out[:, :, start + k]
        noise = torch.randn(tuple(block.shape), generator=gen, dtype=torch.float32).mul_(scale)
        out[:, :, start + k] = block * (1.0 - amount) + noise.to(device=block.device, dtype=block.dtype) * amount
    parts[0] = out[0] if unbatched else out
    header = (f"H3 context taper noise (latent)\n  steps {start}..{total - 1} of {total} injected ({frames} pixel "
              f"frames), audio untouched\n  noise_scale={noise_scale} (sigma={scale:.4f}) seed={int(seed)}")
    if start % 5:
        header += (f"\n  warning: the tail starts at cycle position {start % 5}, not 0 - Motion Context refuses this "
                   "window; fix the clip length or tail_frames")
    return _repack(context_latent, parts, container), format_schedule(alphas, unit="step", header=header)


_TAIL_TIP = ("How many frames at the END get noise. Motion Context pins the LAST context_length frames, so keep this at "
             "least context_length. Earlier frames pass through untouched.")
_ALPHA_TIP = "Flat noise strength over the window (0.45 validated). Lower keeps more of the previous clip's look."
_END_TIP = "Strength at the frame touching the seam. Not larger than alpha."


class MiniMaxH3_ContextTaperNoise(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MiniMaxH3_ContextTaperNoise",
            display_name="H3 Context Taper Noise (frames, experimental)",
            category="MiniMax H3/Long",
            description=("Blocky chroma noise on the tail of the previous clip before it reaches Motion Context's "
                         "context_frames, ramping down towards the seam, so H3 keeps the motion but stops copying the "
                         "old colour (character-swap chains). The output is only a context: never chain it on as the "
                         "next clip's source. Experimental (validated upstream, not on your graphs yet)."),
            inputs=[
                io.Image.Input("images", tooltip="The previous clip's decoded frames (what would go to context_frames)."),
                io.Int.Input("tail_frames", default=22, min=1, max=4096, tooltip=_TAIL_TIP),
                io.Float.Input("alpha", default=0.45, min=0.0, max=1.0, step=0.01, tooltip=_ALPHA_TIP),
                io.Float.Input("alpha_end", default=0.10, min=0.0, max=1.0, step=0.01, tooltip=_END_TIP),
                io.Int.Input("ramp_frames", default=3, min=1, max=4096,
                             tooltip="Frames at the end that ramp from alpha to alpha_end (3 validated: 0.333/0.217/0.100)."),
                io.Int.Input("seed", default=0, min=0, max=_MAX_SEED, control_after_generate=True,
                             tooltip="Noise pattern seed; one stream for every frame, so the whole tail is reproducible."),
                io.Combo.Input("pattern", options=list(PATTERNS), default=PATTERNS[0], optional=True,
                               tooltip="poc_chroma_blocks = the validated 6-colour palette. The other two are untested."),
                io.Combo.Input("grid_mode", options=["poc_36x64", "block_size"], default="poc_36x64", optional=True,
                               tooltip="poc_36x64 reproduces the recipe exactly (16 px blocks only at 576x1024). "
                                       "block_size keeps the block size constant at any resolution."),
                io.Int.Input("block_size", default=POC_BLOCK, min=1, max=256, optional=True,
                             tooltip="Block size in pixels for grid_mode = block_size."),
            ],
            outputs=[io.Image.Output(display_name="images"), io.String.Output(display_name="alpha_schedule")],
        )

    @classmethod
    def execute(cls, images, tail_frames, alpha, alpha_end, ramp_frames, seed, pattern="poc_chroma_blocks",
                grid_mode="poc_36x64", block_size=POC_BLOCK) -> io.NodeOutput:
        try:
            out, text = taper_frames(images, tail_frames, alpha, alpha_end, ramp_frames, seed, pattern, grid_mode,
                                     block_size)
        except (TaperError, ValueError) as exc:
            raise ValueError(f"H3 Context Taper Noise (frames): {exc}") from exc
        return io.NodeOutput(out, text, ui={"text": [text]})


class MiniMaxH3_ContextLatentTaperNoise(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MiniMaxH3_ContextLatentTaperNoise",
            display_name="H3 Context Taper Noise (latent, experimental)",
            category="MiniMax H3/Long",
            description=("Gaussian noise on the last latent steps of the previous clip's AV latent, ramping down "
                         "towards the seam, for Motion Context's context_latent (which wins over context_frames). The "
                         "audio stream is untouched so the join's sound stays continuous. Only for context_latent: do "
                         "not decode it. Experimental."),
            inputs=[
                io.Latent.Input("context_latent", tooltip="The previous clip's AV latent (Motion Context Load Latent)."),
                io.Combo.Input("tail_frames", options=[str(v) for v in LATENT_SAFE_RUNS], default="22",
                               tooltip="Pixel frames to noise; only these are whole latent steps (2/7/12/17). Use the "
                                       "same value as Motion Context's context_length."),
                io.Float.Input("alpha", default=0.45, min=0.0, max=1.0, step=0.01, tooltip=_ALPHA_TIP),
                io.Float.Input("alpha_end", default=0.10, min=0.0, max=1.0, step=0.01, tooltip=_END_TIP),
                io.Int.Input("ramp_steps", default=2, min=1, max=64,
                             tooltip="Latent steps at the end that ramp down (one step is up to 4 frames)."),
                io.Int.Input("seed", default=0, min=0, max=_MAX_SEED, control_after_generate=True),
                io.Combo.Input("noise_scale", options=["match_latent_std", "unit"], default="match_latent_std",
                               optional=True,
                               tooltip="match_latent_std scales the noise to the window's own spread, so alpha means "
                                       "the same at any latent scale. unit = plain N(0, 1)."),
            ],
            outputs=[io.Latent.Output(display_name="context_latent"), io.String.Output(display_name="alpha_schedule")],
        )

    @classmethod
    def execute(cls, context_latent, tail_frames, alpha, alpha_end, ramp_steps, seed,
                noise_scale="match_latent_std") -> io.NodeOutput:
        try:
            out, text = taper_latent(context_latent, tail_frames, alpha, alpha_end, ramp_steps, seed, noise_scale)
        except (TaperError, ValueError) as exc:
            raise ValueError(f"H3 Context Taper Noise (latent): {exc}") from exc
        return io.NodeOutput(out, text, ui={"text": [text]})


__all__ = ["MiniMaxH3_ContextTaperNoise", "MiniMaxH3_ContextLatentTaperNoise", "taper_frames", "taper_latent"]
