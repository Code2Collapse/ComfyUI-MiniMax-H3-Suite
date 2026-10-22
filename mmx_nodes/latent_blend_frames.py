# PORTED FROM: https://github.com/xyzDist/H3-LongTakeNoCuts@556085d h3_blend_latents_by_frames.py, keyframe_mask.py,
# mask_level.py (MIT, Copyright (c) 2026 xyzDist). Changed here: a pixel-frame keyframe maps to the latent step that
# holds it (H3's 1,4,4,4,4 grouping) instead of round(p * T / frames), so the frame count comes from the latent and the
# duration / fps inputs are gone; the optional mask is reduced on H3's grid; smooth interpolation is also offered for
# the keyframe mask.
"""Against the "photocopy effect" of long chained H3 shots (detail and colour drift from segment 5-6 on): run a short
refine pass (denoise ~0.5) on the new segment, then blend the refined latent back in BY FRAMES - the frames that join
the previous segment stay as they are, later frames take the refined detail.

* H3 Blend Latents By Frames: per-frame blend weight from keyframes ("0:0, 22:0, 44:1"), optional mask.
* H3 Keyframe Mask: a per-frame mask from keyframes, e.g. for the refine pass's noise mask.
* H3 Mask Levels: black / white / gamma on a mask (black = the least the mask can be, white = the most).
"""

from __future__ import annotations

import math
import re
import sys
from pathlib import Path

import torch
from comfy_api.latest import io

_PKG = Path(__file__).resolve().parents[1]
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from mmx_nodes.av_noise_mask import _streams, pixel_frames, step_offsets  # noqa: E402
from mmx_utils import h3_geometry as geo  # noqa: E402
from mmx_utils.h3_constants import FRAME_PER_TOKEN  # noqa: E402

CATEGORY = "MiniMax H3/Long"


def _nested(parts):
    import comfy.nested_tensor  # noqa: WPS433
    return comfy.nested_tensor.NestedTensor(tuple(parts))


def parse_keyframes(text: str) -> list[tuple[int, float]]:
    pairs = []
    for part in str(text or "").split(","):
        m = re.match(r"\s*(-?\d+)\s*:\s*(-?[0-9]*\.?[0-9]+)\s*$", part)
        if part.strip() and not m:
            raise ValueError(f"cannot read keyframe {part.strip()!r}; write frame:value, e.g. 0:0, 22:0, 44:1")
        if m:
            pairs.append((int(m.group(1)), float(m.group(2))))
    if not pairs:
        raise ValueError("no keyframes given; write frame:value pairs, e.g. 0:0, 22:0, 44:1")
    out: dict[int, float] = {}
    for f, v in sorted(pairs):
        out[f] = v                     # a repeated frame: the last value wins
    return sorted(out.items())


def curve(pairs: list[tuple[int, float]], frames: int, interpolation: str = "linear") -> list[float]:
    """Value at every frame 0..frames-1: held before the first and after the last keyframe."""
    vals = []
    for f in range(frames):
        if f <= pairs[0][0]:
            v = pairs[0][1]
        elif f >= pairs[-1][0]:
            v = pairs[-1][1]
        else:
            j = max(i for i, (kf, _) in enumerate(pairs) if kf <= f)
            (f0, v0), (f1, v1) = pairs[j], pairs[j + 1]
            t = (f - f0) / (f1 - f0)
            if interpolation == "hold":
                t = 0.0
            elif interpolation == "smooth":
                t = 0.5 * (1 - math.cos(math.pi * t))
            v = v0 + (v1 - v0) * t
        vals.append(v)
    return vals


def step_weights(frame_w: list[float], latent_t: int) -> list[float]:
    """Each latent step takes the mean weight of the pixel frames it holds."""
    offs = step_offsets(latent_t)
    return [sum(frame_w[o:o + FRAME_PER_TOKEN[k % 5]]) / FRAME_PER_TOKEN[k % 5] for k, o in enumerate(offs)]


def _latent_mask(mask, t, h, w):
    """Frame-space mask -> soft [T, h, w] on H3's grid (mean per latent step and per 16 px latent pixel)."""
    from mmx_nodes.mask_to_latent import MiniMaxH3_MaskToLatentSpace, _pattern_geometry  # noqa: WPS433

    m = mask.float()
    if m.ndim == 2:
        m = m[None]
    if tuple(m.shape) == (t, h, w):
        return m.clamp(0, 1)
    frames = pixel_frames(t)
    if m.shape[0] == 1:
        m = m.expand(frames, -1, -1)
    if m.shape[0] != frames or (m.shape[1], m.shape[2]) != (h * 16, w * 16):
        raise ValueError(f"the mask must be {frames} frames of {w * 16}x{h * 16} px (or a latent-size mask), got "
                         f"{tuple(m.shape)}")
    spatial, groups, _ = _pattern_geometry(16, list(FRAME_PER_TOKEN))
    return MiniMaxH3_MaskToLatentSpace._reduce(m, spatial, groups, 1, "mean", "mean", 0, 0).clamp(0, 1)


def blend_by_frames(latent1, latent2, keyframes, interpolation="linear", audio="latent 1", mask=None):
    what = "H3 Blend Latents By Frames"
    v1, a1 = _streams(latent1, what)
    v2, a2 = _streams(latent2, what)
    if v1.shape != v2.shape or a1.shape != a2.shape:
        raise ValueError(f"{what}: the latents differ: {tuple(v1.shape)}/{tuple(a1.shape)} vs {tuple(v2.shape)}/"
                         f"{tuple(a2.shape)}. Blend two passes of the same segment.")
    t, h, w = int(v1.shape[2]), int(v1.shape[3]), int(v1.shape[4])
    frames = pixel_frames(t)
    fw = [min(1.0, max(0.0, x)) for x in curve(parse_keyframes(keyframes), frames, interpolation)]
    sw = step_weights(fw, t)
    wt = torch.tensor(sw, dtype=torch.float32).view(1, 1, t, 1, 1)
    if mask is not None:
        wt = wt * _latent_mask(mask, t, h, w)[None, None]
    wt = wt.to(v1.device, v1.dtype)
    v = v1 * (1 - wt) + v2.to(v1.device, v1.dtype) * wt
    ticks = int(a1.shape[-1])
    if audio == "blend":
        tw = torch.tensor([fw[min(frames - 1, int(k * geo.FPS / geo.AUDIO_TICKS_PER_SECOND))] for k in range(ticks)],
                          dtype=a1.dtype, device=a1.device)
        a = a1 * (1 - tw) + a2.to(a1.device, a1.dtype) * tw
    else:
        a = a1 if audio == "latent 1" else a2
    out = dict(latent2)
    out["samples"] = _nested((v, a.clone()))
    out.pop("noise_mask", None)
    first = next((i for i, x in enumerate(fw) if x > 0), None)
    report = (f"{frames} frames ({t} latent steps): weight of latent 2 from {fw[0]:.2f} to {fw[-1]:.2f}; latent 2 "
              + (f"starts at frame {first}" if first is not None else "never used")
              + f"; audio: {audio}" + ("; masked" if mask is not None else ""))
    return out, report


class MiniMaxH3_BlendLatentsByFrames(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MiniMaxH3_BlendLatentsByFrames",
            display_name="H3 Blend Latents By Frames",
            category=CATEGORY,
            description=("Blend two latents of the same segment frame by frame: keyframes give latent 2's weight per "
                         "pixel frame (0 = all latent 1, 1 = all latent 2), e.g. 0:0, 22:0, 44:1 keeps the first 22 "
                         "frames that join the previous segment and fades into a refined pass by frame 44 - the "
                         "LongTakeNoCuts fix for detail loss over long chained shots. A mask limits where latent 2 is "
                         "used (white = latent 2 allowed). Removes any noise mask."),
            inputs=[
                io.Latent.Input("latent1", tooltip="E.g. the segment as generated."),
                io.Latent.Input("latent2", tooltip="E.g. the same segment after a short refine pass."),
                io.String.Input("keyframes", default="0:0, 22:0, 44:1",
                                tooltip="frame:weight pairs (0-based pixel frames). Held before the first and after "
                                        "the last."),
                io.Combo.Input("interpolation", options=["linear", "smooth", "hold"], default="linear"),
                io.Combo.Input("audio", options=["latent 1", "latent 2", "blend"], default="latent 1",
                               tooltip="Which sound to keep; blend follows the same weights."),
                io.Mask.Input("mask", optional=True,
                              tooltip="Where latent 2 may be used (frames at the canvas size, or latent size)."),
            ],
            outputs=[io.Latent.Output(display_name="latent"), io.String.Output(display_name="report")],
        )

    @classmethod
    def execute(cls, latent1, latent2, keyframes, interpolation="linear", audio="latent 1", mask=None):
        try:
            out, report = blend_by_frames(latent1, latent2, keyframes, interpolation, audio, mask)
        except ValueError as exc:
            raise ValueError(f"H3 Blend Latents By Frames: {exc}") from exc
        return io.NodeOutput(out, report, ui={"text": [report]})


def keyframe_mask(keyframes, frame_count, width, height, interpolation="linear"):
    vals = [min(1.0, max(0.0, v)) for v in curve(parse_keyframes(keyframes), int(frame_count), interpolation)]
    return torch.tensor(vals, dtype=torch.float32).view(-1, 1, 1).expand(-1, int(height), int(width)).contiguous()


class MiniMaxH3_KeyframeMask(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MiniMaxH3_KeyframeMask",
            display_name="H3 Keyframe Mask",
            category="MiniMax H3/Masking",
            description=("A mask whose value changes over time, from frame:value keyframes (e.g. 15:1, 22:0, 30:1); "
                         "every pixel of a frame has the frame's value. Use it as a time-varying denoise mask (H3 Set "
                         "AV Noise Mask) or to fade a blend."),
            inputs=[
                io.String.Input("keyframes", default="15:1, 22:0, 30:1", tooltip="frame:value pairs (0-based)."),
                io.Int.Input("frame_count", default=124, min=1, max=100000),
                io.Int.Input("width", default=768, min=16, max=16384),
                io.Int.Input("height", default=768, min=16, max=16384),
                io.Combo.Input("interpolation", options=["linear", "smooth", "hold"], default="linear"),
                io.Image.Input("size_reference", optional=True,
                               tooltip="Frames whose size and count to use instead of the numbers above."),
            ],
            outputs=[io.Mask.Output(display_name="mask")],
        )

    @classmethod
    def execute(cls, keyframes, frame_count, width, height, interpolation="linear", size_reference=None):
        if size_reference is not None:
            frame_count, height, width = (int(x) for x in size_reference.shape[:3])
        try:
            return io.NodeOutput(keyframe_mask(keyframes, frame_count, width, height, interpolation))
        except ValueError as exc:
            raise ValueError(f"H3 Keyframe Mask: {exc}") from exc


def mask_levels(mask, black_level, white_level, gamma):
    m = mask.float().clamp(0, 1)
    if gamma != 1.0:
        m = torch.pow(m.clamp(min=1e-6), 1.0 / gamma)
    return (black_level + (white_level - black_level) * m).clamp(0, 1)


class MiniMaxH3_MaskLevels(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MiniMaxH3_MaskLevels",
            display_name="H3 Mask Levels",
            category="MiniMax H3/Masking",
            description=("Output levels for a mask: black becomes black_level, white becomes white_level, gamma bends "
                         "the middle. For a denoise mask, black_level is the least any area is regenerated and "
                         "white_level the most (e.g. 0.3 / 0.6 for a refine pass)."),
            inputs=[
                io.Mask.Input("mask"),
                io.Float.Input("black_level", default=0.0, min=0.0, max=1.0, step=0.01),
                io.Float.Input("white_level", default=1.0, min=0.0, max=1.0, step=0.01),
                io.Float.Input("gamma", default=1.0, min=0.1, max=5.0, step=0.05),
            ],
            outputs=[io.Mask.Output(display_name="mask")],
        )

    @classmethod
    def execute(cls, mask, black_level, white_level, gamma):
        return io.NodeOutput(mask_levels(mask, black_level, white_level, gamma))


__all__ = ["MiniMaxH3_BlendLatentsByFrames", "MiniMaxH3_KeyframeMask", "MiniMaxH3_MaskLevels", "blend_by_frames",
           "keyframe_mask", "mask_levels", "curve", "step_weights", "parse_keyframes"]
