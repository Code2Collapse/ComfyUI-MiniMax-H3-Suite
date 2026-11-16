# PORTED FROM: https://github.com/rdh9616/ComfyUI-Open-H3-Regenerate@46ff778 regen_core.py (Apache-2.0)
# Copyright 2026 rdh9616. Licensed under the Apache License, Version 2.0 (THIRD_PARTY_LICENSES/
# ComfyUI-Open-H3-Regenerate-LICENSE.txt, NOTICE in ComfyUI-Open-H3-Regenerate-NOTICE.txt).
# Modified by Code2Collapse, 2026-11: V3 schema, Suite upscaler engine, comfy.audio resample, mmap-safe cache.
# Distributed as part of a GPL-3.0 work.
# Changes in this file: the 17k+5 frame maths delegate to the Suite's mmx_utils/av_latent_ops (one implementation);
# frames_for_latent_t refuses a latent length that is not 5k+2 (upstream rounded it); resample_indices refuses a
# non-finite or non-positive fps; draft_size / upscale_passes collect node arithmetic so it is testable without ComfyUI.
"""ComfyUI-free maths of Open-H3 Regenerate (spec docs/research/H3_REGEN_STD_spec.md; shared with the IC pipeline).

- refine sigma grids (NVlabs open_h3/sigmas.py): Euler, CFG 1, model time 1 - sigma
- Diffusers -> ComfyUI key mapping for the merged refiner + Turbo + LMS Sharp LoRA
- the LMS Sharp curve branches: delta(t) = B @ A @ lerp(adaln_t_table, t), added to every block's AdaLN projection
- draft-frame and output-size helpers (H3 frames come in 17k + 5 steps; latent frames = 5k + 2)
Only torch is imported here.
"""
from __future__ import annotations

import math
import re

import torch

from . import av_latent_ops as _av

REFINE_SHIFT = 6.0
SCHEDULES = {3: [0.909375, 0.8571428571428571, 0.6666666666666666, 0.0]}

# draft sizes (W, H) the reference encodes a video at
INPUT_SIZES = {"768p": (1344, 768), "576p": (1024, 576)}
# output size -> (computed width, computed height, kept height): 1080P / 4K run on whole latent patches (1088 / 2176
# rows) and the centre 1080 / 2160 rows are kept
RESOLUTIONS = {"768p": (1344, 768, 768), "1080p": (1920, 1088, 1080), "2k": (2560, 1440, 1440),
               "4k": (3840, 2176, 2160)}
UPSCALER_MAX = RESOLUTIONS["2k"]       # one upscaler pass stays within its trained range; 4K goes through 2K first
CROP_TRIM = {1088: 1080, 2176: 2160}

# sha256 of the 1025 x 8 adaln_t_table the reference reads (Comfy-Org minimax_h3_ref2va_pruned_int8_convrot)
REFERENCE_TABLE_SHA256 = "c02a6c11888297688c1e6278185ea1f947023acfc69f9003bbcdcec9a229a8e7"
FPS = 24.0


# ------------------------------------------------------------------ sigmas

def shifted_grid(start, steps, shift=REFINE_SHIFT):
    """Euler grid from `start` to 0: even spacing in unshifted sigma, then s*x / (1 + (s - 1)*x)."""
    raw = start / (shift - (shift - 1) * start)
    return [start] + [shift * x / (1 + (shift - 1) * x) for x in (raw * (1 - i / steps) for i in range(1, steps))] + [0.0]


SCHEDULES[2] = shifted_grid(0.7, 2)


def refine_sigmas(steps, start_sigma=0.0):
    """The release grid for `steps` (3 or 2), or the shift-6 grid from `start_sigma` when it is > 0."""
    steps = int(steps)
    if steps < 1:
        raise ValueError("steps must be at least 1")
    if start_sigma and start_sigma > 0:
        if not math.isfinite(float(start_sigma)) or float(start_sigma) > 1.0:
            raise ValueError(f"start_sigma must be in (0, 1], got {start_sigma}")
        return shifted_grid(float(start_sigma), steps)
    if steps not in SCHEDULES:
        raise ValueError(f"steps must be {' or '.join(str(s) for s in sorted(SCHEDULES, reverse=True))} without a "
                         f"start_sigma (got {steps}); set start_sigma > 0 (e.g. 0.8) for other step counts")
    return list(SCHEDULES[steps])


# ------------------------------------------------------------------ frames

def align_frame_count(n):
    """H3 clip lengths are 17k + 5 frames at 24 fps (121 -> 124 = 5 s, 362 = 15 s)."""
    return _av.align_frame_count(n)


def video_latent_t(frame_count):
    return _av.video_latent_t(frame_count)


def audio_latent_t(frame_count):
    return _av.audio_latent_t(frame_count)


def frames_for_latent_t(latent_t):
    """17k+5 frame count of a 5k+2 latent; anything else is not an H3 clip length and is refused."""
    return _av.frame_count_from_video_t(latent_t)


def resample_indices(count, fps, target=FPS):
    """Source frame per output frame for a `count`-frame clip at `fps` resampled to `target` (ffmpeg fps filter:
    output frame i shows the source frame nearest to time i / target). Round, not floor."""
    fps = float(fps)
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError(f"source_fps must be a positive number, got {fps}")
    count = int(count)
    if count < 1:
        raise ValueError("no frames to resample")
    out = max(1, int(round(count / fps * target)))
    return [min(count - 1, int(round(i * fps / target))) for i in range(out)]


def cover_crop_box(src_w, src_h, dst_w, dst_h):
    """(scaled_w, scaled_h, x0, y0): scale to cover dst (ffmpeg force_original_aspect_ratio=increase), centre crop."""
    scale = max(dst_w / src_w, dst_h / src_h)
    sw, sh = max(dst_w, round(src_w * scale)), max(dst_h, round(src_h * scale))
    return sw, sh, (sw - dst_w) // 2, (sh - dst_h) // 2


def orient(size, width, height):
    """A landscape (W, H) preset turned to the draft's orientation."""
    w, h = size
    return (h, w) if height > width else (w, h)


def draft_size(preset, src_w, src_h):
    """(W, H) of the prepared draft: a preset turned to the draft's orientation, or 'keep' = each side to 32 px."""
    if preset == "keep":
        if src_w < 16 or src_h < 16:
            raise ValueError(f"the draft is {src_w}x{src_h} px; 'keep' needs both sides of at least 16 px "
                             "(each side is rounded to 32 px)")
        return max(32, round(src_w / 32) * 32), max(32, round(src_h / 32) * 32)
    if preset not in INPUT_SIZES:
        raise ValueError(f"size must be one of {', '.join(INPUT_SIZES)} or keep, got {preset!r}")
    return orient(INPUT_SIZES[preset], src_w, src_h)


def output_size(resolution, width, height):
    """(computed W, computed H, kept W, kept H) for a draft of width x height."""
    if resolution not in RESOLUTIONS:
        raise ValueError(f"resolution must be one of {', '.join(RESOLUTIONS)}, got {resolution!r}")
    w, h, keep = RESOLUTIONS[resolution]
    if height > width:
        return h, w, keep, w
    return w, h, w, keep


def upscale_passes(lat_w, lat_h, resolution):
    """[(W, H), ...] target sizes in px for the upscaler passes, plus (computed W, H, kept W, H).
    A target above 2K (oriented) from an input below it goes through 2K first (upstream rule)."""
    tw, th, kw, kh = output_size(resolution, lat_w * 16, lat_h * 16)
    cap_w, cap_h = orient(UPSCALER_MAX[:2], tw, th)
    steps = []
    if th * tw > cap_w * cap_h and lat_h * lat_w * 256 < cap_w * cap_h:
        steps.append((cap_w, cap_h))
    steps.append((tw, th))
    return steps, (tw, th, kw, kh)


def finish_crop(h, w):
    """(top, left, out_h, out_w) of the auto crop of the latent-grid overhang (1088 -> 1080, 2176 -> 2160)."""
    if h in CROP_TRIM and w > h:
        return (h - CROP_TRIM[h]) // 2, 0, CROP_TRIM[h], w
    if w in CROP_TRIM and h > w:
        return 0, (w - CROP_TRIM[w]) // 2, h, CROP_TRIM[w]
    return 0, 0, h, w


def decoded_frames(latent_t):
    """Frames the H3 decoders give for T latent steps: the first latent of every 5-latent block gives 1 frame."""
    latent_t = int(latent_t)
    return 4 * latent_t - 3 * ((latent_t + 4) // 5)


# ------------------------------------------------------------------ LoRA keys

_BLOCK = re.compile(r"^transformer\.(transformer_blocks|token_refiner\.refiner_blocks)\.(\d+)\.(.+)$")
_TARGET = {
    "attn.to_q": ("attn.qkv_proj", 0),
    "attn.to_k": ("attn.qkv_proj", 1),
    "attn.to_v": ("attn.qkv_proj", 2),
    "attn.to_out.0": ("attn.out_proj", None),
    "ff.net.0.proj": ("mlp.fc1", "swiglu"),
    "ff.net.2": ("mlp.fc2", None),
}


def comfy_target(module):
    """Diffusers module name ('transformer.transformer_blocks.3.attn.to_q') ->
    (ComfyUI weight key, qkv slot or 'swiglu' or None)."""
    match = _BLOCK.match(module)
    if match is None or match.group(3) not in _TARGET:
        raise KeyError(f"unexpected LoRA module {module}")
    stack, index, leaf = match.groups()
    root = "blocks" if stack == "transformer_blocks" else "token_refiner.blocks"
    name, slot = _TARGET[leaf]
    return f"diffusion_model.{root}.{index}.{name}.weight", slot


def swap_swiglu_halves(b):
    """LoRA B of ff.net.0.proj: Diffusers rows [value; gate] -> ComfyUI fc1 rows [gate; value]."""
    half = b.shape[0] // 2
    return torch.cat([b[half:], b[:half]], dim=0)


def lora_modules(keys):
    """Module names of a PEFT LoRA ('<module>.lora_A.weight' / '.lora_B.weight')."""
    names = sorted(k[:-len(".lora_A.weight")] for k in keys if k.endswith(".lora_A.weight"))
    missing = [n for n in names if f"{n}.lora_B.weight" not in keys]
    if missing or len(names) * 2 != len(keys) or not names:
        raise ValueError("expected only lora_A / lora_B pairs (a Diffusers PEFT LoRA)"
                         + (f", missing B for {missing[:3]}" if missing else ""))
    return names


# ------------------------------------------------------------------ Sharp curve branches

_CURVE = re.compile(r"^transformer\.transformer_blocks\.(\d+)\.adaln_proj\.linear\.curve_A\.weight$")


def curve_blocks(keys):
    """{block index: (curve_A key, curve_B key)} of a sharp_curve_branches file."""
    out = {}
    for key in keys:
        match = _CURVE.match(key)
        if match:
            out[int(match.group(1))] = (key, key.replace(".curve_A.", ".curve_B."))
    return out


def table_coords(table, t):
    """Interpolate the 8-D curve at model time t in [0, 1] (rows of `table` span t = 0..1)."""
    t = t.to(torch.float32).clamp(0.0, 1.0)
    pos = t * (table.shape[0] - 1)
    lo = pos.floor().long().clamp(max=table.shape[0] - 2)
    return torch.lerp(table[lo], table[lo + 1], (pos - lo).unsqueeze(-1))


def invert_table(table, coords):
    """Model time t of each row of `coords`, points on the piecewise-linear curve `table` (as ComfyUI's curve-form
    H3 computes its AdaLN input).  Returns (t, max distance from the curve)."""
    table = table.to(coords.device, torch.float32)
    coords = coords.to(torch.float32)
    p0, d = table[:-1], table[1:] - table[:-1]                       # segments [S, C]
    rel = coords[:, None, :] - p0[None]                               # [M, S, C]
    s = ((rel * d[None]).sum(-1) / (d * d).sum(-1).clamp(min=1e-20)[None]).clamp(0.0, 1.0)
    dist = (rel - s[..., None] * d[None]).norm(dim=-1)               # [M, S]
    best = dist.argmin(dim=1)
    rows = torch.arange(coords.shape[0], device=coords.device)
    t = (best.to(torch.float32) + s[rows, best]) / (table.shape[0] - 1)
    return t, float(dist[rows, best].max()) if coords.shape[0] else 0.0


def curve_delta(weight, table, t):
    """B @ A @ curve(t) for model times t [M] -> [M, out]; weight = B @ A [out, 8]."""
    return table_coords(table.to(t.device), t) @ weight.to(t.device, torch.float32).t()
