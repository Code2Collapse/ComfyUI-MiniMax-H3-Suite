"""Split / tiled latent upscaling — pure logic for MiniMax H3.

PORTED FROM: Comfyui_Minimax_h3_latent_Upscaler (author LBH-123-AI, MIT License).
This is a direct port of the upstream split-upscale maths, not a clean-room
reimplementation. ComfyUI wiring lives in mmx_nodes/split_upscale.py.

Temporal grid helpers (clip_tokens, compute_h3_segments_adaptive, …) are imported
from mmx_utils/h3_grid.py so there is a single verified copy of the H3 grid.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import torch

from .h3_constants import FRAME_PER_TOKEN
from .h3_grid import (
    audio_range,
    clip_tokens,
    compute_h3_segments_adaptive,
    frames_for_tokens,
    snap_clip_frames,
    snap_overlap_frames,
    steps_for_frames,
    token_start_at_or_before,
    tokens_for_frames,
)

VAE_DOWNSAMPLE = 16
ALIGN = 2
MAX_IDENTITY_ANCHORS = 6
POLISH_HALO = 16
DC_MATCH_CLAMP = 0.05
SEAM_CORR_GATE = 0.85
SEAM_DC_GATE = 0.6
COLOR_CLAMP = 0.05


# ── latent validation ───────────────────────────────────────────────────────


def describe_samples(samples: Any) -> str:
    """One-line description of what was received instead of an H3 AV latent."""
    if samples is None:
        return "missing samples (None)"
    if hasattr(samples, "unbind"):
        members = list(samples.unbind())
        shapes = [tuple(m.shape) for m in members]
        return f"NestedTensor with {len(members)} stream(s), shapes {shapes}"
    if isinstance(samples, torch.Tensor):
        return f"plain torch.Tensor shape {tuple(samples.shape)}"
    nested = getattr(samples, "is_nested", False)
    tensors = getattr(samples, "tensors", None)
    if nested and tensors is not None:
        shapes = [tuple(t.shape) for t in tensors]
        return f"nested pair with shapes {shapes}"
    return f"{type(samples).__name__}"


def is_h3_av_latent(samples: Any) -> bool:
    """True when *samples* looks like a MiniMax H3 joint video+audio latent."""
    if samples is None:
        return False
    if hasattr(samples, "unbind"):
        members = list(samples.unbind())
        if len(members) != 2:
            return False
        video, audio = members
        return (
            video.ndim == 5
            and video.shape[1] == 24
            and audio.ndim == 4
            and audio.shape[1] == 32
        )
    if getattr(samples, "is_nested", False) and hasattr(samples, "tensors"):
        tensors = samples.tensors
        if len(tensors) != 2:
            return False
        video, audio = tensors
        return (
            video.ndim == 5
            and video.shape[1] == 24
            and audio.ndim == 4
            and audio.shape[1] == 32
        )
    return False


def refuse_unless_h3_av_latent(samples: Any) -> None:
    if not is_h3_av_latent(samples):
        raise ValueError(
            "Expected a MiniMax H3 joint AV latent (nested video+audio: "
            "video [B,24,T,H,W], audio [B,32,2,Ta]). "
            f"Received {describe_samples(samples)}."
        )


def unpack_av_samples(samples: Any) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (video, audio) tensors from either NestedTensor API."""
    refuse_unless_h3_av_latent(samples)
    if hasattr(samples, "unbind"):
        video, audio = samples.unbind()
        return video, audio
    return samples.tensors[0], samples.tensors[1]


# ── spatial grid (latent pixels) ────────────────────────────────────────────


def px_to_lat(px: int) -> int:
    return max(ALIGN, (round(int(px) / VAE_DOWNSAMPLE) // ALIGN) * ALIGN)


def snap_align(v: float | int) -> int:
    return max(0, int(round(float(v) / ALIGN)) * ALIGN)


def _grid_1d(size: int, tile: int, ol: int, min_tile: int) -> tuple[list[int], list[int], list[int]]:
    if size <= tile:
        return [0], [size], [0]
    sh = tile - ol
    n = math.ceil((size - ol) / sh)
    if (n - 1) * sh + tile < size:
        n += 1
    rows = [i * sh for i in range(n)]
    trows = [min(tile, size - r) for r in rows]
    if min_tile > 0 and n >= 2:
        edge = size - rows[-1]
        if edge < min_tile:
            new_last = size - min_tile
            if rows[-2] < new_last < rows[-2] + trows[-2]:
                rows[-1] = new_last
                trows[-1] = size - new_last
    ovl = [0] * n
    for i in range(1, n):
        ovl[i] = max(0, rows[i - 1] + trows[i - 1] - rows[i])
    return rows, trows, ovl


def compute_spatial_grid(
    h: int,
    w: int,
    th: int,
    tw: int,
    ol_h: int,
    ol_w: int,
    min_th: int = 0,
    min_tw: int = 0,
) -> tuple[list[int], list[int], list[int], list[int], list[int], list[int]]:
    rows, trows, row_ovl = _grid_1d(h, th, ol_h, min_th)
    cols, tcols, col_ovl = _grid_1d(w, tw, ol_w, min_tw)
    return rows, cols, trows, tcols, row_ovl, col_ovl


def spatial_crossfade_weights(
    overlap: int,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Left/right blend weights for an overlap band; left + right == 1."""
    n = int(overlap)
    if n <= 0:
        z = torch.zeros(0, device=device, dtype=dtype)
        return z, z
    right = torch.linspace(0.0, 1.0, n, device=device, dtype=dtype)
    return 1.0 - right, right


def spatial_fade_mask(
    tile_h: int,
    tile_w: int,
    ovh: int,
    ovw: int,
    done_top: bool,
    done_left: bool,
    fade_h: int = 0,
    fade_w: int = 0,
    seam_cap: float = 1.0,
) -> torch.Tensor:
    """Denoise-freeze mask: 1 = free resample, 0 = frozen neighbour.

    Overlap = frozen band (seam side) + ramp (0 -> seam_cap). When seam_cap < 1,
    a second ramp (seam_cap -> 1) follows so the seam neighbourhood can
    "continue" frozen content at moderate denoise — upstream behaviour.
    """
    mask = torch.ones(tile_h, tile_w, dtype=torch.float32)

    def profile(n: int, ov: int, fade: int) -> torch.Tensor:
        p = torch.ones(n, dtype=torch.float32)
        f = min(fade, ov)
        frozen = ov - f
        p[:frozen] = 0.0
        if f > 0:
            p[frozen:ov] = torch.linspace(0.0, seam_cap, f)
        ramp = min(ov, n - ov)
        if ramp > 0 and seam_cap < 1.0:
            start = seam_cap if f > 0 else 0.0
            p[ov:ov + ramp] = torch.linspace(start, 1.0, ramp)
        return p

    if done_left and ovw > 0:
        mask = torch.minimum(mask, profile(tile_w, ovw, fade_w)[None, :])
    if done_top and ovh > 0:
        mask = torch.minimum(mask, profile(tile_h, ovh, fade_h)[:, None])
    return mask


# ── colour / seam metrics ───────────────────────────────────────────────────


def dc_correct(
    new: torch.Tensor,
    refs: Sequence[tuple[torch.Tensor | None, torch.Tensor | None] | None],
    clamp: float = COLOR_CLAMP,
    min_samples: int = 256,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    pairs = [
        p for p in refs
        if p is not None and p[0] is not None and p[0].numel() >= min_samples
    ]
    if not pairs:
        return new, None
    pa = torch.cat([
        a.float().permute(0, 2, 3, 4, 1).reshape(-1, a.shape[1]) for a, _ in pairs
    ])
    pb = torch.cat([
        b.float().permute(0, 2, 3, 4, 1).reshape(-1, b.shape[1]) for _, b in pairs
    ])
    dc = (pa - pb).median(dim=0).values.clamp(-clamp, clamp)
    return new - dc.view(1, -1, 1, 1, 1).to(new.device, new.dtype), dc


def grade_pin(
    chunk: torch.Tensor,
    ref: torch.Tensor,
    clamp: float = COLOR_CLAMP,
) -> tuple[torch.Tensor, torch.Tensor]:
    a = chunk.float().permute(0, 2, 3, 4, 1).reshape(-1, chunk.shape[1])
    b = ref.float().permute(0, 2, 3, 4, 1).reshape(-1, ref.shape[1])
    dc = (a - b).median(dim=0).values.clamp(-clamp, clamp)
    return chunk - dc.view(1, -1, 1, 1, 1).to(chunk.device, chunk.dtype), dc


def seam_metrics(sub: torch.Tensor, region: torch.Tensor) -> tuple[torch.Tensor | None, float | None]:
    if sub.numel() < 4096:
        return None, None
    sp = sub.float().permute(0, 2, 3, 4, 1).reshape(-1, sub.shape[1])
    rp = region.float().permute(0, 2, 3, 4, 1).reshape(-1, region.shape[1])
    dc = (sp - rp).median(dim=0).values.clamp(-DC_MATCH_CLAMP, DC_MATCH_CLAMP)
    a = sp - sp.mean(dim=0)
    b = rp - rp.mean(dim=0)
    corr = ((a * b).mean(dim=0) / (a.std(dim=0) * b.std(dim=0) + 1e-6)).median()
    return dc, float(corr)


def should_polish_seam(mode: str, sub: torch.Tensor, region: torch.Tensor) -> bool:
    if mode == "all":
        return True
    dc, corr = seam_metrics(sub, region)
    if dc is None:
        return False
    return (
        (corr is not None and corr < SEAM_CORR_GATE)
        or dc.abs().max().item() > SEAM_DC_GATE * DC_MATCH_CLAMP
    )


# ── temporal stitch ─────────────────────────────────────────────────────────


def _crossfade(a: torch.Tensor, b: torch.Tensor, dim: int) -> torch.Tensor:
    n = a.shape[dim]
    w = torch.linspace(0.0, 1.0, n, device=a.device, dtype=a.dtype)
    shape = [1] * a.ndim
    shape[dim] = n
    return a + (b - a) * w.view(shape)


def temporal_append(
    acc_v: torch.Tensor | None,
    acc_a: torch.Tensor | None,
    chunk_v: torch.Tensor,
    chunk_a: torch.Tensor,
    index: int,
    k0: int,
    f0: int,
    *,
    color_match: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    if acc_v is None:
        return chunk_v.clone(), chunk_a.clone()
    from .h3_grid import FRAME_RESCALE  # local import keeps the constant in one place

    gi, agi = k0, round(f0 * FRAME_RESCALE)
    total_v = max(acc_v.shape[2], gi + chunk_v.shape[2])
    total_a = max(acc_a.shape[-1], agi + chunk_a.shape[-1])
    rv = torch.zeros(
        (1, acc_v.shape[1], total_v, acc_v.shape[3], acc_v.shape[4]),
        device=acc_v.device,
        dtype=acc_v.dtype,
    )
    ra = torch.zeros((1, 32, 2, total_a), device=acc_a.device, dtype=acc_a.dtype)
    rv[:, :, :acc_v.shape[2]] = acc_v
    ra[:, :, :, :acc_a.shape[-1]] = acc_a
    v, a = chunk_v, chunk_a
    if index > 0:
        ov = min(acc_v.shape[2] - gi, v.shape[2])
        if ov > 0:
            if color_match:
                v, _ = dc_correct(v, [(v[:, :, :ov], rv[:, :, gi:gi + ov])])
            rv[:, :, gi:gi + ov] = _crossfade(rv[:, :, gi:gi + ov], v[:, :, :ov], dim=2)
            v = v[:, :, ov:]
        gi += ov
        ova = min(acc_a.shape[-1] - agi, a.shape[-1])
        if ova > 0:
            ra[:, :, :, agi:agi + ova] = _crossfade(
                ra[:, :, :, agi:agi + ova], a[:, :, :, :ova], dim=3,
            )
            a = a[:, :, :, ova:]
        agi += ova
    if v.shape[2] > 0:
        rv[:, :, gi:gi + v.shape[2]] = v
    if a.shape[-1] > 0:
        ra[:, :, :, agi:agi + a.shape[-1]] = a
    return rv, ra


# ── spatial tile merge (testable without ComfyUI) ───────────────────────────


def merge_spatial_tile(
    chunk: torch.Tensor,
    tile_v: torch.Tensor,
    r0: int,
    c0: int,
    tr: int,
    tc: int,
    ovh: int,
    ovw: int,
    ri: int,
    cj: int,
) -> torch.Tensor:
    """Blend one sampled tile back into *chunk* without mutating either input."""
    out = chunk.clone()
    region = out[:, :, :, r0:r0 + tr, c0:c0 + tc].clone()
    if cj > 0 and ovw > 0:
        left, right = spatial_crossfade_weights(
            ovw, device=region.device, dtype=region.dtype,
        )
        view = left.view(1, 1, 1, 1, ovw)
        region[:, :, :, :, :ovw] = (
            region[:, :, :, :, :ovw] * view + tile_v[:, :, :, :, :ovw] * right.view(1, 1, 1, 1, ovw)
        )
    if ri > 0 and ovh > 0:
        top, bottom = spatial_crossfade_weights(
            ovh, device=region.device, dtype=region.dtype,
        )
        view = top.view(1, 1, 1, ovh, 1)
        region[:, :, :, :ovh, :] = (
            region[:, :, :, :ovh, :] * view + tile_v[:, :, :, :ovh, :] * bottom.view(1, 1, 1, ovh, 1)
        )
    band = torch.zeros((1, 1, 1, tr, tc), dtype=torch.bool, device=region.device)
    if cj > 0 and ovw > 0:
        band[:, :, :, :, :ovw] = True
    if ri > 0 and ovh > 0:
        band[:, :, :, :ovh, :] = True
    region = torch.where(band, region, tile_v)
    out[:, :, :, r0:r0 + tr, c0:c0 + tc] = region
    return out


# ── keyframe helpers (torch only) ───────────────────────────────────────────


def trim_keyframe(kf: dict, f0: int, f1: int) -> dict | None:
    idx = kf["resolved_frame_index"]
    latent, audio_latent = kf.get("latent"), kf.get("audio_latent")
    if latent is None and audio_latent is None:
        return None if (idx < f0 or idx >= f1) else {"resolved_frame_index": idx - f0}
    out: dict = {}
    if latent is not None:
        t_start = t_end = None
        pos = idx
        for k in range(latent.shape[2]):
            span = FRAME_PER_TOKEN[k % 5]
            if f0 <= pos and pos + span <= f1:
                if t_start is None:
                    t_start = k
                t_end = k + 1
            pos += span
        if t_start is None:
            return None
        out["latent"] = latent[:, :, t_start:t_end].contiguous()
        out["resolved_frame_index"] = idx + frames_for_tokens(t_start) - f0
    if audio_latent is not None:
        from .h3_grid import FRAME_RESCALE

        rt = audio_latent.shape[-1]
        a_start = max(0, math.ceil((f0 - idx) * FRAME_RESCALE))
        a_end = min(rt, math.floor((f1 - idx) / FRAME_RESCALE))
        if a_end > a_start:
            out["audio_latent"] = audio_latent[..., a_start:a_end].contiguous()
            if "resolved_frame_index" not in out:
                out["resolved_frame_index"] = max(0, idx - f0)
    return out if ("latent" in out or "audio_latent" in out) else None


def motion_keyframes(
    prev_video: torch.Tensor,
    prev_tokens: int,
    f0: int,
    n_frames: int,
) -> list[dict]:
    n = next((g for g in (56, 39, 22, 5) if g <= n_frames), 0)
    steps = steps_for_frames(n) if n else None
    if not steps or steps > prev_tokens or (prev_tokens - steps) % 5 != 0:
        return []
    start = prev_tokens - steps
    return [
        {
            "resolved_frame_index": frames_for_tokens(start + k) - f0,
            "latent": prev_video[:, :, start + k:start + k + 1].contiguous(),
        }
        for k in range(steps)
    ]


def identity_keyframes(
    source: torch.Tensor,
    f0: int,
    f1: int,
    spacing: int,
) -> list[dict]:
    kfs: list[dict] = []
    p = f0 + spacing
    while p < f1:
        k = token_start_at_or_before(p)
        kfs.append({
            "resolved_frame_index": frames_for_tokens(k) - f0,
            "latent": source[:, :, k:k + 1].contiguous(),
        })
        p += spacing
    if len(kfs) > MAX_IDENTITY_ANCHORS:
        kfs = [kfs[i * len(kfs) // MAX_IDENTITY_ANCHORS] for i in range(MAX_IDENTITY_ANCHORS)]
    return kfs


# ── parameter builders / reports ────────────────────────────────────────────


def build_spatial_param(
    tile_width: int,
    tile_height: int,
    overlap_ratio: float,
    fade_ratio: float,
    min_tile_size: int,
    seam_denoise: float,
) -> dict[str, Any]:
    tw, th = px_to_lat(tile_width), px_to_lat(tile_height)
    ol_w = min(tw - ALIGN, snap_align(tw * overlap_ratio))
    ol_h = min(th - ALIGN, snap_align(th * overlap_ratio))
    fw = min(ol_w, int(round(ol_w * fade_ratio)))
    fh = min(ol_h, int(round(ol_h * fade_ratio)))
    mt = min(px_to_lat(min_tile_size), th, tw) if min_tile_size > 0 else 0
    return {
        "tw": tw, "th": th, "ol_w": ol_w, "ol_h": ol_h,
        "fw": fw, "fh": fh, "mt": mt, "cap": float(seam_denoise),
        "tile_width_px": tile_width, "tile_height_px": tile_height,
        "overlap_ratio": overlap_ratio, "fade_ratio": fade_ratio,
    }


def spatial_param_report(param: dict[str, Any], h: int, w: int) -> str:
    rows, cols, trows, tcols, _, _ = compute_spatial_grid(
        h, w, param["th"], param["tw"], param["ol_h"], param["ol_w"], param["mt"], param["mt"],
    )
    return (
        f"spatial grid {len(rows)}x{len(cols)} on {h}x{w} latent\n"
        f"  tile {param['tile_width_px']}x{param['tile_height_px']}px "
        f"-> {param['tw']}x{param['th']}lat\n"
        f"  overlap {param['overlap_ratio']:.0%} -> {param['ol_w']}x{param['ol_h']}lat\n"
        f"  fade {param['fade_ratio']:.0%} -> {param['fw']}x{param['fh']}lat\n"
        f"  seam_denoise cap {param['cap']:.2f}\n"
        f"  row tile heights {trows}, col tile widths {tcols}"
    )


def build_temporal_param(
    chunk_frames: int,
    temporal_overlap_frames: int,
    anchor_strength: float,
    motion_anchor_frames: int,
    identity_anchor_frames: int,
) -> dict[str, Any]:
    chunk = snap_clip_frames(chunk_frames)
    overlap = snap_overlap_frames(temporal_overlap_frames)
    if overlap >= chunk:
        overlap = snap_overlap_frames(chunk - 17)
    return {
        "chunk_frames": chunk,
        "overlap_frames": overlap,
        "anchor_strength": float(anchor_strength),
        "motion_anchor_frames": int(motion_anchor_frames),
        "identity_anchor_frames": int(identity_anchor_frames),
    }


def temporal_param_report(param: dict[str, Any], total_tokens: int) -> str:
    bounds, total_px = compute_h3_segments_adaptive(
        total_tokens, param["chunk_frames"], param["overlap_frames"],
    )
    return (
        f"{len(bounds)} temporal segment(s), {total_px} pixel frames from {total_tokens} latent rows\n"
        f"  chunk {param['chunk_frames']} frames, overlap {param['overlap_frames']} frames\n"
        f"  anchor_strength {param['anchor_strength']:.3f}, "
        f"motion_anchor {param['motion_anchor_frames']}f, "
        f"identity_anchor every {param['identity_anchor_frames']}f"
    )


def split_upscale_report(
    *,
    bounds: Sequence[tuple[int, int, int, int]],
    nrows: int,
    ncols: int,
    row_ovl: Sequence[int],
    col_ovl: Sequence[int],
    seam_polish: str,
    polish_count: int,
    color_match: bool,
) -> str:
    max_ro = max(row_ovl) if row_ovl else 0
    max_co = max(col_ovl) if col_ovl else 0
    lines = [
        f"{len(bounds)} temporal segment(s), spatial grid {nrows}x{ncols}",
        f"  overlap up to {max_ro} latent rows x {max_co} latent cols in pixels",
        f"  colour_match {'on' if color_match else 'off'}",
    ]
    if seam_polish == "off":
        lines.append("  seam polish off")
    elif polish_count:
        lines.append(f"  seam polish {seam_polish}: {polish_count} seam(s) repolished")
    else:
        lines.append(f"  seam polish {seam_polish}: no seam needed repolishing")
    return "\n".join(lines)


__all__ = [
    "ALIGN",
    "POLISH_HALO",
    "VAE_DOWNSAMPLE",
    "audio_range",
    "build_spatial_param",
    "build_temporal_param",
    "clip_tokens",
    "compute_h3_segments_adaptive",
    "compute_spatial_grid",
    "dc_correct",
    "describe_samples",
    "frames_for_tokens",
    "grade_pin",
    "identity_keyframes",
    "is_h3_av_latent",
    "merge_spatial_tile",
    "motion_keyframes",
    "px_to_lat",
    "refuse_unless_h3_av_latent",
    "seam_metrics",
    "should_polish_seam",
    "snap_align",
    "snap_clip_frames",
    "snap_overlap_frames",
    "spatial_crossfade_weights",
    "spatial_fade_mask",
    "spatial_param_report",
    "split_upscale_report",
    "steps_for_frames",
    "temporal_append",
    "temporal_param_report",
    "token_start_at_or_before",
    "tokens_for_frames",
    "trim_keyframe",
    "unpack_av_samples",
]
