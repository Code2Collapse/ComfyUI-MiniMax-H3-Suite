"""Seam tone compensation for chained H3 segments.

PORTED FROM: ComfyUI-MiniMaxH3-ToneCompensate (author rkfg, MIT License).
Fitting maths is the upstream algorithm; the measurement/report layer and the
input validation are added here.

WHY THIS EXISTS: H3's denoiser carries a tone bias - generated segments tend to
run darker than the anchored keyframes they continue from. Chain segments for a
long shot and that bias shows up as a brightness STEP at every seam. This
estimates the bias on the overlap (the frames the new segment regenerates from
the previous one) and removes it across the whole segment.

Three fits, most to least specific:
  frame_shift  per-frame per-channel additive shift. The overlap is the model's
               REGENERATION of the source, not a pixel-wise transform of it, so
               a shift of the mean captures the bias without pretending pixels
               correspond. Default for that reason.
  gain_bias    one affine per channel, s = A*g + C. Extrapolates cleanly; right
               when the drift is a roughly uniform lift or compression.
  lut          per-pixel piecewise-linear curve. Catches nonlinear drift, but
               OVERFITS on regenerated content because the pixels genuinely
               differ. Use when you can see a non-uniform shift.

Pure torch: no model, no weights, testable on CPU.
"""

from __future__ import annotations

import torch

_TABLE = 4096  # dense LUT resolution used when applying `lut`


class ToneCompensateError(ValueError):
    """Raised for an input combination that cannot be fitted."""


# ── fitting primitives (upstream algorithm) ─────────────────────────────────

def fit_affine(src: torch.Tensor, tgt: torch.Tensor):
    """Per-channel regression s = A*g + C. Returns (gain, bias) as [1,1,1,C]."""
    c_out = src.shape[-1]
    gain = torch.ones(1, 1, 1, c_out, dtype=torch.float32, device=src.device)
    bias = torch.zeros(1, 1, 1, c_out, dtype=torch.float32, device=src.device)
    for c in range(c_out):
        s = src[..., c].reshape(-1).float()
        g = tgt[..., c].reshape(-1).float()
        gm, sm = g.mean(), s.mean()
        dg = g - gm
        den = (dg * dg).sum()
        if den < 1e-12:
            A, C = 1.0, float(sm - gm)
        else:
            A = float((dg * (s - sm)).sum() / den)
            C = float(sm - A * gm)
        if abs(A) < 1e-6:
            A = 1.0
        gain[0, 0, 0, c] = A
        bias[0, 0, 0, c] = C
    return gain, bias


def _monotone(ys: torch.Tensor) -> torch.Tensor:
    """Force non-decreasing. Bin means can invert on noise; a tone curve that
    goes backwards produces posterised banding."""
    out, best = [], None
    for y in ys.tolist():
        best = y if best is None or y > best else best
        out.append(best)
    return torch.tensor(out, dtype=torch.float32, device=ys.device)


def lut_control(s: torch.Tensor, g: torch.Tensor, bins: int):
    """Control points from paired pixels: per occupied bin, (mean g, mean s).

    Using the MEANS as x rather than bin centres keeps the outer segments exact;
    centres skew the boundary slopes.
    """
    dev = g.device
    idx = torch.clamp(torch.floor(g * bins), 0, bins - 1).long()
    sums_s = torch.zeros(bins, dtype=torch.float32, device=dev)
    sums_g = torch.zeros(bins, dtype=torch.float32, device=dev)
    counts = torch.zeros(bins, dtype=torch.float32, device=dev)
    sums_s.index_add_(0, idx, s.float())
    sums_g.index_add_(0, idx, g.float())
    counts.index_add_(0, idx, torch.ones_like(idx, dtype=torch.float32))
    nz = counts > 0
    return (sums_g / counts)[nz], _monotone((sums_s / counts)[nz])


def _linfit(x: torch.Tensor, y: torch.Tensor):
    xm, ym = x.mean(), y.mean()
    dx = x - xm
    den = (dx * dx).sum()
    if den < 1e-12:
        return 0.0, float(ym)
    slope = float((dx * (y - ym)).sum() / den)
    return slope, float(ym - slope * xm)


def pwl(query: torch.Tensor, xs: torch.Tensor, ys: torch.Tensor) -> torch.Tensor:
    """Piecewise-linear evaluation with robust end extrapolation.

    The ends use a least-squares slope over the outermost few points so one
    noisy boundary bin cannot swing the extrapolation.
    """
    n = xs.numel()
    if n == 1:
        return torch.full_like(query, float(ys.item()))
    k = min(5, n)
    ls, lb = _linfit(xs[:k], ys[:k])
    rs, rb = _linfit(xs[-k:], ys[-k:])
    i = torch.clamp(torch.searchsorted(xs, query), 1, n - 1)
    xl, xr = xs[i - 1], xs[i]
    yl, yr = ys[i - 1], ys[i]
    out = yl + (yr - yl) * (query - xl) / (xr - xl)
    out = torch.where(query < xs[0], ls * query + lb, out)
    out = torch.where(query > xs[-1], rs * query + rb, out)
    return out


def apply_lut(x: torch.Tensor, xs: torch.Tensor, ys: torch.Tensor,
              table: int = _TABLE) -> torch.Tensor:
    dense = pwl(torch.linspace(0, 1, table, device=x.device), xs, ys)
    idx = torch.clamp(torch.floor(x * table), 0, table - 1).long()
    return dense[idx]


# ── the operation, with measurement ─────────────────────────────────────────

def compensate(source: torch.Tensor, target: torch.Tensor, mode: str,
               overlap: int, lut_bins: int = 64):
    """Return (corrected_target, stats).

    `stats` carries the measured per-channel drift and what had to be adjusted,
    so the node can report it and a widget can plot it. Upstream returned only
    the image, which makes a wrong `overlap` invisible.
    """
    if source.ndim != 4 or target.ndim != 4:
        raise ToneCompensateError(
            "source and target must be IMAGE batches shaped [frames, H, W, C]."
        )
    if source.shape[-1] != target.shape[-1]:
        raise ToneCompensateError(
            f"source has {source.shape[-1]} channels but target has "
            f"{target.shape[-1]}; they must match."
        )

    src, tgt = source.float(), target.float()
    requested = int(overlap)
    if requested <= 0:
        raise ToneCompensateError("overlap must be at least 1 frame.")

    n = min(requested, src.shape[0], tgt.shape[0])
    clamped = n != requested
    if n <= 0:
        raise ToneCompensateError("No overlapping frames to fit on.")

    fit_src, fit_tgt = src[-n:], tgt[:n]

    # Per-frame per-channel drift is worth measuring in EVERY mode, because it
    # is the number that tells you whether the seam was actually the problem.
    drift = (fit_tgt.mean(dim=(1, 2), keepdim=True)
             - fit_src.mean(dim=(1, 2), keepdim=True))

    if mode == "frame_shift":
        out = tgt.clone()
        out[:n] = out[:n] - drift
        out[n:] = out[n:] - drift[-1]
    elif mode == "gain_bias":
        gain, bias = fit_affine(fit_src, fit_tgt)
        out = gain * tgt + bias
    elif mode == "lut":
        out = torch.empty_like(tgt)
        for c in range(tgt.shape[-1]):
            xs, ys = lut_control(fit_src[..., c].reshape(-1),
                                 fit_tgt[..., c].reshape(-1), int(lut_bins))
            out[..., c] = apply_lut(tgt[..., c], xs, ys)
    else:
        raise ToneCompensateError(
            f"Unknown mode {mode!r}; expected frame_shift, gain_bias or lut."
        )

    out = out.clamp_(0.0, 1.0).to(target.dtype)

    per_frame = drift.reshape(n, -1)
    stats = {
        "mode": mode,
        "overlap_used": int(n),
        "overlap_requested": requested,
        "overlap_clamped": bool(clamped),
        "source_frames": int(src.shape[0]),
        "target_frames": int(tgt.shape[0]),
        "drift_per_channel": [round(float(v), 6)
                              for v in drift.reshape(n, -1).mean(dim=0)],
        "drift_per_frame": [[round(float(v), 6) for v in row]
                            for row in per_frame],
        "drift_mean": round(float(drift.mean()), 6),
        "drift_abs_max": round(float(drift.abs().max()), 6),
    }
    return out, stats


def compensate_report(stats: dict) -> str:
    ch = ", ".join(f"{v:+.4f}" for v in stats["drift_per_channel"])
    lines = [
        f"tone compensate: {stats['mode']} over {stats['overlap_used']} overlap frame(s)",
        f"  measured drift  per channel [{ch}]  mean {stats['drift_mean']:+.4f}",
        f"  largest single-frame drift  {stats['drift_abs_max']:.4f}",
    ]
    if stats["overlap_clamped"]:
        lines.append(
            f"  NOTE: overlap {stats['overlap_requested']} exceeds the footage "
            f"({stats['source_frames']} source / {stats['target_frames']} target "
            f"frames) and was clamped to {stats['overlap_used']}. If that is not "
            f"the keyframe count used to generate the segment, the fit is being "
            f"taken over the wrong frames and the seam will not line up."
        )
    if stats["drift_abs_max"] < 1e-4:
        lines.append(
            "  reading: essentially no drift - the seam was not a tone problem, "
            "so look at motion or content continuity instead."
        )
    return "\n".join(lines)
