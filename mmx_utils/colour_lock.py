"""Low-frequency colour lock (owner 2026-11-25: "balanced colours, less clamping").

A regenerate pass (2K / 4K refine) can drift the overall colour and brightness of the draft. This puts the draft's
large-scale colour back while keeping the new fine detail:

    out + strength * (low(reference) - low(out))

`low` is a very coarse low-pass (downscale to about `grid` cells on the short side, area average, then bilinear
upscale), so only colour cast, exposure and broad tone move; edges, texture and the regenerated detail stay. Nothing
is clamped here: values outside 0..1 are kept for whoever saves the frames (HDR-safe). Original implementation.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

DEFAULT_GRID = 24          # cells on the short side of the low-pass: colour and broad tone only
CHUNK = 8                  # frames per step (bounded memory on long 2K / 4K clips)


def _low(x: torch.Tensor, grid: int) -> torch.Tensor:
    """x [N, 3, H, W] -> its low-pass at the same size."""
    h, w = x.shape[-2:]
    short = min(h, w)
    k = max(1, short // max(1, int(grid)))
    small = F.interpolate(x, size=(max(1, h // k), max(1, w // k)), mode="area")
    return F.interpolate(small, size=(h, w), mode="bilinear", align_corners=False)


METHODS = ("global", "local")
GAIN_RANGE = (0.67, 1.5)   # the broad-contrast gain is never allowed to flatten or blow up the new detail


def _global(o: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    """Per frame and channel: the reference's mean, and its broad-tone contrast (std of a coarse low-pass), applied
    to the whole frame as an offset + gain. A per-frame affine map: no halos, no spatial artefacts."""
    mo, mr = o.mean(dim=(2, 3), keepdim=True), r.mean(dim=(2, 3), keepdim=True)
    so = _low(o, 8).std(dim=(2, 3), keepdim=True)
    sr = _low(r, 8).std(dim=(2, 3), keepdim=True)
    gain = (sr / so.clamp(min=1e-4)).clamp(*GAIN_RANGE)
    return mr + (o - mo) * gain


def lock(out: torch.Tensor, reference: torch.Tensor, strength: float = 1.0, grid: int = DEFAULT_GRID,
         device=None, method: str = "global") -> torch.Tensor:
    """out [T, H, W, C] and reference [T', h, w, C] (resized to out's size; the last reference frame repeats when
    shorter). Returns a new float32 tensor [T, H, W, C] on out's device; channels beyond 3 are passed through.

    method global (default): per-frame colour offset + broad-contrast gain (cannot create halos). local: the
    low-frequency difference is added back (also fixes regional drift, but a small very bright highlight can leave
    a soft halo around it)."""
    if method not in METHODS:
        raise ValueError(f"colour lock method must be one of {', '.join(METHODS)}")
    if not 0.0 <= float(strength) <= 1.0:
        raise ValueError(f"colour lock strength must be between 0 and 1, got {strength}")
    if out.ndim != 4 or reference.ndim != 4:
        raise ValueError("colour lock needs IMAGE batches [frames, H, W, C]")
    if float(strength) == 0.0 or reference.shape[0] == 0:
        return out.to(torch.float32).clone()
    device = device or out.device
    T, H, W = out.shape[0], out.shape[1], out.shape[2]
    result = torch.empty((T, H, W, out.shape[3]), dtype=torch.float32, device=out.device)
    for a in range(0, T, CHUNK):
        b = min(T, a + CHUNK)
        o = out[a:b, :, :, :3].to(device, torch.float32).movedim(-1, 1)
        idx = torch.clamp(torch.arange(a, b), max=reference.shape[0] - 1)
        r = reference[idx][:, :, :, :3].to(device, torch.float32).movedim(-1, 1)
        if r.shape[-2:] != (H, W):
            r = F.interpolate(r, size=(H, W), mode="bilinear", align_corners=False,
                              antialias=r.shape[-2] > H or r.shape[-1] > W)
        target = _global(o, r) if method == "global" else o + (_low(r, grid) - _low(o, grid))
        o = o + float(strength) * (target - o)
        result[a:b, :, :, :3] = o.movedim(1, -1).to(result.device)
        if out.shape[3] > 3:
            result[a:b, :, :, 3:] = out[a:b, :, :, 3:].to(result.device, torch.float32)
    return result


def mean_colour_error(a: torch.Tensor, b: torch.Tensor) -> float:
    """Mean absolute difference of the per-frame mean RGB of two IMAGE batches (a quick drift metric)."""
    n = min(a.shape[0], b.shape[0])
    ma = a[:n, :, :, :3].to(torch.float32).mean(dim=(1, 2))
    mb = b[:n, :, :, :3].to(torch.float32).mean(dim=(1, 2))
    return float((ma - mb).abs().mean())
