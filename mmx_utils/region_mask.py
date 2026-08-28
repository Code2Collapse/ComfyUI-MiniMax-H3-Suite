"""Deterministic region masks from depth or painted input — pixel space only (R18)."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .feather_composite import gaussian_blur_mask

# water/atmosphere removed — screen-Y heuristics were not depth-derived (R7).
REGION_MODES = ("sky", "ground", "painted")


def _depth_plane_single(depth: torch.Tensor, h: int, w: int, device: torch.device) -> torch.Tensor:
    d = depth.float()
    if d.ndim == 4:
        d = d[0]
    if d.ndim == 3 and d.shape[-1] >= 1:
        plane = d[..., :3].mean(dim=-1)
    elif d.ndim == 2:
        plane = d
    else:
        raise ValueError(f"depth must be [H,W,C] or [H,W], got {tuple(depth.shape)}")
    if plane.shape[-2:] != (h, w):
        plane = F.interpolate(
            plane.unsqueeze(0).unsqueeze(0),
            size=(h, w),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0).squeeze(0)
    return plane.to(device)


def _vertical_ramp_plane(h: int, w: int, device: torch.device) -> torch.Tensor:
    yy = torch.linspace(0.0, 1.0, h, device=device)
    return yy.view(-1, 1).expand(h, w).contiguous()


def _horizon_threshold_from_range(lo: float, hi: float, horizon_bias: float) -> float:
    span = max(hi - lo, 1e-6)
    return lo + span * (0.5 + float(horizon_bias))


def compute_clip_global_threshold(
    *,
    height: int,
    width: int,
    depth_sequence: torch.Tensor | None,
    horizon_bias: float = 0.0,
    device: torch.device | None = None,
) -> tuple[float, bool]:
    """One threshold for the whole clip — returns (threshold, used_ramp_fallback)."""
    dev = device or torch.device("cpu")
    h, w = int(height), int(width)
    if depth_sequence is None:
        plane = _vertical_ramp_plane(h, w, dev)
        return _horizon_threshold_from_range(float(plane.min()), float(plane.max()), horizon_bias), True
    seq = depth_sequence.float()
    if seq.ndim == 3:
        seq = seq.unsqueeze(0)
    planes = [_depth_plane_single(seq[i], h, w, dev) for i in range(seq.shape[0])]
    stacked = torch.stack(planes, dim=0)
    lo = float(stacked.min())
    hi = float(stacked.max())
    return _horizon_threshold_from_range(lo, hi, horizon_bias), False


def region_mask_from_mode(
    *,
    mode: str,
    height: int,
    width: int,
    depth: torch.Tensor | None = None,
    painted_mask: torch.Tensor | None = None,
    horizon_bias: float = 0.0,
    threshold: float | None = None,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Return binary region mask [H,W] float32 in pixel space."""
    dev = device or torch.device("cpu")
    h, w = int(height), int(width)
    m = (mode or "sky").lower()
    if m not in REGION_MODES:
        raise ValueError(f"Unknown region mode {mode!r}; expected one of {REGION_MODES}")

    if m == "painted":
        if painted_mask is None:
            raise ValueError("painted mode requires painted_mask — connect a MASK or switch mode.")
        pm = painted_mask.float()
        if pm.ndim == 3:
            pm = pm[0]
        if pm.ndim != 2:
            raise ValueError(f"painted_mask must be [H,W], got {tuple(painted_mask.shape)}")
        if pm.shape != (h, w):
            pm = F.interpolate(
                pm.unsqueeze(0).unsqueeze(0),
                size=(h, w),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0).squeeze(0)
        return pm.clamp(0, 1).to(dev)

    if depth is None:
        plane = _vertical_ramp_plane(h, w, dev)
    else:
        plane = _depth_plane_single(depth, h, w, dev)

    if threshold is None:
        thresh = _horizon_threshold_from_range(float(plane.min()), float(plane.max()), horizon_bias)
    else:
        thresh = float(threshold)

    far = plane > thresh
    near = ~far

    if m == "sky":
        mask = far.float()
    elif m == "ground":
        mask = near.float()
    else:
        mask = far.float()

    return mask.clamp(0, 1)


def feather_mask_pixel_space(mask: torch.Tensor, feather_px: int) -> torch.Tensor:
    """Gaussian feather in pixel space (D7). mask [H,W] or [B,H,W]."""
    if int(feather_px) <= 0:
        return mask.float()
    x = mask.float()
    batched = x.ndim == 3
    if not batched:
        x = x.unsqueeze(0)
    out = []
    for i in range(x.shape[0]):
        m = x[i].unsqueeze(0).unsqueeze(0)
        out.append(gaussian_blur_mask(m, int(feather_px)).squeeze(0).squeeze(0))
    y = torch.stack(out, dim=0)
    return y if batched else y[0]


def mask_preview(images: torch.Tensor, mask: torch.Tensor, *, tint: tuple[float, float, float] = (0.2, 0.6, 1.0)) -> torch.Tensor:
    """Overlay tinted mask on images [B,H,W,C]."""
    img = images.float()
    if img.ndim == 3:
        img = img.unsqueeze(0)
    m = mask.float()
    if m.ndim == 2:
        m = m.unsqueeze(0)
    if m.shape[0] == 1 and img.shape[0] > 1:
        m = m.expand(img.shape[0], -1, -1)
    b, h, w, _c = img.shape
    if m.shape[-2:] != (h, w):
        m = F.interpolate(m.unsqueeze(1), size=(h, w), mode="bilinear", align_corners=False).squeeze(1)
    color = torch.tensor(tint, device=img.device, dtype=img.dtype).view(1, 1, 1, 3)
    overlay = img * (1.0 - 0.45 * m.unsqueeze(-1)) + color * (0.45 * m.unsqueeze(-1))
    return overlay.clamp(0, 1)
