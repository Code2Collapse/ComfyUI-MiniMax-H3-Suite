# MIT License — ComfyUI-NKD-Basic-Tools
# PORTED FROM: ComfyUI-NKD-Basic-Tools :: nkd_frequency.py @ HEAD
"""Low-frequency filters and detail reinject (N6 core logic)."""

from __future__ import annotations

import torch
import torch.nn.functional as F

_EPS = 1e-6


def _box(x: torch.Tensor, r: int) -> torch.Tensor:
    if r < 1:
        return x
    k = 2 * r + 1
    c = x.shape[1]
    kh = torch.ones(c, 1, 1, k, device=x.device, dtype=x.dtype) / k
    kv = kh.transpose(2, 3)
    x = F.conv2d(F.pad(x, (r, r, 0, 0), mode="replicate"), kh, groups=c)
    x = F.conv2d(F.pad(x, (0, 0, r, r), mode="replicate"), kv, groups=c)
    return x


def _gaussian(x: torch.Tensor, r: int) -> torch.Tensor:
    if r < 1:
        return x
    sigma = max(r / 2.0, 0.5)
    k = 2 * r + 1
    t = torch.arange(k, device=x.device, dtype=x.dtype) - r
    g = torch.exp(-(t * t) / (2 * sigma * sigma))
    g = g / g.sum()
    c = x.shape[1]
    kh = g.view(1, 1, 1, k).repeat(c, 1, 1, 1)
    kv = kh.transpose(2, 3)
    x = F.conv2d(F.pad(x, (r, r, 0, 0), mode="replicate"), kh, groups=c)
    x = F.conv2d(F.pad(x, (0, 0, r, r), mode="replicate"), kv, groups=c)
    return x


def low_freq(img_nchw: torch.Tensor, method: str = "Gaussian", radius: int = 8) -> torch.Tensor:
    if method == "Box":
        return _box(img_nchw, int(radius))
    return _gaussian(img_nchw, int(radius))


def to_nchw(image: torch.Tensor) -> torch.Tensor:
    return image[..., :3].permute(0, 3, 1, 2).contiguous().float()


def to_nhwc(x: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    return x.permute(0, 2, 3, 1).contiguous().to(dtype)


def frequency_separate(
    image: torch.Tensor,
    *,
    mode: str = "divide",
    radius: int = 8,
    method: str = "Gaussian",
    eps: float = _EPS,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (low_freq, detail) in NCHW. detail is divide-ratio or subtract-residual."""
    src = to_nchw(image)
    lf = low_freq(src, method=method, radius=radius)
    safe = lf.abs() + float(eps)
    if mode.lower() == "subtract":
        detail = src - lf
    else:
        detail = src / safe
    return lf, detail


def frequency_combine(
    low_freq_nchw: torch.Tensor,
    detail_nchw: torch.Tensor,
    *,
    mode: str = "divide",
) -> torch.Tensor:
    if mode.lower() == "subtract":
        return low_freq_nchw + detail_nchw
    return low_freq_nchw * detail_nchw.clamp(min=0.0)


def detail_reinject_frame(
    plate: torch.Tensor,
    generated: torch.Tensor,
    mask: torch.Tensor | None,
    *,
    mode: str = "divide",
    radius: int = 8,
    method: str = "Gaussian",
    strength: float = 1.0,
    eps: float = _EPS,
) -> torch.Tensor:
    """Single frame [H,W,C]. Mask [H,W] — detail applied only inside mask."""
    p = to_nchw(plate.unsqueeze(0))
    g = to_nchw(generated.unsqueeze(0))
    lf = low_freq(p, method=method, radius=radius)
    if mode.lower() == "subtract":
        detail = p - lf
        target = g + detail * float(strength)
    else:
        detail = p / (lf.abs() + float(eps))
        neutral = torch.ones_like(detail)
        target = g * (neutral + (detail - neutral) * float(strength))
    out = g.clone()
    if mask is not None:
        m = mask.float()
        if m.ndim == 3:
            m = m[..., 0]
        m = m.unsqueeze(0).unsqueeze(0)
        if m.shape[-2:] != out.shape[-2:]:
            m = F.interpolate(m, size=out.shape[-2:], mode="bilinear", align_corners=False)
        out = g * (1.0 - m) + target * m
    else:
        out = target
    return to_nhwc(out, plate.dtype).squeeze(0)
