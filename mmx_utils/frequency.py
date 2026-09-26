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


_GAUSS5_1D = torch.tensor([1.0, 4.0, 6.0, 4.0, 1.0], dtype=torch.float32) / 16.0


def _gauss5_sep(x: torch.Tensor) -> torch.Tensor:
    """Separable 5-tap Gaussian blur on NCHW with reflect padding."""
    c = x.shape[1]
    k = _GAUSS5_1D.to(device=x.device, dtype=x.dtype)
    kh = k.view(1, 1, 1, 5).repeat(c, 1, 1, 1)
    kv = kh.transpose(2, 3)
    x = F.conv2d(F.pad(x, (2, 2, 0, 0), mode="reflect"), kh, groups=c)
    x = F.conv2d(F.pad(x, (0, 0, 2, 2), mode="reflect"), kv, groups=c)
    return x


def _pyr_down(x: torch.Tensor) -> torch.Tensor:
    x = _gauss5_sep(x)
    return x[:, :, ::2, ::2]


def _pyr_up(x: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    x = F.interpolate(x, size=size, mode="bilinear", align_corners=False)
    return _gauss5_sep(x)


def laplacian_pyramid(img: torch.Tensor, levels: int = 5) -> list[torch.Tensor]:
    """Build a Laplacian pyramid from ``img`` [H,W,C] or [C,H,W] (auto-detected).

    Returns ``levels`` band-pass tensors in NCHW layout (finest first).
    """
    if img.ndim == 3 and img.shape[-1] in (1, 3, 4):
        x = img[..., :3].permute(2, 0, 1).unsqueeze(0).float()
    elif img.ndim == 3:
        x = img[:3].unsqueeze(0).float()
    else:
        raise ValueError(f"laplacian_pyramid expects [H,W,C] or [C,H,W], got {tuple(img.shape)}")

    gauss: list[torch.Tensor] = [x]
    for _ in range(levels - 1):
        nxt = _pyr_down(gauss[-1])
        if nxt.shape[-2] < 2 or nxt.shape[-1] < 2:
            break
        gauss.append(nxt)

    actual = len(gauss)
    lap: list[torch.Tensor] = []
    for i in range(actual - 1):
        up = _pyr_up(gauss[i + 1], gauss[i].shape[-2:])
        lap.append(gauss[i] - up)
    lap.append(gauss[-1])
    return lap


def reconstruct_pyramid(pyr: list[torch.Tensor]) -> torch.Tensor:
    """Reconstruct an image from a Laplacian pyramid (NCHW bands, finest first)."""
    x = pyr[-1]
    for i in range(len(pyr) - 2, -1, -1):
        up = _pyr_up(x, pyr[i].shape[-2:])
        x = pyr[i] + up
    return x


reconstruct = reconstruct_pyramid


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
