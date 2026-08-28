"""Pure-torch Canny edge detection — CPU-testable, no weights, no cv2 requirement."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _gaussian_kernel1d(sigma: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    sigma = max(float(sigma), 1e-3)
    radius = max(1, int(round(3.0 * sigma)))
    x = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    g = torch.exp(-(x * x) / (2.0 * sigma * sigma))
    return g / g.sum()


def gaussian_blur_gray(gray: torch.Tensor, sigma: float) -> torch.Tensor:
    """gray [B,1,H,W] separable Gaussian."""
    g = _gaussian_kernel1d(sigma, gray.device, gray.dtype)
    r = (g.numel() - 1) // 2
    c = gray.shape[1]
    kh = g.view(1, 1, 1, -1).repeat(c, 1, 1, 1)
    kv = kh.transpose(2, 3)
    x = F.pad(gray, (r, r, 0, 0), mode="reflect")
    x = F.conv2d(x, kh, groups=c)
    x = F.pad(x, (0, 0, r, r), mode="reflect")
    x = F.conv2d(x, kv, groups=c)
    return x


def rgb_to_gray_nchw(img: torch.Tensor) -> torch.Tensor:
    """[B,H,W,C] or [B,C,H,W] -> [B,1,H,W] luminance."""
    if img.ndim != 4:
        raise ValueError("expected 4D image batch")
    if img.shape[-1] in (1, 3, 4):
        x = img[..., :3].permute(0, 3, 1, 2).float()
    else:
        x = img[:, :3].float()
    w = torch.tensor((0.299, 0.587, 0.114), device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    return (x * w).sum(1, keepdim=True)


def sobel_gradients(gray: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=gray.device, dtype=gray.dtype)
    ky = kx.t()
    kx = kx.view(1, 1, 3, 3)
    ky = ky.view(1, 1, 3, 3)
    # Replicate-pad, never conv2d(padding=1). `padding=1` pads with ZEROS, so any
    # image whose border is not black reads as a full-frame step edge and Canny
    # lights up the entire 1px border ring. Measured on a uniform 0.5 image at
    # 32x32: 124 edge pixels, all 124 in the border ring, interior 0.
    # That matters here because N3 runs this on a CROP: the spurious ring is a
    # perfect rectangle that ControlNet would happily lock structure onto,
    # producing a visible box in the output. The Gaussian above already uses
    # reflect padding; the Sobel must match it.
    padded = F.pad(gray, (1, 1, 1, 1), mode="replicate")
    gx = F.conv2d(padded, kx)
    gy = F.conv2d(padded, ky)
    mag = torch.hypot(gx, gy)
    ang = torch.atan2(gy, gx)
    return mag, ang, mag


def non_max_suppression(mag: torch.Tensor, ang: torch.Tensor) -> torch.Tensor:
    """Thin edges via 4-direction NMS."""
    b, _, h, w = mag.shape
    out = torch.zeros_like(mag)
    ang = ang % (torch.pi)
    for dy, dx in ((0, 1), (1, 1), (1, 0), (-1, 1)):
        shifted = torch.roll(mag, shifts=(dy, dx), dims=(2, 3))
        keep = mag >= shifted
        if dy == 0 and dx == 1:
            mask = (ang < torch.pi / 8) | (ang >= 7 * torch.pi / 8)
        elif dy == 1 and dx == 1:
            mask = (ang >= torch.pi / 8) & (ang < 3 * torch.pi / 8)
        elif dy == 1 and dx == 0:
            mask = (ang >= 3 * torch.pi / 8) & (ang < 5 * torch.pi / 8)
        else:
            mask = (ang >= 5 * torch.pi / 8) & (ang < 7 * torch.pi / 8)
        out = torch.where(mask, torch.maximum(out, mag * keep.float()), out)
    return out


def hysteresis_threshold(
    mag: torch.Tensor,
    low: float,
    high: float,
    max_iters: int = 64,
) -> torch.Tensor:
    strong = mag >= high
    weak = (mag >= low) & ~strong
    out = strong.float()
    for _ in range(max_iters):
        prev = out.clone()
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)):
            nbr = torch.roll(out, shifts=(dy, dx), dims=(2, 3))
            out = torch.where(weak & (nbr > 0), torch.ones_like(out), out)
        if torch.equal(out, prev):
            break
    return out


def canny_torch(
    images: torch.Tensor,
    *,
    low_threshold: float = 0.1,
    high_threshold: float = 0.2,
    blur_sigma: float = 1.4,
) -> torch.Tensor:
    """Run Canny on a batch. Returns [B,H,W,3] edge visualization in [0,1]."""
    if images.ndim == 3:
        images = images.unsqueeze(0)
    gray = rgb_to_gray_nchw(images)
    blurred = gaussian_blur_gray(gray, blur_sigma)
    mag, ang, _ = sobel_gradients(blurred)
    mag = mag / (mag.amax(dim=(2, 3), keepdim=True).clamp_min(1e-6))
    thin = non_max_suppression(mag, ang)
    edges = hysteresis_threshold(thin, float(low_threshold), float(high_threshold))
    edges = edges[:, 0].clamp(0, 1)
    return edges.unsqueeze(-1).expand(-1, -1, -1, 3).to(images.dtype)
