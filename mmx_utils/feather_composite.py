# MIT License — ComfyUI-H3-FaceRefine
# PORTED FROM: ComfyUI-H3-FaceRefine :: nodes.py @ HEAD

from __future__ import annotations

import torch
import torch.nn.functional as F


def gaussian_blur_mask(mask: torch.Tensor, feather: int) -> torch.Tensor:
    """Separable Gaussian blur on [B,1,H,W] or [1,1,H,W]."""
    if feather <= 0:
        return mask
    k = 2 * int(feather) + 1
    shortest = min(mask.shape[-2], mask.shape[-1])
    if shortest <= k:
        k = max(3, int(shortest / 2) | 1)
    sigma = max(k / 6.0, 0.5)
    x = torch.arange(k, device=mask.device, dtype=torch.float32) - k // 2
    g = torch.exp(-(x ** 2) / (2 * sigma ** 2))
    g = (g / g.sum()).to(mask.dtype)
    pad = k // 2
    m = F.conv2d(F.pad(mask, (pad, pad, 0, 0), mode="replicate"), g.view(1, 1, 1, k))
    m = F.conv2d(F.pad(m, (0, 0, pad, pad), mode="replicate"), g.view(1, 1, k, 1))
    return m


def subject_region_mask(
    ch: int,
    cw: int,
    rect: tuple[float, float, float, float],
    dilation: int,
    feather: int,
    ellipse: bool,
    device: torch.device,
) -> torch.Tensor:
    m = torch.zeros((1, 1, int(ch), int(cw)), device=device, dtype=torch.float32)
    fx, fy, fwd, fhd = rect
    fx -= dilation
    fy -= dilation
    fwd += 2 * dilation
    fhd += 2 * dilation
    if ellipse:
        yy = torch.arange(ch, device=device, dtype=torch.float32).view(-1, 1)
        xx = torch.arange(cw, device=device, dtype=torch.float32).view(1, -1)
        ccx, ccy = fx + fwd / 2.0, fy + fhd / 2.0
        rx, ry = max(fwd / 2.0, 1.0), max(fhd / 2.0, 1.0)
        m[0, 0] = (((xx - ccx) / rx) ** 2 + ((yy - ccy) / ry) ** 2 <= 1.0).float()
    else:
        x0 = max(0, int(round(fx)))
        y0 = max(0, int(round(fy)))
        x1 = min(int(cw), int(round(fx + fwd)))
        y1 = min(int(ch), int(round(fy + fhd)))
        if x1 > x0 and y1 > y0:
            m[0, 0, y0:y1, x0:x1] = 1.0
    return gaussian_blur_mask(m, feather).clamp(0, 1)


def colour_match_region(
    patch: torch.Tensor,
    base: torch.Tensor,
    mask: torch.Tensor,
    strength: float,
) -> torch.Tensor:
    if strength <= 0.0:
        return patch
    wsum = mask.sum(dim=(1, 2), keepdim=True).clamp_min(1e-6)
    bmu = (base * mask).sum(dim=(1, 2), keepdim=True) / wsum
    pmu = (patch * mask).sum(dim=(1, 2), keepdim=True) / wsum
    bsd = (((base - bmu) ** 2) * mask).sum(dim=(1, 2), keepdim=True) / wsum
    bsd = bsd.sqrt().clamp_min(1e-6)
    psd = (((patch - pmu) ** 2) * mask).sum(dim=(1, 2), keepdim=True) / wsum
    psd = psd.sqrt().clamp_min(1e-6)
    adj = (patch - pmu) * (bsd / psd) + bmu
    out = patch + (adj - patch) * float(strength)
    return out.clamp(0, 1)


def mask_confined_blend(
    plate: torch.Tensor,
    warped: torch.Tensor,
    mask: torch.Tensor,
    *,
    colour_match: float = 0.0,
) -> torch.Tensor:
    """Composite with invariant 5: exterior pixels are the literal plate tensor."""
    m = mask.float()
    if m.ndim == 3:
        m = m.unsqueeze(-1)
    plate_rgb = plate[..., :3].float()
    warped_rgb = warped[..., :3].float()
    # NaN/Inf in warped must not poison the exterior via 0*m arithmetic; interior
    # falls back to plate for non-finite samples before the weighted sum.
    warped_rgb = torch.where(torch.isfinite(warped_rgb), warped_rgb, plate_rgb)
    if m.shape[-1] == 1 and plate_rgb.shape[-1] == 3:
        m_blend = m.expand_as(plate_rgb)
    else:
        m_blend = m[..., : plate_rgb.shape[-1]]

    adjusted = warped_rgb
    if colour_match > 0.0:
        adjusted = colour_match_region(adjusted, plate_rgb, m_blend[..., :1], colour_match)

    blended = plate_rgb * (1.0 - m_blend) + adjusted * m_blend
    active = m_blend[..., :1] > 0
    active = active.expand_as(plate_rgb)
    out_rgb = torch.where(active, blended, plate_rgb)
    if not torch.isfinite(out_rgb).all():
        raise RuntimeError("Stitch produced non-finite pixels — check mask and crop inputs.")
    if plate.shape[-1] > 3:
        out = torch.cat([out_rgb.to(plate.dtype), plate[..., 3:]], dim=-1)
    else:
        out = out_rgb.to(plate.dtype)
    return out
