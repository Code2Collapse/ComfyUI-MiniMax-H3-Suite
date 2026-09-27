"""Colour QC metrics for H3 Hybrid HDR A/B — Hasler-Süsstrunk, luma DR, texture, flicker, Oklab."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from mmx_utils.color_space import linear_luma, linear_srgb_to_oklab, srgb_to_linear

_EPS = 1e-6


@dataclass
class ColorQCMetrics:
    colourfulness: float
    colourfulness_band: str
    luma_p1: float
    luma_p50: float
    luma_p99: float
    clipped_fraction: float
    headroom: float
    dynamic_range_stops: float
    texture_energy: float
    temporal_flicker: float
    oklab_mean_dist: float | None
    oklab_std_dist: float | None


def colourfulness_band(value: float) -> str:
    if value < 15:
        return "not colourful (<15)"
    if value < 33:
        return "slightly colourful (15-33)"
    if value < 45:
        return "moderately colourful (33-45)"
    if value < 59:
        return "averagely colourful (45-59)"
    if value < 82:
        return "quite colourful (59-82)"
    if value < 109:
        return "highly colourful (82-109)"
    return "extremely colourful (>109)"


def hasler_susstrunk_display(images: torch.Tensor) -> tuple[float, str]:
    """
    Hasler-Süsstrunk on display sRGB scaled to 0-255 (NOT linear light).
    rg = R-G, yb = 0.5(R+G)-B; M = sqrt(sigma^2) + 0.3*sqrt(mu^2).
    """
    rgb = images[..., :3].float().clamp(0.0, 1.0) * 255.0
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    rg = r - g
    yb = 0.5 * (r + g) - b
    sigma_rg = rg.std(unbiased=False)
    sigma_yb = yb.std(unbiased=False)
    mu_rg = rg.mean()
    mu_yb = yb.mean()
    m = torch.sqrt(sigma_rg * sigma_rg + sigma_yb * sigma_yb) + 0.3 * torch.sqrt(
        mu_rg * mu_rg + mu_yb * mu_yb
    )
    val = float(m.item())
    return val, colourfulness_band(val)


def _percentile(x: torch.Tensor, q: float) -> float:
    flat = x.reshape(-1).float()
    if flat.numel() == 0:
        return 0.0
    return float(torch.quantile(flat, q).item())


def _laplacian_luma(luma_hw: torch.Tensor) -> torch.Tensor:
    x = luma_hw.unsqueeze(0).unsqueeze(0)
    k = torch.tensor(
        [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
        device=x.device,
        dtype=x.dtype,
    ).view(1, 1, 3, 3)
    return F.conv2d(F.pad(x, (1, 1, 1, 1), mode="replicate"), k)[0, 0]


def _box_lowpass_luma(luma_hw: torch.Tensor, radius: int = 4) -> torch.Tensor:
    x = luma_hw.unsqueeze(0).unsqueeze(0)
    k = 2 * radius + 1
    kernel = torch.ones(1, 1, k, k, device=x.device, dtype=x.dtype) / (k * k)
    x = F.conv2d(F.pad(x, (radius, radius, radius, radius), mode="replicate"), kernel)
    return x[0, 0]


def measure_color_qc(
    images: torch.Tensor,
    reference: torch.Tensor | None = None,
) -> tuple[ColorQCMetrics, str]:
    if images.ndim == 3:
        images = images.unsqueeze(0)
    b = images.shape[0]

    tex_vals = []
    luma_p1s, luma_p50s, luma_p99s = [], [], []
    clip_fracs = []
    headrooms = []
    dr_stops = []
    flicker_vals: list[float] = []
    prev_lp: torch.Tensor | None = None

    for i in range(b):
        frame = images[i]
        linear = srgb_to_linear(frame)
        luma = linear_luma(linear)
        p1 = _percentile(luma, 0.01)
        p50 = _percentile(luma, 0.50)
        p99 = _percentile(luma, 0.99)
        luma_p1s.append(p1)
        luma_p50s.append(p50)
        luma_p99s.append(p99)

        mx = frame[..., :3].amax(dim=-1)
        clip_fracs.append(float((mx >= 0.995).float().mean().item()))
        headrooms.append(float(max(0.0, 1.0 - p99)))
        dr_stops.append(float(math.log2(max(p99, 1e-4) / max(p1, 1e-4))))

        lap = _laplacian_luma(luma)
        local_mean = luma.mean().clamp(min=_EPS)
        tex_vals.append(float(lap.var(unbiased=False).item() / local_mean.item()))

        lp = _box_lowpass_luma(luma)
        if prev_lp is not None:
            flicker_vals.append(float((lp - prev_lp).abs().mean().item()))
        prev_lp = lp

    cf_mean, band = hasler_susstrunk_display(images)

    oklab_mean_dist = None
    oklab_std_dist = None
    if reference is not None:
        if reference.ndim == 3:
            reference = reference.unsqueeze(0)
        ref_frame = reference[0]
        ok_ref = linear_srgb_to_oklab(srgb_to_linear(ref_frame))
        ok_all = linear_srgb_to_oklab(srgb_to_linear(images))
        ref_mean = ok_ref.mean(dim=(0, 1))
        ref_std = ok_ref.std(dim=(0, 1), unbiased=False)
        clip_mean = ok_all.mean(dim=(1, 2)).mean(dim=0)
        clip_std = ok_all.std(dim=(1, 2), unbiased=False).mean(dim=0)
        oklab_mean_dist = float((clip_mean - ref_mean).pow(2).sum().sqrt().item())
        oklab_std_dist = float((clip_std - ref_std).pow(2).sum().sqrt().item())

    metrics = ColorQCMetrics(
        colourfulness=cf_mean,
        colourfulness_band=band,
        luma_p1=float(sum(luma_p1s) / len(luma_p1s)),
        luma_p50=float(sum(luma_p50s) / len(luma_p50s)),
        luma_p99=float(sum(luma_p99s) / len(luma_p99s)),
        clipped_fraction=float(sum(clip_fracs) / len(clip_fracs)),
        headroom=float(sum(headrooms) / len(headrooms)),
        dynamic_range_stops=float(sum(dr_stops) / len(dr_stops)),
        texture_energy=float(sum(tex_vals) / len(tex_vals)),
        temporal_flicker=float(sum(flicker_vals) / max(1, len(flicker_vals))),
        oklab_mean_dist=oklab_mean_dist,
        oklab_std_dist=oklab_std_dist,
    )

    lines = [
        "H3 Color QC",
        f"  colourfulness: {metrics.colourfulness:.2f} ({metrics.colourfulness_band})",
        f"  linear luma p1/p50/p99: {metrics.luma_p1:.4f} / {metrics.luma_p50:.4f} / {metrics.luma_p99:.4f}",
        f"  clipped_fraction (max RGB >= 0.995): {metrics.clipped_fraction:.4f}",
        f"  headroom (1 - p99 luma): {metrics.headroom:.4f}",
        f"  dynamic_range_stops: {metrics.dynamic_range_stops:.2f}",
        f"  texture_energy: {metrics.texture_energy:.6f}",
        f"  temporal_flicker: {metrics.temporal_flicker:.6f}",
    ]
    if oklab_mean_dist is not None:
        lines.append(f"  oklab_mean_dist: {oklab_mean_dist:.4f}")
        lines.append(f"  oklab_std_dist: {oklab_std_dist:.4f}")
    return metrics, "\n".join(lines)


def build_preview_strip(images: torch.Tensor, metrics: ColorQCMetrics) -> torch.Tensor:
    if images.ndim == 3:
        images = images.unsqueeze(0)
    h, w = 48, 256
    canvas = torch.zeros(h, w, 3)
    x0 = int(metrics.luma_p1 * (w - 1))
    x99 = int(min(1.0, metrics.luma_p99) * (w - 1))
    canvas[8:16, x0 : x99 + 1, :] = 0.7
    clip_w = max(1, int(metrics.clipped_fraction * w))
    canvas[20:28, :clip_w, 0] = 1.0
    cf_w = max(1, int(min(1.0, metrics.colourfulness / 109.0) * w))
    canvas[32:40, :cf_w, 1] = 0.8
    thumb = F.interpolate(
        images[:1].permute(0, 3, 1, 2),
        size=(h, min(w // 4, w)),
        mode="bilinear",
        align_corners=False,
    )[0].permute(1, 2, 0)
    tw = thumb.shape[1]
    canvas[:, -tw:, :] = thumb.clamp(0, 1)
    return canvas.unsqueeze(0)
