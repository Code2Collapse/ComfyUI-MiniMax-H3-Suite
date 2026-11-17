"""Colour QC metrics for H3 Hybrid HDR A/B — Hasler-Süsstrunk, luma DR, texture, flicker, Oklab."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

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
    # L8.06 / L8.12 extension: per-frame trends, histograms, banding, deltaE vs reference (see measure_color_qc)
    extra: dict = field(default_factory=dict)


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


HIST_BINS = 16
CHROMA_MAX = 0.32          # Oklab chroma of the sRGB primaries tops out near 0.32 (blue)
BAND_PLATEAU = 8           # a visible band: >= 8 px flat on both sides of a one-code step


def _hasler_sums(frame: torch.Tensor) -> torch.Tensor:
    """float64 [n, sum rg, sum rg^2, sum yb, sum yb^2] of one display frame (0-255): the clip score is exact without
    holding the whole clip in float (the old path converted the full clip at once)."""
    rgb = frame[..., :3].double().clamp(0.0, 1.0) * 255.0
    rg = rgb[..., 0] - rgb[..., 1]
    yb = 0.5 * (rgb[..., 0] + rgb[..., 1]) - rgb[..., 2]
    n = torch.tensor(float(rg.numel()), dtype=torch.float64, device=rg.device)
    return torch.stack([n, rg.sum(), (rg * rg).sum(), yb.sum(), (yb * yb).sum()]).cpu()


def _hasler_from_sums(s: torch.Tensor) -> float:
    n = max(float(s[0]), 1.0)
    mu_rg, mu_yb = float(s[1]) / n, float(s[3]) / n
    var_rg = max(float(s[2]) / n - mu_rg * mu_rg, 0.0)
    var_yb = max(float(s[4]) / n - mu_yb * mu_yb, 0.0)
    return math.sqrt(var_rg + var_yb) + 0.3 * math.sqrt(mu_rg * mu_rg + mu_yb * mu_yb)


def banding_edges(frame: torch.Tensor, plateau: int = BAND_PLATEAU) -> tuple[int, int]:
    """HEURISTIC false-contour count on 8-bit display luma: one-code steps with >= `plateau` flat pixels on both
    sides, horizontally and vertically. Returns (edges, pixels). A smooth gradient quantised to 8 bit scores high; the
    same gradient dithered, a flat field and real texture score ~0. Not a perceptual banding model."""
    rgb = frame[..., :3].float().clamp(0.0, 1.0)
    y = (0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]) * 255.0
    codes = torch.round(y)
    edges = 0
    for c in (codes, codes.transpose(0, 1)):
        if c.shape[1] < 2 * plateau + 2:
            continue
        d = c[:, 1:] - c[:, :-1]                                              # [H, W-1]
        flat = (d == 0).float().unsqueeze(1)                                  # [H, 1, W-1]
        win = F.conv1d(flat, torch.ones(1, 1, plateau, device=flat.device)).squeeze(1)
        full = win >= plateau - 0.5                                           # steps k..k+plateau-1 all flat
        j = torch.arange(plateau, d.shape[1] - plateau, device=d.device)
        ok = (d[:, j].abs() == 1) & full[:, j - plateau] & full[:, j + 1]
        edges += int(ok.sum().item())
    return edges, int(codes.numel())


def _slope(values: list[float]) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    xm = (n - 1) / 2.0
    ym = sum(values) / n
    den = sum((i - xm) ** 2 for i in range(n))
    return sum((i - xm) * (v - ym) for i, v in enumerate(values)) / den


def _reference_for(reference: torch.Tensor, i: int, h: int, w: int) -> torch.Tensor:
    """The reference frame for frame i: the same frame when the reference is a clip of the same length, else its first
    frame; resized (area) when its size differs."""
    ref = reference[i] if reference.shape[0] > 1 and reference.shape[0] > i else reference[0]
    ref = ref[..., :3]
    if ref.shape[0] != h or ref.shape[1] != w:
        ref = F.interpolate(ref.permute(2, 0, 1).unsqueeze(0).float(), size=(h, w), mode="area")[0].permute(1, 2, 0)
    return ref


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
    hs_total = None
    cf_frames: list[float] = []
    ok_means: list[torch.Tensor] = []
    ok_stds: list[torch.Tensor] = []
    drift: list[float] = []
    sat_hist = torch.zeros(HIST_BINS, dtype=torch.float64)
    luma_hist = torch.zeros(HIST_BINS, dtype=torch.float64)
    band_edges, band_px = 0, 0
    de_frames: list[float] = []
    if reference is not None and reference.ndim == 3:
        reference = reference.unsqueeze(0)

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

        hs = _hasler_sums(frame)
        hs_total = hs if hs_total is None else hs_total + hs
        cf_frames.append(_hasler_from_sums(hs))
        ok = linear_srgb_to_oklab(linear[..., :3])
        ok_means.append(ok.mean(dim=(0, 1)).double().cpu())
        ok_stds.append(ok.std(dim=(0, 1), unbiased=False).double().cpu())
        if len(ok_means) > 1:
            drift.append(float((ok_means[-1] - ok_means[-2]).pow(2).sum().sqrt()))
        chroma = ok[..., 1:3].pow(2).sum(-1).sqrt()
        sat_hist += torch.histc(chroma.float().cpu().clamp(0, CHROMA_MAX), HIST_BINS, 0, CHROMA_MAX).double()
        disp = frame[..., :3].float().clamp(0, 1)
        y_disp = 0.2126 * disp[..., 0] + 0.7152 * disp[..., 1] + 0.0722 * disp[..., 2]
        luma_hist += torch.histc(y_disp.cpu(), HIST_BINS, 0, 1).double()
        e, px = banding_edges(frame)
        band_edges += e
        band_px += px
        if reference is not None:
            ref = _reference_for(reference.to(frame.device), i, frame.shape[0], frame.shape[1])
            ok_ref = linear_srgb_to_oklab(srgb_to_linear(ref))
            de_frames.append(float((ok - ok_ref).pow(2).sum(-1).sqrt().mean().item()) * 100.0)

    # same definitions as before, accumulated per frame (equal up to float rounding; no whole-clip float copies)
    cf_mean = _hasler_from_sums(hs_total)
    band = colourfulness_band(cf_mean)

    oklab_mean_dist = None
    oklab_std_dist = None
    if reference is not None:
        ok_ref = linear_srgb_to_oklab(srgb_to_linear(reference[0][..., :3]))
        ref_mean = ok_ref.mean(dim=(0, 1)).double().cpu()
        ref_std = ok_ref.std(dim=(0, 1), unbiased=False).double().cpu()
        clip_mean = torch.stack(ok_means).mean(dim=0)
        clip_std = torch.stack(ok_stds).mean(dim=0)
        oklab_mean_dist = float((clip_mean - ref_mean).pow(2).sum().sqrt().item())
        oklab_std_dist = float((clip_std - ref_std).pow(2).sum().sqrt().item())

    def _r(x, k=4):
        return round(float(x), k)

    def _frac(h):
        return [_r(v) for v in (h / max(float(h.sum()), 1.0)).tolist()]

    worst = max(range(len(drift)), key=lambda k: drift[k]) if drift else None
    extra = {
        "frames": b,
        "colourfulness_per_frame": [_r(v, 2) for v in cf_frames],
        "colourfulness_min": _r(min(cf_frames), 2),
        "colourfulness_max": _r(max(cf_frames), 2),
        "colourfulness_slope_per_frame": _r(_slope(cf_frames), 4),
        "oklab_drift_per_frame": [_r(v, 5) for v in drift],
        "oklab_drift_mean": _r(sum(drift) / len(drift), 5) if drift else 0.0,
        "oklab_drift_max": _r(drift[worst], 5) if drift else 0.0,
        "oklab_drift_max_at_frame": (worst + 1) if drift else None,
        "chroma_hist": {"range": [0.0, CHROMA_MAX], "space": "Oklab chroma", "fractions": _frac(sat_hist)},
        "luma_hist": {"range": [0.0, 1.0], "space": "display (sRGB-encoded) Rec.709 luma", "fractions": _frac(luma_hist)},
        "banding_edges_per_10k_px": _r(band_edges * 1e4 / max(band_px, 1), 3),
        "banding_note": "heuristic: one-code 8-bit luma steps with >= 8 flat px on both sides",
    }
    if de_frames:
        worst_de = max(range(len(de_frames)), key=lambda k: de_frames[k])
        srt = sorted(de_frames)
        extra.update({
            "delta_e_ok_per_frame": [_r(v, 3) for v in de_frames],
            "delta_e_ok_mean": _r(sum(de_frames) / len(de_frames), 3),
            "delta_e_ok_p95": _r(srt[int(round(0.95 * (len(srt) - 1)))], 3),
            "delta_e_ok_max": _r(de_frames[worst_de], 3),
            "delta_e_ok_max_at_frame": worst_de,
            "delta_e_note": "per-pixel Oklab distance x100 against the reference (the same frame when the reference is "
                            "a clip of equal length, else its first frame); not CIEDE2000",
        })

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
        extra=extra,
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
    lines.append(f"  colourfulness per frame: min {extra['colourfulness_min']:.2f} / max "
                 f"{extra['colourfulness_max']:.2f} / slope {extra['colourfulness_slope_per_frame']:+.3f} per frame")
    if drift:
        lines.append(f"  frame-to-frame Oklab drift: mean {extra['oklab_drift_mean']:.5f}, max "
                     f"{extra['oklab_drift_max']:.5f} at frame {extra['oklab_drift_max_at_frame']}")
    lines.append(f"  banding (heuristic): {extra['banding_edges_per_10k_px']:.3f} contour edges per 10k px")
    if de_frames:
        lines.append(f"  deltaE_OK vs reference: mean {extra['delta_e_ok_mean']:.3f}, p95 {extra['delta_e_ok_p95']:.3f}, "
                     f"max {extra['delta_e_ok_max']:.3f} at frame {extra['delta_e_ok_max_at_frame']}")
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
