"""Detail match — measured MTF compensation + plate grain on decoded crops (N3)."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F

from .temporal import centred_window
from .color_space import linear_luma, srgb_to_linear
from .feather_composite import gaussian_blur_mask
from .frequency import laplacian_pyramid, reconstruct_pyramid
from .plate_restore import _build_ring, _grow_mask, _resize_hwc, _resize_mask_hw, interior_mask

_PYR_LEVELS = 5
_TEMPORAL_WINDOW = 5
_GRAIN_DEADBAND = 0.25
_NLF_BINS = 8


def _to_nchw(img: torch.Tensor) -> torch.Tensor:
    return img[..., :3].permute(2, 0, 1).unsqueeze(0).float()


def _from_nchw(x: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    return x.squeeze(0).permute(1, 2, 0).to(dtype)


def _mask_down(mask_hw: torch.Tensor, h: int, w: int) -> torch.Tensor:
    if mask_hw.shape[0] == h and mask_hw.shape[1] == w:
        return mask_hw
    m = mask_hw.unsqueeze(0).unsqueeze(0).float()
    m = F.interpolate(m, size=(h, w), mode="bilinear", align_corners=False)
    return m[0, 0]


def _band_energy(band: torch.Tensor, mask: torch.Tensor, ch: int) -> float:
    """Mean squared value of one channel band on masked pixels."""
    m = mask > 0.5
    if not m.any():
        return 0.0
    b = band[0, ch][m]
    return float((b * b).mean().item())


def _median_temporal(values: list[list[float]], i: int, window: int = _TEMPORAL_WINDOW) -> list[float]:
    lo, hi = centred_window(i, len(values), window)
    chunk = values[lo:hi]
    return [float(torch.tensor([row[k] for row in chunk]).median().item()) for k in range(len(values[0]))]


def _median3x3_nchw(x: torch.Tensor) -> torch.Tensor:
    c = x.shape[1]
    patches = F.unfold(F.pad(x, (1, 1, 1, 1), mode="reflect"), 3).view(c, 9, -1)
    return patches.median(dim=1).values.view(c, x.shape[2], x.shape[3]).unsqueeze(0)


def _box3x3(x: torch.Tensor) -> torch.Tensor:
    c = x.shape[1]
    k = torch.ones(c, 1, 3, 3, device=x.device, dtype=x.dtype) / 9.0
    return F.conv2d(F.pad(x, (1, 1, 1, 1), mode="reflect"), k, groups=c)


def _flat_mask(luma_hw: torch.Tensor) -> torch.Tensor:
    """Pixels with local gradient magnitude below the 40th percentile."""
    gy = torch.zeros_like(luma_hw)
    gx = torch.zeros_like(luma_hw)
    gy[1:-1, :] = (luma_hw[2:, :] - luma_hw[:-2, :]) * 0.5
    gx[:, 1:-1] = (luma_hw[:, 2:] - luma_hw[:, :-2]) * 0.5
    grad = torch.sqrt(gx * gx + gy * gy)
    thresh = torch.quantile(grad, 0.40)
    return grad < thresh


def _nlf_per_bin(
    img_hwc: torch.Tensor,
    region_mask: torch.Tensor,
    flat_mask: torch.Tensor,
    edges: torch.Tensor,
) -> list[float]:
    """Noise level function: per-channel std(residual) in luma bins.

    `edges` is SHARED by every caller for a frame. Each call used to derive
    its own edges from its own pixels' luma range, so the plate's bin 3, the
    generated crop's bin 3 and the grain synthesis's bin 3 were three
    different brightnesses, and the per-bin comparison compared nothing.
    """
    lin = srgb_to_linear(img_hwc[..., :3])
    luma = linear_luma(lin)
    residual = img_hwc[..., :3].float() - _from_nchw(_median3x3_nchw(_to_nchw(img_hwc)), img_hwc.dtype)
    sel = (region_mask > 0.5) & flat_mask
    if not sel.any():
        return [0.0] * _NLF_BINS

    nlfs: list[float] = []
    for b in range(_NLF_BINS):
        in_bin = sel & (luma >= edges[b]) & (luma < edges[b + 1] if b < _NLF_BINS - 1 else luma <= edges[b + 1])
        if not in_bin.any():
            nlfs.append(0.0)
            continue
        vals = residual[in_bin]
        nlfs.append(float(vals.std(unbiased=False).item()))
    return nlfs


def _interp_nlf_at_luma(luma: torch.Tensor, bin_edges: torch.Tensor, nlfs: list[float]) -> torch.Tensor:
    """Piecewise-linear NLF lookup per pixel."""
    t = torch.zeros_like(luma)
    for b in range(_NLF_BINS - 1):
        lo, hi = bin_edges[b], bin_edges[b + 1]
        w = ((luma - lo) / (hi - lo + 1e-8)).clamp(0, 1)
        in_seg = (luma >= lo) & (luma < hi)
        t = torch.where(in_seg, nlfs[b] + w * (nlfs[b + 1] - nlfs[b]), t)
    t = torch.where(luma >= bin_edges[-1], torch.tensor(nlfs[-1], device=luma.device), t)
    return t


# ── MTF compensation in the frequency domain ───────────────────────────────
#
# The first two attempts used a Laplacian pyramid, and both overshot. A 5-tap
# Gaussian / bilinear pyramid is not a clean filter bank: boosting its finest
# band also raises the next one once the image is re-decomposed, so open-loop
# gains left mid-frequencies at 1.44x the plate's energy, and a closed loop
# could not fix it (the fine gain hit its cap while the mid band, already
# over, was pinned at the "never blur" floor of 1.0).
#
# So the measurement and the correction now share ONE representation: the
# radial power spectrum. Spectra are Welch-averaged over Hann-windowed patches
# lying wholly inside the ring, for plate and generated; their ratio per
# radial frequency is the VAE's MTF loss in THIS shot; its square root (an
# amplitude) is applied as a single zero-phase FFT filter. A linear filter's
# response is exact, so what is measured is what is applied.

_MTF_BINS = 16
_MTF_PATCH = 32


def _radial_bins(n: int, device) -> torch.Tensor:
    """Radial frequency bin index for an n x n FFT grid, 0 .. _MTF_BINS-1."""
    f = torch.fft.fftfreq(n, device=device)
    r = torch.sqrt(f[:, None] ** 2 + f[None, :] ** 2)          # 0 .. ~0.707
    return (r / 0.5 * (_MTF_BINS - 1)).round().clamp(0, _MTF_BINS - 1).long()


_PATCH_MIN_COVER = 0.75


def _ring_patch_spectrum(img_hwc: torch.Tensor, ring: torch.Tensor, patch: int):
    """Welch-averaged radial power spectrum of luma over ring patches.

    A patch qualifies when at least 75% of it lies in the ring; the ring is
    applied INSIDE the window as a weight. Requiring patches wholly inside the
    ring left a single 16 px patch on a tight crop and the measurement fell
    back to "no gain". The same mask is applied to plate and generated, so
    whatever the mask edge does to the spectrum it does to both, and their
    ratio - the only thing used - is unaffected.

    Returns (spectrum [_MTF_BINS], counts [_MTF_BINS], number_of_patches).
    `counts` says which radial bins this patch size can populate at all; on a
    16-point grid some cannot, and they must not be read as zero power.
    """
    luma = linear_luma(srgb_to_linear(img_hwc[..., :3].float()))
    stride = max(4, patch // 4)
    h, w = luma.shape
    bins = _radial_bins(patch, luma.device)
    counts = torch.bincount(bins.reshape(-1), minlength=_MTF_BINS).float()
    empty = (torch.zeros(_MTF_BINS), counts.cpu(), 0)
    if h < patch or w < patch:
        return empty
    rf = ring.float().to(luma.device)
    cover = F.avg_pool2d(rf[None, None], patch, stride=stride)[0, 0]
    ys, xs = torch.nonzero(cover >= _PATCH_MIN_COVER, as_tuple=True)
    if ys.numel() == 0:
        return empty
    win = torch.hann_window(patch, periodic=False, device=luma.device)
    win2 = win[:, None] * win[None, :]
    acc = torch.zeros(_MTF_BINS, device=luma.device)
    for y, x in zip((ys * stride).tolist(), (xs * stride).tolist()):
        tile = luma[y:y + patch, x:x + patch]
        wgt = rf[y:y + patch, x:x + patch] * win2
        mean = (tile * wgt).sum() / wgt.sum().clamp_min(1e-8)
        pw = torch.fft.fft2((tile - mean) * wgt).abs() ** 2
        acc.index_add_(0, bins.reshape(-1), pw.reshape(-1))
    spec = acc / counts.clamp_min(1) / ys.numel()
    return spec.cpu(), counts.cpu(), int(ys.numel())


def _gain_field(curve: list[float], h: int, w: int, device) -> torch.Tensor:
    """Interpolate the radial gain curve onto an h x w FFT grid."""
    fy = torch.fft.fftfreq(h, device=device)
    fx = torch.fft.fftfreq(w, device=device)
    r = torch.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)
    pos = (r / 0.5 * (_MTF_BINS - 1)).clamp(0, _MTF_BINS - 1)
    lo = pos.floor().long()
    hi = (lo + 1).clamp(max=_MTF_BINS - 1)
    t = pos - lo.float()
    c = torch.tensor(curve, dtype=torch.float32, device=device)
    return c[lo] * (1 - t) + c[hi] * t


def _apply_mtf(
    gen_hwc: torch.Tensor,
    edit_feather: torch.Tensor,
    gains: list[float],
    sharpen: float,
) -> torch.Tensor:
    """Filter the crop by the radial gain curve and blend it in inside the
    feathered edit mask. Reflect-padded so the FFT does not wrap edges."""
    eff = [max(1.0, 1.0 + float(sharpen) * (float(g) - 1.0)) for g in gains]
    if max(eff) <= 1.0 + 1e-4:
        return gen_hwc
    h, w = gen_hwc.shape[0], gen_hwc.shape[1]
    pad = min(32, h - 1, w - 1)
    x = gen_hwc[..., :3].float().permute(2, 0, 1).unsqueeze(0)
    xp = F.pad(x, (pad, pad, pad, pad), mode="reflect")
    H, W = xp.shape[-2], xp.shape[-1]
    field = _gain_field(eff, H, W, xp.device)
    y = torch.fft.ifft2(torch.fft.fft2(xp) * field).real
    y = y[..., pad:pad + h, pad:pad + w][0].permute(1, 2, 0)
    m = edit_feather.to(y.device).unsqueeze(-1).float()
    rgb = gen_hwc[..., :3].float() * (1 - m) + y * m
    out = rgb.to(gen_hwc.dtype)
    if gen_hwc.shape[-1] > 3:
        return torch.cat([out, gen_hwc[..., 3:]], dim=-1)
    return out


def _add_grain(
    img_hwc: torch.Tensor,
    edit_feather: torch.Tensor,
    missing_nlf: list[float],
    luma_edges: torch.Tensor,
    grain_strength: float,
    frame_seed: int,
) -> torch.Tensor:
    if grain_strength <= 0.0 or max(missing_nlf) <= 1e-8:
        return img_hwc

    h, w = img_hwc.shape[0], img_hwc.shape[1]
    gen = torch.Generator(device="cpu").manual_seed(frame_seed)
    noise = torch.randn(h, w, 3, generator=gen, dtype=torch.float32)
    noise_nchw = _box3x3(noise.permute(2, 0, 1).unsqueeze(0))
    noise_hwc = noise_nchw.squeeze(0).permute(1, 2, 0)
    noise_hwc = noise_hwc - noise_hwc.mean(dim=(0, 1), keepdim=True)
    # CALIBRATE through the same estimator that measured the deficit. The
    # box filter divides white noise's std by 3, and the median-residual
    # estimator sees only part of what remains - so scaling raw filtered
    # noise by the "missing" NLF added about a third of what was missing.
    # Scale so that the ESTIMATOR reads the synthetic grain as exactly 1.0;
    # then multiplying by `missing` adds what the estimator said was missing.
    resid = noise_hwc - _from_nchw(_median3x3_nchw(_to_nchw(noise_hwc)), noise_hwc.dtype)
    response = resid.std(dim=(0, 1), unbiased=False).clamp_min(1e-6)
    noise_hwc = noise_hwc / response

    lin = srgb_to_linear(img_hwc[..., :3])
    luma = linear_luma(lin)
    scale = _interp_nlf_at_luma(luma, luma_edges, missing_nlf) * float(grain_strength)
    scaled = noise_hwc * scale.unsqueeze(-1)

    m = edit_feather.unsqueeze(-1).to(img_hwc.dtype)
    out_rgb = img_hwc[..., :3].float() + scaled * m
    if img_hwc.shape[-1] > 3:
        return torch.cat([out_rgb.to(img_hwc.dtype), img_hwc[..., 3:]], dim=-1)
    return out_rgb.to(img_hwc.dtype)


def _lowpass_contrast_ratio(plate_hwc: torch.Tensor, gen_hwc: torch.Tensor,
                            ring: torch.Tensor, sigma: float = 4.0) -> float:
    """Energy ratio plate/generated of the ring's heavily low-passed luma.

    This is the contrast reference, and it has to be measured where no MTF
    acts. Taking it from the lowest spectral bin was wrong on tight crops:
    only 16 px patches fit, whose lowest usable bin sits at 0.067 cycles/px,
    where a 1.2 px softening already costs ~12% amplitude - so part of the
    genuine sharpness loss was written off as "contrast" and never restored.
    A sigma-4 Gaussian passes nothing above ~0.04 cycles/px.
    """
    def lp(img):
        luma = linear_luma(srgb_to_linear(img[..., :3].float()))[None, None]
        r = max(1, int(round(3 * sigma)))
        x = torch.arange(-r, r + 1, dtype=torch.float32, device=luma.device)
        k = torch.exp(-(x ** 2) / (2 * sigma ** 2))
        k = k / k.sum()
        t = F.conv2d(F.pad(luma, (r, r, 0, 0), mode="reflect"), k.view(1, 1, 1, -1))
        t = F.conv2d(F.pad(t, (0, 0, r, r), mode="reflect"), k.view(1, 1, -1, 1))
        return t[0, 0]

    sel = ring > 0.5
    if int(sel.sum()) < 16:
        return 1.0
    a = lp(plate_hwc)[sel]
    b = lp(gen_hwc)[sel]
    va = float(a.var(unbiased=False))
    vb = float(b.var(unbiased=False))
    if va <= 1e-12 or vb <= 1e-12:
        return 1.0
    return va / vb


def _compute_level_gains(
    plate_hwc: torch.Tensor,
    gen_hwc: torch.Tensor,
    ring: torch.Tensor,
    max_gain: float,
) -> list[float]:
    """Radial amplitude gain curve (length _MTF_BINS) from ring spectra.

    Normalised by the lowest non-DC bin: a contrast difference belongs to
    PlateRestore's colour affine, not to sharpness. Never below 1 (this node
    does not blur), never above max_gain. Bins where either spectrum is at the
    numerical floor keep gain 1 - there is nothing reliable to invert there.
    Falls back to smaller patches on a thin ring, and to all-ones if even those
    do not fit.
    """
    for patch in (_MTF_PATCH, 16):
        ps, cnt, n_p = _ring_patch_spectrum(plate_hwc, ring, patch)
        gs, _, n_g = _ring_patch_spectrum(gen_hwc, ring, patch)
        if n_p >= 2 and n_g >= 2:
            break
    else:
        return [1.0] * _MTF_BINS
    floor = 1e-12
    valid = [k for k in range(1, _MTF_BINS)
             if cnt[k] > 0 and ps[k] > floor and gs[k] > floor]
    if len(valid) < 2:
        return [1.0] * _MTF_BINS
    # Contrast reference from the low-passed ring, not from a spectral bin
    # (see _lowpass_contrast_ratio). Bins at or below the first populated one
    # carry no reliable MTF information and keep gain 1.
    ref = valid[0]
    base = _lowpass_contrast_ratio(plate_hwc, gen_hwc, ring)
    raw: dict[int, float] = {}
    for k in valid:
        ratio = (float(ps[k]) / float(gs[k])) / base
        raw[k] = min(max(math.sqrt(max(ratio, 0.0)), 1.0), float(max_gain))
    # Bins this patch size cannot populate take their nearest measured
    # neighbour, so the curve stays smooth instead of dropping to 1.0 and
    # notching the filter.
    curve: list[float] = []
    for k in range(_MTF_BINS):
        if k <= ref:
            curve.append(1.0)
        elif k in raw:
            curve.append(raw[k])
        else:
            near = min(raw, key=lambda j: abs(j - k))
            curve.append(raw[near])
    return curve


def detail_match_frames(
    cropped_images: torch.Tensor,
    original_images: torch.Tensor,
    boxes: list[dict[str, float | int]],
    edit_mask: torch.Tensor | None,
    grow_px: int,
    sharpen: float,
    max_gain: float,
    grain: float,
    seed: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Run MTF + grain match on raw decoded crops (before PlateRestore)."""
    n = min(cropped_images.shape[0], original_images.shape[0], len(boxes))
    out = cropped_images[:n].clone()
    no_edit = edit_mask is None

    per_frame_gains: list[list[float]] = []
    per_frame: list[dict[str, Any]] = []

    # Pass 1 — measure per-frame gains and NLF
    frame_ctx: list[dict[str, Any]] = []
    for i in range(n):
        b = boxes[i]
        x, y, w, h = (int(round(b[k])) for k in ("x", "y", "width", "height"))
        plate_crop = original_images[i, y : y + h, x : x + w, :]
        gen_crop = _resize_hwc(cropped_images[i], h, w)
        em = (
            torch.ones(h, w, device=out.device, dtype=torch.float32)
            if no_edit
            else _resize_mask_hw(edit_mask[min(i, edit_mask.shape[0] - 1)], h, w).clamp(0, 1)
        )
        interior = interior_mask(h, w).to(out.device)
        ring, ring_count, ring_ok = _build_ring(em, grow_px, interior)
        edit_grown = _grow_mask(em, grow_px)
        edit_feather = gaussian_blur_mask(edit_grown.unsqueeze(0).unsqueeze(0), max(1, grow_px // 2))[0, 0]

        gains = _compute_level_gains(plate_crop, gen_crop, ring, max_gain) if ring_ok else [1.0] * _MTF_BINS
        per_frame_gains.append(gains)

        # Grain loss is measured LIKE FOR LIKE: plate ring vs generated ring,
        # the same pixels, both valid - the VAE degraded the whole crop, not
        # just the edit. It used to compare the plate's ring with the
        # generated EDIT region: different pixels, different content. And
        # since sqrt(max(0, p^2 - g^2)) can only ADD, any estimation noise
        # between two regions became added grain - an identical input came
        # out grainier. The deficit measured here is then applied inside the
        # edit, exactly as the MTF gains are.
        plate_luma = linear_luma(srgb_to_linear(plate_crop[..., :3]))
        flat_plate = _flat_mask(plate_luma)
        flat_gen = _flat_mask(linear_luma(srgb_to_linear(gen_crop[..., :3])))
        if ring_ok:
            rv = plate_luma[ring > 0.5]
            lo, hi = float(rv.min()), float(rv.max())
        else:
            lo, hi = 0.0, 1.0
        edges = torch.linspace(lo, max(hi, lo + 1e-6) + 1e-6, _NLF_BINS + 1, device=plate_luma.device)
        nlf_plate = _nlf_per_bin(plate_crop, ring, flat_plate, edges) if ring_ok else [0.0] * _NLF_BINS
        nlf_gen = _nlf_per_bin(gen_crop, ring, flat_gen, edges) if ring_ok else [0.0] * _NLF_BINS

        frame_ctx.append(
            {
                "gen_crop": gen_crop,
                "edit_feather": edit_feather,
                "ring_ok": ring_ok,
                "ring_count": ring_count,
                "nlf_plate": nlf_plate,
                "nlf_gen": nlf_gen,
                "gains": gains,
                "edges": edges,
            }
        )

    # Pass 2 — apply temporally smoothed gains + grain
    for i, ctx in enumerate(frame_ctx):
        gains_sm = _median_temporal(per_frame_gains, i) if per_frame_gains else ctx["gains"]
        working = ctx["gen_crop"].clone()

        if ctx["ring_ok"] and sharpen > 0:
            working = _apply_mtf(working, ctx["edit_feather"], gains_sm, sharpen)

        # A deficit smaller than a quarter of the plate's own grain is inside
        # the estimator's error on a few hundred ring pixels; adding "grain"
        # for it would add noise to a region that has none missing.
        missing = [
            (math.sqrt(max(0.0, p * p - g * g))
             if p > 0.0 and (p - g) > _GRAIN_DEADBAND * p else 0.0)
            for p, g in zip(ctx["nlf_plate"], ctx["nlf_gen"])
        ]
        edges = ctx["edges"]

        working = _add_grain(working, ctx["edit_feather"], missing, edges, grain, seed + i)

        # Write back at crop resolution (geometry unchanged)
        if working.shape[0] != cropped_images[i].shape[0] or working.shape[1] != cropped_images[i].shape[1]:
            out[i] = _resize_hwc(working, cropped_images[i].shape[0], cropped_images[i].shape[1])
        else:
            out[i] = working.to(cropped_images.dtype)

        per_frame.append(
            {
                "frame": i,
                "ring_pixels": ctx["ring_count"],
                "gains": [round(g, 4) for g in gains_sm],
                "nlf_plate": [round(v, 5) for v in ctx["nlf_plate"]],
                "nlf_gen": [round(v, 5) for v in ctx["nlf_gen"]],
                "grain_missing_rms": [round(v, 5) for v in missing],
            }
        )

    summary = (
        f"frames={n} sharpen={sharpen:g} max_gain={max_gain:g} grain={grain:g} "
        f"median_gains_L0={per_frame[0]['gains'][0] if per_frame else 1.0}"
    )
    report: dict[str, Any] = {"frames": per_frame, "summary": summary}
    return out, report
