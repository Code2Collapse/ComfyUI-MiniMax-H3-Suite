# Plate restore / uncrop composite maths (linear colour, masked registration).

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F

from .temporal import centred_window
from .color_space import linear_luma, linear_to_srgb, srgb_to_linear
from .drift_qc import inject_pixel_shift
from .feather_composite import gaussian_blur_mask, mask_confined_blend
from .registration import masked_ncc_shift

_RING_MIN_FRAC = 0.01
_AFFINE_COND_MAX = 1e3
_TEMPORAL_WINDOW = 5


def border_ramp_alpha(
    h: int,
    w: int,
    x: int,
    y: int,
    img_h: int,
    img_w: int,
    feather: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Feather ramp [h,w] in crop space (1 in interior, 0 at crop edges touching frame edge)."""
    alpha = torch.ones(h, w, dtype=dtype, device=device)
    fx, fy = min(feather, w // 2), min(feather, h // 2)
    if fy > 0:
        ramp = torch.linspace(0, 1, fy + 2, dtype=dtype, device=device)[1:-1]
        if y > 0:
            alpha[:fy, :] *= ramp[:, None]
        if y + h < img_h:
            alpha[h - fy :, :] *= ramp.flip(0)[:, None]
    if fx > 0:
        ramp = torch.linspace(0, 1, fx + 2, dtype=dtype, device=device)[1:-1]
        if x > 0:
            alpha[:, :fx] *= ramp[None, :]
        if x + w < img_w:
            alpha[:, w - fx :] *= ramp.flip(0)[None, :]
    return alpha


_EVIDENCE_INSET_PX = 2
_SHIFT_DEADBAND_PX = 0.01
_AFFINE_MAX_SAMPLES = 200_000


def interior_mask(h: int, w: int) -> torch.Tensor:
    """Where inside the crop a pixel may be used as EVIDENCE: everywhere but a
    thin inset from the crop border.

    This used to also exclude the whole border-feather band, which confused
    compositing with measurement. The feather band matters for BLENDING the
    result back; for measuring offset and colour, the generated crop and the
    plate are both fully valid there - and it is the best background evidence
    the crop has, since the subject sits in the middle. On a 64 px crop,
    grow_px=8 plus a 12 px band per side left zero ring pixels, so
    registration and colour silently never ran.

    The 2 px inset only keeps resampling edge effects (the crop was resized
    on the way in) out of the fit.
    """
    m = torch.zeros(h, w, dtype=torch.float32)
    i = _EVIDENCE_INSET_PX
    if h > 2 * i and w > 2 * i:
        m[i : h - i, i : w - i] = 1.0
    return m


def paste_crop_with_feather(
    frame: torch.Tensor,
    crop: torch.Tensor,
    x: int,
    y: int,
    w: int,
    h: int,
    feather: int,
    img_h: int,
    img_w: int,
    cropped_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Paste *crop* into *frame* at (x,y) with SubjectUncrop feather semantics.

    *crop* must already be resized to (h, w). Returns a new frame tensor.
    """
    out = frame.clone()
    alpha = border_ramp_alpha(h, w, x, y, img_h, img_w, feather, device=out.device, dtype=out.dtype)
    if cropped_mask is not None:
        alpha = alpha * cropped_mask.clamp(0.0, 1.0).to(dtype=out.dtype, device=out.device)
    alpha = alpha[..., None]
    region = out[y : y + h, x : x + w, :]
    out[y : y + h, x : x + w, :] = crop.to(region) * alpha + region * (1.0 - alpha)
    return out


def uncrop_paste_frames(
    cropped_images: torch.Tensor,
    original_images: torch.Tensor,
    boxes: list[dict[str, float | int]],
    feather: int,
    cropped_masks: torch.Tensor | None = None,
    *,
    resize_image,
    resize_mask,
) -> torch.Tensor:
    """Batch uncrop paste (SubjectUncrop body)."""
    img_h, img_w = original_images.shape[1], original_images.shape[2]
    out = original_images.clone()
    for i, b in enumerate(boxes):
        x, y, w, h = (int(round(b[k])) for k in ("x", "y", "width", "height"))
        crop = resize_image(cropped_images[i], w, h)
        m = resize_mask(cropped_masks[i], w, h) if cropped_masks is not None else None
        out[i] = paste_crop_with_feather(
            out[i], crop, x, y, w, h, feather, img_h, img_w, cropped_mask=m
        )
    return out


def _grow_mask(mask: torch.Tensor, px: int) -> torch.Tensor:
    """Dilate by px with a square element - ONE pooling pass.

    `px` iterated 3x3 max-pools compose to exactly a (2px+1) square, so a
    single pool gives the identical result. The loop took 1.3 s per call on a
    2 MP crop.
    """
    if px <= 0:
        return mask
    k = 2 * int(px) + 1
    x = mask.unsqueeze(0).unsqueeze(0)
    x = F.max_pool2d(x, kernel_size=(1, k), stride=1, padding=(0, px))
    x = F.max_pool2d(x, kernel_size=(k, 1), stride=1, padding=(px, 0))
    return x[0, 0]


def _erode_mask(mask: torch.Tensor, px: int) -> torch.Tensor:
    if px <= 0:
        return mask
    # One separable pass; identical to px iterated 3x3 erosions (see _grow_mask).
    k = 2 * int(px) + 1
    x = -mask.unsqueeze(0).unsqueeze(0)
    x = F.max_pool2d(x, kernel_size=(1, k), stride=1, padding=(0, px))
    x = F.max_pool2d(x, kernel_size=(k, 1), stride=1, padding=(px, 0))
    return -x[0, 0]


def _resize_hwc(img: torch.Tensor, h: int, w: int) -> torch.Tensor:
    if img.shape[0] == h and img.shape[1] == w:
        return img
    x = img.movedim(-1, 0).unsqueeze(0).float()
    x = F.interpolate(x, size=(h, w), mode="bilinear", align_corners=False)
    return x.squeeze(0).movedim(0, -1).to(img.dtype)


def _resize_mask_hw(mask: torch.Tensor, h: int, w: int) -> torch.Tensor:
    if mask.shape[0] == h and mask.shape[1] == w:
        return mask
    x = mask.unsqueeze(0).unsqueeze(0).float()
    x = F.interpolate(x, size=(h, w), mode="bilinear", align_corners=False)
    return x[0, 0].to(mask.dtype)


def _identity_affine(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    mat = torch.zeros(3, 4, device=device, dtype=dtype)
    mat[0, 0] = 1.0
    mat[1, 1] = 1.0
    mat[2, 2] = 1.0
    return mat


def _mean_offset_matrix(plate_lin: torch.Tensor, gen_lin: torch.Tensor, ring: torch.Tensor) -> torch.Tensor:
    sel = ring > 0.5
    w = sel.float().sum().clamp_min(1.0)
    pm = (plate_lin * sel.unsqueeze(-1)).sum(dim=(0, 1)) / w
    gm = (gen_lin * sel.unsqueeze(-1)).sum(dim=(0, 1)) / w
    mat = _identity_affine(plate_lin.device, plate_lin.dtype)
    mat[:, 3] = pm - gm
    return mat


def _fit_affine_3x4(plate_lin: torch.Tensor, gen_lin: torch.Tensor, ring: torch.Tensor) -> tuple[torch.Tensor, str]:
    sel = ring > 0.5
    n = int(sel.sum().item())
    if n < 12:
        return _mean_offset_matrix(plate_lin, gen_lin, ring), "mean_ring_too_small"
    g = gen_lin[sel]
    p = plate_lin[sel]
    # Twelve coefficients do not need a million samples. An evenly spaced
    # subset of at most 200k ring pixels gives the same fit; the full set took
    # over a second a frame on a 2 MP crop.
    if g.shape[0] > _AFFINE_MAX_SAMPLES:
        idx = torch.linspace(0, g.shape[0] - 1, _AFFINE_MAX_SAMPLES, device=g.device).long()
        g, p = g[idx], p[idx]
        n = g.shape[0]
    ones = torch.ones(g.shape[0], 1, device=g.device, dtype=g.dtype)
    aug = torch.cat([g, ones], dim=1)
    try:
        coeffs = torch.linalg.lstsq(aug, p).solution
    except RuntimeError:
        return _mean_offset_matrix(plate_lin, gen_lin, ring), "lstsq_fail"
    pred = aug @ coeffs
    resid = (p - pred).norm(dim=1)
    k = max(1, int(math.floor(0.1 * n)))
    if k < n:
        keep = resid.argsort()[: n - k]
        aug = aug[keep]
        p = p[keep]
        try:
            coeffs = torch.linalg.lstsq(aug, p).solution
        except RuntimeError:
            return _mean_offset_matrix(plate_lin, gen_lin, ring), "lstsq_trim_fail"
    a3 = coeffs[:3, :].T
    cond = float(torch.linalg.cond(a3).item())
    if cond > _AFFINE_COND_MAX or not torch.isfinite(a3).all():
        return _mean_offset_matrix(plate_lin, gen_lin, ring), "ill_conditioned"
    mat = torch.zeros(3, 4, device=g.device, dtype=g.dtype)
    mat[:, :3] = coeffs[:3, :].T
    mat[:, 3] = coeffs[3, :]
    return mat, "affine"


def _apply_affine_3x4(gen_lin: torch.Tensor, mat: torch.Tensor) -> torch.Tensor:
    flat = gen_lin.reshape(-1, 3)
    return (flat @ mat[:, :3].T + mat[:, 3]).reshape(gen_lin.shape)


def _median_filter_coeffs(coeffs: list[torch.Tensor], i: int, window: int = _TEMPORAL_WINDOW) -> torch.Tensor:
    lo, hi = centred_window(i, len(coeffs), window)
    return torch.stack(coeffs[lo:hi], dim=0).median(dim=0).values


def _temporal_lag(ring_means: list[float], plate_means: list[float]) -> int:
    if len(ring_means) < 2:
        return 0
    diffs = [r - p for r, p in zip(ring_means, plate_means)]
    best_lag, best_var = 0, float("inf")
    for lag in range(-3, 4):
        aligned = []
        for t in range(len(diffs)):
            u = t + lag
            if 0 <= u < len(diffs):
                aligned.append(diffs[t] - diffs[u])
        if len(aligned) < 2:
            continue
        v = float(torch.tensor(aligned, dtype=torch.float32).var().item())
        if v < best_var:
            best_var = v
            best_lag = lag
    return best_lag


def _colour_magnitude(plate_lin: torch.Tensor, corrected_lin: torch.Tensor, ring: torch.Tensor) -> float:
    sel = ring > 0.5
    if not sel.any():
        return 0.0
    return float((plate_lin - corrected_lin)[sel].norm(dim=-1).mean().item())


def _build_ring(
    em: torch.Tensor,
    grow_px: int,
    interior: torch.Tensor,
) -> tuple[torch.Tensor, int, bool]:
    grown = _grow_mask(em, grow_px)
    ring = _erode_mask((1.0 - grown).clamp(0, 1), 2) * interior
    ring_count = int((ring > 0.5).sum().item())
    h, w = em.shape
    ring_ok = ring_count >= max(1, int(_RING_MIN_FRAC * h * w))
    return ring, ring_count, ring_ok


def restore_frames(
    cropped_images: torch.Tensor,
    original_images: torch.Tensor,
    boxes: list[dict[str, float | int]],
    edit_mask: torch.Tensor | None,
    grow_px: int,
    feather_px: int,
    register: str,
    max_shift_px: float,
    colour: str,
    temporal_lag_check: bool,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Per-frame plate restore in crop space then paste into full frames."""
    n = min(cropped_images.shape[0], original_images.shape[0], len(boxes))
    img_h, img_w = original_images.shape[1], original_images.shape[2]
    out_dev = original_images.device
    out_dtype = original_images.dtype
    out = original_images[:n].clone()
    full_restore_mask = torch.zeros(n, img_h, img_w, dtype=torch.float32, device=out_dev)
    streaming = device is not None and device != out_dev
    cd = device if streaming else out_dev

    no_edit_mask = edit_mask is None
    per_frame: list[dict[str, Any]] = []
    raw_coeffs: list[torch.Tensor] = []
    ring_means: list[float] = []
    plate_means: list[float] = []

    # Pass 1: measure shifts and colour coefficients
    frame_data: list[dict[str, Any]] = []
    for i in range(n):
        b = boxes[i]
        x, y, w, h = (int(round(b[k])) for k in ("x", "y", "width", "height"))
        plate_crop = out[i, y : y + h, x : x + w, :]
        gen_crop = _resize_hwc(cropped_images[i], h, w)
        if streaming:
            plate_crop = plate_crop.to(cd)
            gen_crop = gen_crop.to(cd)

        em = (
            torch.ones(h, w, device=cd, dtype=torch.float32)
            if no_edit_mask
            else _resize_mask_hw(edit_mask[min(i, edit_mask.shape[0] - 1)], h, w).clamp(0, 1).to(cd)
        )
        interior = interior_mask(h, w).to(cd)
        ring, ring_count, ring_ok = _build_ring(em, grow_px, interior)

        plate_lin = srgb_to_linear(plate_crop[..., :3])
        gen_lin = srgb_to_linear(gen_crop[..., :3])
        plate_luma = linear_luma(plate_lin)
        gen_luma = linear_luma(gen_lin)

        rec: dict[str, Any] = {
            "dx": 0.0,
            "dy": 0.0,
            "shift_applied": False,
            "shift_refused": False,
            "ring_pixels": ring_count,
            "colour_mode": "off",
            "colour_magnitude_linear": 0.0,
            "skipped_registration": not ring_ok,
            "skipped_colour": not ring_ok,
        }

        if ring_ok:
            rs = ring.sum().clamp_min(1.0)
            ring_means.append(float(((plate_luma - gen_luma) * ring).sum() / rs))
            plate_means.append(float((plate_luma * ring).sum() / rs))
        else:
            ring_means.append(0.0)
            plate_means.append(0.0)

        dx, dy = 0.0, 0.0
        if ring_ok and register != "off":
            dy, dx, peak = masked_ncc_shift(plate_luma, gen_luma, ring > 0.5, ring > 0.5)
            rec["dx"] = dx
            rec["dy"] = dy
            rec["ncc_peak"] = peak
            mag = math.hypot(dx, dy)
            if register == "correct":
                if mag > max_shift_px:
                    rec["shift_refused"] = True
                elif mag >= _SHIFT_DEADBAND_PX:
                    rec["shift_applied"] = True
                # Below the deadband the offset is inside the estimator's
                # own error (measured at <= 0.002 px on band-limited
                # texture), so warping would resample the crop through
                # bilinear for nothing - softening it slightly - and the
                # report would claim an offset was corrected.

        if ring_ok and colour == "affine":
            mat, mode = _fit_affine_3x4(plate_lin, gen_lin, ring)
            rec["colour_mode"] = mode
            raw_coeffs.append(mat)
        elif ring_ok and colour == "mean":
            raw_coeffs.append(_mean_offset_matrix(plate_lin, gen_lin, ring))
            rec["colour_mode"] = "mean"
        else:
            raw_coeffs.append(_identity_affine(cd, torch.float32))

        if streaming:
            raw_coeffs[-1] = raw_coeffs[-1].cpu()

        fd_entry: dict[str, Any] = {
            "x": x,
            "y": y,
            "w": w,
            "h": h,
            "ring_ok": ring_ok,
            "rec": rec,
            "dx": dx,
            "dy": dy,
        }
        if not streaming:
            fd_entry.update(em=em, ring=ring, plate_crop=plate_crop, gen_crop=gen_crop)
        frame_data.append(fd_entry)
        per_frame.append(rec)

    # Pass 2: apply corrections and composite
    for i, fd in enumerate(frame_data):
        x, y, w, h = fd["x"], fd["y"], fd["w"], fd["h"]
        rec = fd["rec"]
        if streaming:
            plate_crop = out[i, y : y + h, x : x + w, :].to(cd)
            working = _resize_hwc(cropped_images[i], h, w).to(cd)
            em = (
                torch.ones(h, w, device=cd, dtype=torch.float32)
                if no_edit_mask
                else _resize_mask_hw(edit_mask[min(i, edit_mask.shape[0] - 1)], h, w).clamp(0, 1).to(cd)
            )
            interior = interior_mask(h, w).to(cd)
            ring, _, _ = _build_ring(em, grow_px, interior)
        else:
            plate_crop = fd["plate_crop"]
            working = fd["gen_crop"].clone()
            ring = fd["ring"]
            em = fd["em"]

        if fd["ring_ok"] and rec.get("shift_applied"):
            working = inject_pixel_shift(working.unsqueeze(0), fd["dx"], fd["dy"]).squeeze(0).to(working.dtype)

        if fd["ring_ok"] and colour != "off":
            mat = _median_filter_coeffs(raw_coeffs, i)
            if streaming:
                mat = mat.to(cd)
            gen_lin = srgb_to_linear(working[..., :3])
            corrected_lin = _apply_affine_3x4(gen_lin, mat)
            rec["colour_magnitude_linear"] = _colour_magnitude(srgb_to_linear(plate_crop[..., :3]), corrected_lin, ring)
            corrected_rgb = linear_to_srgb(corrected_lin)
            if working.shape[-1] > 3:
                working = torch.cat([corrected_rgb.to(working.dtype), working[..., 3:]], dim=-1)
            else:
                working = corrected_rgb.to(working.dtype)

        border = border_ramp_alpha(h, w, x, y, img_h, img_w, feather_px, device=cd, dtype=torch.float32)
        edit_grown = _grow_mask(em, grow_px)
        alpha_crop = gaussian_blur_mask(edit_grown.unsqueeze(0).unsqueeze(0), feather_px)[0, 0]
        alpha_crop = (alpha_crop * border).clamp(0, 1)

        blended = mask_confined_blend(plate_crop, working, alpha_crop, colour_match=0.0)
        if streaming:
            out[i, y : y + h, x : x + w, :] = blended.to(out_dev, out_dtype)
            full_restore_mask[i, y : y + h, x : x + w] = alpha_crop.to(out_dev)
        else:
            out[i, y : y + h, x : x + w, :] = blended
            full_restore_mask[i, y : y + h, x : x + w] = alpha_crop

    best_lag = _temporal_lag(ring_means, plate_means) if temporal_lag_check else 0
    refused = sum(1 for f in per_frame if f.get("shift_refused"))
    applied = sum(1 for f in per_frame if f.get("shift_applied"))

    summary_parts = [
        f"frames={n}",
        f"register={register}",
        f"colour={colour}",
    ]
    if no_edit_mask:
        summary_parts.append(
            "WARNING: no edit_mask — full crop treated as regenerated (SubjectUncrop-equivalent alpha)."
        )
    if temporal_lag_check:
        summary_parts.append(f"temporal_lag_best={best_lag}" + (" WARN≠0" if best_lag != 0 else ""))
    summary_parts.append(f"shifts_applied={applied} refused={refused}")

    report: dict[str, Any] = {
        "frames": per_frame,
        "temporal_lag": best_lag,
        "no_edit_mask": no_edit_mask,
        "summary": " ".join(summary_parts),
    }
    return out, full_restore_mask, report
