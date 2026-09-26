# BSD-3-Clause — scikit-image
# PORTED FROM: scikit-image :: registration/_masked_phase_cross_correlation.py @ 0.26
#   (D. Padfield, "Masked Object Registration in the Fourier Domain", IEEE TIP 2012)
#
# Torch port for MiniMax H3 plate restore. No scikit-image import at runtime.
# Sub-pixel refinement: masked Lucas–Kanade on translations using drift_qc.inject_pixel_shift.

from __future__ import annotations

import torch
import torch.nn.functional as F

from .drift_qc import inject_pixel_shift


def _next_fast_len(n: int) -> int:
    """Smallest fast FFT size >= n (factors of 2, 3, 5 only)."""
    if n <= 1:
        return 1
    out = int(n)
    while True:
        x = out
        for p in (2, 3, 5):
            while x % p == 0:
                x //= p
        if x == 1:
            return out
        out += 1


def _flip2d(arr: torch.Tensor) -> torch.Tensor:
    return arr.flip(dims=(-2, -1))


def _cross_correlate_masked(
    arr1: torch.Tensor,
    arr2: torch.Tensor,
    m1: torch.Tensor,
    m2: torch.Tensor,
    *,
    overlap_ratio: float = 0.3,
) -> torch.Tensor:
    """2-D masked normalised cross-correlation (Padfield 2012), full mode."""
    fixed_image = arr1.float()
    moving_image = arr2.float()
    fixed_mask = m1.bool()
    moving_mask = m2.bool()
    eps = torch.finfo(fixed_image.dtype).eps

    h1, w1 = fixed_image.shape
    h2, w2 = moving_image.shape
    fh, fw = h1 + h2 - 1, w1 + w2 - 1
    fast_h, fast_w = _next_fast_len(fh), _next_fast_len(fw)

    fixed_image = fixed_image.clone()
    moving_image = moving_image.clone()
    fixed_image[~fixed_mask] = 0.0
    moving_image[~moving_mask] = 0.0

    rotated_moving_image = _flip2d(moving_image)
    rotated_moving_mask = _flip2d(moving_mask.float())

    def _rfft2(x: torch.Tensor) -> torch.Tensor:
        return torch.fft.rfft2(x, s=(fast_h, fast_w))

    def _irfft2(x: torch.Tensor, oh: int, ow: int) -> torch.Tensor:
        return torch.fft.irfft2(x, s=(fast_h, fast_w))[..., :oh, :ow]

    fixed_fft = _rfft2(fixed_image)
    rotated_moving_fft = _rfft2(rotated_moving_image)
    fixed_mask_fft = _rfft2(fixed_mask.float())
    rotated_moving_mask_fft = _rfft2(rotated_moving_mask)

    number_overlap = _irfft2(rotated_moving_mask_fft * fixed_mask_fft, fh, fw)
    number_overlap = torch.round(number_overlap).clamp_min(eps)

    masked_correlated_fixed = _irfft2(rotated_moving_mask_fft * fixed_fft, fh, fw)
    masked_correlated_moving = _irfft2(fixed_mask_fft * rotated_moving_fft, fh, fw)

    numerator = _irfft2(rotated_moving_fft * fixed_fft, fh, fw)
    numerator = numerator - (
        masked_correlated_fixed * masked_correlated_moving / number_overlap
    )

    fixed_squared_fft = _rfft2(fixed_image.square())
    fixed_denom = _irfft2(rotated_moving_mask_fft * fixed_squared_fft, fh, fw)
    fixed_denom = fixed_denom - (masked_correlated_fixed.square() / number_overlap)
    fixed_denom = fixed_denom.clamp_min(0.0)

    moving_squared_fft = _rfft2(rotated_moving_image.square())
    moving_denom = _irfft2(fixed_mask_fft * moving_squared_fft, fh, fw)
    moving_denom = moving_denom - (masked_correlated_moving.square() / number_overlap)
    moving_denom = moving_denom.clamp_min(0.0)

    denom = torch.sqrt(fixed_denom * moving_denom)
    tol = 1e3 * eps * denom.abs().amax()
    out = torch.zeros_like(denom)
    valid = denom > tol
    out[valid] = numerator[valid] / denom[valid]
    out = out.clamp(-1.0, 1.0)

    threshold = overlap_ratio * number_overlap.max()
    out = torch.where(number_overlap >= threshold, out, torch.zeros_like(out))
    return out


def _integer_shift(
    fixed: torch.Tensor,
    moving: torch.Tensor,
    fixed_mask: torch.Tensor,
    moving_mask: torch.Tensor,
    overlap_ratio: float,
) -> tuple[float, float]:
    """Integer (dy, dx) from masked NCC (skimage / Padfield convention)."""
    xcorr = _cross_correlate_masked(
        moving, fixed, moving_mask, fixed_mask, overlap_ratio=overlap_ratio
    )
    flat_idx = int(xcorr.argmax().item())
    iy, ix = flat_idx // xcorr.shape[1], flat_idx % xcorr.shape[1]
    maxima = (xcorr == xcorr[iy, ix]).nonzero(as_tuple=False).float().mean(dim=0)
    ref_shape = torch.tensor(fixed.shape, dtype=torch.float32)
    mov_shape = torch.tensor(moving.shape, dtype=torch.float32)
    shifts = maxima - ref_shape + 1.0
    size_mismatch = mov_shape - ref_shape
    out = -shifts + size_mismatch / 2.0
    return float(out[0].item()), float(out[1].item())


def _erode_mask_1px(mask: torch.Tensor) -> torch.Tensor:
    m = mask.float().unsqueeze(0).unsqueeze(0)
    return (-F.max_pool2d(-m, kernel_size=3, stride=1, padding=1))[0, 0] > 0.5


def _warp_scalar(img: torch.Tensor, dx: float, dy: float) -> torch.Tensor:
    """Warp 2-D scalar image with inject_pixel_shift (bilinear, border)."""
    hwc = img.unsqueeze(0).unsqueeze(-1).expand(1, img.shape[0], img.shape[1], 3)
    out = inject_pixel_shift(hwc, float(dx), float(dy)).squeeze(0)[..., 0]
    return out


def _grad_central(img: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    gx = torch.zeros_like(img)
    gy = torch.zeros_like(img)
    gx[:, 1:-1] = (img[:, 2:] - img[:, :-2]) * 0.5
    gy[1:-1, :] = (img[2:, :] - img[:-2, :]) * 0.5
    return gx, gy


def _masked_peak(fixed: torch.Tensor, warped: torch.Tensor, weight: torch.Tensor) -> float:
    w = weight.float()
    wf = fixed * w
    wm = warped * w
    num = (wf * wm).sum()
    den = torch.sqrt((wf * wf).sum() * (wm * wm).sum()).clamp_min(1e-12)
    return float((num / den).clamp(-1.0, 1.0).item())


def _lk_refine(
    fixed: torch.Tensor,
    moving: torch.Tensor,
    dx0: float,
    dy0: float,
    weight: torch.Tensor,
    *,
    max_iter: int = 30,
) -> tuple[float, float, float]:
    """Masked Gauss–Newton translation refine; W is inject_pixel_shift."""
    dx, dy = float(dx0), float(dy0)
    w = weight.float()
    peak = _masked_peak(fixed, _warp_scalar(moving, dx, dy), w)

    for _ in range(max_iter):
        warped = _warp_scalar(moving, dx, dy)
        res = (fixed - warped) * w
        gx, gy = _grad_central(warped)
        a11 = (w * gx * gx).sum()
        a12 = (w * gx * gy).sum()
        a22 = (w * gy * gy).sum()
        b1 = (w * gx * res).sum()
        b2 = (w * gy * res).sum()
        trace_sq = (a11 + a22).square()
        det = a11 * a22 - a12 * a12
        if float(trace_sq.item()) < 1e-18 or float(det.item()) < 1e-9 * float(trace_sq.item()):
            return dx, dy, peak * 0.5
        ddx = (a22 * b1 - a12 * b2) / det
        ddy = (-a12 * b1 + a11 * b2) / det
        if abs(float(ddx.item())) + abs(float(ddy.item())) < 1e-4:
            break
        dx += float(ddx.item())
        dy += float(ddy.item())

    warped = _warp_scalar(moving, dx, dy)
    peak = _masked_peak(fixed, warped, w)
    return dx, dy, peak


def masked_ncc_shift(
    fixed: torch.Tensor,
    moving: torch.Tensor,
    fixed_mask: torch.Tensor,
    moving_mask: torch.Tensor | None = None,
    *,
    overlap_ratio: float = 0.3,
) -> tuple[float, float, float]:
    """Masked normalised cross-correlation shift (Padfield 2012) + LK sub-pixel refine.

    Returns
    -------
    dy, dx, peak:
        Shift **(dy, dx)** in pixels for ``drift_qc.inject_pixel_shift(moving, dx, dy)``
        so ``moving`` aligns to ``fixed``.

        ``inject_pixel_shift(img, a, b)`` samples at (x+a, y+b), displacing content by
        (-a, -b). If ``moving = inject_pixel_shift(fixed, a, b)`` then the answer is
        ``dx = -a``, ``dy = -b``.
    """
    if fixed.ndim != 2 or moving.ndim != 2:
        raise ValueError("fixed and moving must be 2-D [H,W]")
    if fixed.shape != moving.shape:
        raise ValueError("fixed and moving must have the same shape")
    if fixed_mask.shape != fixed.shape:
        raise ValueError("fixed_mask must match fixed shape")
    if moving_mask is None:
        moving_mask = fixed_mask
    if moving_mask.shape != moving.shape:
        raise ValueError("moving_mask must match moving shape")

    # float32 throughout. A float64 image (numpy-born, or another node's output)
    # otherwise reaches grid_sample against a float32 affine grid and fails with
    # "expected scalar type Double but found Float" - inside a finishing node,
    # over a dtype.
    fixed = fixed.float()
    moving = moving.float()

    fmask = fixed_mask.bool()
    mmask = moving_mask.bool()

    idy, idx = _integer_shift(fixed, moving, fmask, mmask, overlap_ratio)

    # skimage integer shift is opposite to inject_pixel_shift correction.
    dx0, dy0 = -float(idx), -float(idy)

    weight = _erode_mask_1px(fmask) & _erode_mask_1px(mmask)
    if not weight.any():
        weight = fmask & mmask
    if not weight.any():
        return dy0, dx0, 0.0

    dx, dy, peak = _lk_refine(fixed, moving, dx0, dy0, weight)
    return dy, dx, peak
