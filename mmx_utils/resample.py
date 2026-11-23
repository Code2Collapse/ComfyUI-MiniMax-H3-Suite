"""Float resampling that keeps the plate's values (zero-offset work, docs/PLAN_h3_zero_offset.md).

Original implementation, GPL-3.0. Why it exists: ``comfy.utils.common_upscale(..., "lanczos", ...)`` goes through
8-bit PIL images (comfy/utils.py ``lanczos``: ``np.clip(255 * x, 0, 255).astype(np.uint8)``), so every Lanczos resize
on the V2V path clipped scene-linear / HDR values above 1.0 and quantised a 10/16-bit plate to 256 levels. Measured on
an HDR x1.5 plate through Subject Crop (zoomed) -> Subject Uncrop with an identity edit: 129/255 interior error.

``lanczos_resize`` reproduces PIL's ``Image.resize(..., LANCZOS)`` sampling geometry exactly (libImaging/Resample.c
``precompute_coeffs``: half-pixel centres, support 3 x max(scale, 1), weights normalised per output pixel), in float
with no clamp, so results agree with the old path to 8-bit rounding on SDR input and keep everything else.
"""

from __future__ import annotations

import math

import torch

_LANCZOS_A = 3.0


def _lanczos(x: torch.Tensor) -> torch.Tensor:
    ax = x.abs()
    out = torch.where(ax < 1e-12, torch.ones_like(x), torch.zeros_like(x))
    nz = (ax >= 1e-12) & (ax < _LANCZOS_A)
    px = math.pi * x[nz]
    out[nz] = (torch.sin(px) / px) * (torch.sin(px / _LANCZOS_A) / (px / _LANCZOS_A))
    return out


def lanczos_weights(in_size: int, out_size: int, device=None) -> torch.Tensor:
    """[out_size, in_size] float64 weight matrix with PIL's LANCZOS coefficients for one axis."""
    in_size, out_size = int(in_size), int(out_size)
    scale = in_size / out_size
    filterscale = max(scale, 1.0)
    support = _LANCZOS_A * filterscale
    ss = 1.0 / filterscale
    w = torch.zeros((out_size, in_size), dtype=torch.float64)
    for xx in range(out_size):
        center = (xx + 0.5) * scale
        xmin = max(int(center - support + 0.5), 0)
        xmax = min(int(center + support + 0.5), in_size)
        if xmax <= xmin:
            xmin, xmax = min(max(int(center), 0), in_size - 1), min(max(int(center), 0), in_size - 1) + 1
        taps = torch.arange(xmin, xmax, dtype=torch.float64)
        k = _lanczos((taps - center + 0.5) * ss)
        s = float(k.sum())
        w[xx, xmin:xmax] = k / s if abs(s) > 1e-12 else 1.0 / float(xmax - xmin)
    return w.to(device) if device is not None else w


def lanczos_resize(x: torch.Tensor, width: int, height: int) -> torch.Tensor:
    """Separable Lanczos-3 resize of ``x`` [N, C, H, W] (or [N, H, W]) to ``height`` x ``width``, in float.

    No clamp and no quantisation: values above 1.0 or below 0.0 survive (scene-linear / log / HDR plates). Same size in
    and out returns ``x`` unchanged.
    """
    squeeze = x.ndim == 3
    if squeeze:
        x = x.unsqueeze(1)
    if x.ndim != 4:
        raise ValueError(f"lanczos_resize expects [N, C, H, W]; got {tuple(x.shape)}")
    h, w = int(x.shape[-2]), int(x.shape[-1])
    width, height = int(width), int(height)
    if (h, w) == (height, width):
        return x[:, 0] if squeeze else x
    dt = x.dtype
    y = x.float()
    if w != width:
        wx = lanczos_weights(w, width).to(device=y.device, dtype=torch.float32)
        y = torch.einsum("nchw,ow->ncho", y, wx)
    if h != height:
        wy = lanczos_weights(h, height).to(device=y.device, dtype=torch.float32)
        y = torch.einsum("nchw,oh->ncow", y, wy)
    y = y.to(dt) if dt.is_floating_point else y
    return y[:, 0] if squeeze else y


def common_upscale_float(samples: torch.Tensor, width: int, height: int, upscale_method: str, crop: str):
    """Drop-in for ``comfy.utils.common_upscale`` (same signature, same 'center' crop arithmetic, same >4-D reshape)
    whose ``lanczos`` is :func:`lanczos_resize` (float, unclipped). Every other method is core's own."""
    if upscale_method != "lanczos":
        import comfy.utils  # noqa: WPS433 - core is imported lazily, never at module level

        return comfy.utils.common_upscale(samples, width, height, upscale_method, crop)
    orig_shape = tuple(samples.shape)
    if len(orig_shape) > 4:
        samples = samples.reshape(samples.shape[0], samples.shape[1], -1, samples.shape[-2], samples.shape[-1])
        samples = samples.movedim(2, 1)
        samples = samples.reshape(-1, orig_shape[1], orig_shape[-2], orig_shape[-1])
    if crop == "center":
        old_width = samples.shape[-1]
        old_height = samples.shape[-2]
        old_aspect = old_width / old_height
        new_aspect = width / height
        x = 0
        y = 0
        if old_aspect > new_aspect:
            x = round((old_width - old_width * (new_aspect / old_aspect)) / 2)
        elif old_aspect < new_aspect:
            y = round((old_height - old_height * (old_aspect / new_aspect)) / 2)
        s = samples.narrow(-2, y, old_height - y * 2).narrow(-1, x, old_width - x * 2)
    else:
        s = samples
    out = lanczos_resize(s, width, height)
    if len(orig_shape) == 4:
        return out
    out = out.reshape((orig_shape[0], -1, orig_shape[1]) + (height, width))
    return out.movedim(2, 1).reshape(orig_shape[:-2] + (height, width))


__all__ = ["common_upscale_float", "lanczos_resize", "lanczos_weights"]
