# Colour-space utilities for plate fidelity (linear sRGB + Oklab).
#
# Oklab matrices and pipeline from Björn Ottosson:
#   https://bottosson.github.io/posts/oklab/
#   (public domain / MIT — no licence restriction on the published reference code)
#
# sRGB EOTF: IEC 61966-2-1 piecewise functions.

from __future__ import annotations

import torch

# Linear sRGB ↔ LMS (Oklab step 1)
_M1 = torch.tensor(
    [
        [0.4122214708, 0.5363325363, 0.0514459929],
        [0.2119034982, 0.6806995451, 0.1073969566],
        [0.0883024619, 0.2817188376, 0.6299787005],
    ],
    dtype=torch.float32,
)
_M2 = torch.tensor(
    [
        [0.2104542553, 0.7936177850, -0.0040720468],
        [1.9779984951, -2.4285922050, 0.4505937099],
        [0.0259040371, 0.7827717662, -0.8086757660],
    ],
    dtype=torch.float32,
)
_M1_INV = torch.tensor(
    [
        [4.0767416621, -3.3077115913, 0.2309699292],
        [-1.2684380046, 2.6097574011, -0.3413193965],
        [-0.0041960863, -0.7034186147, 1.7076147010],
    ],
    dtype=torch.float32,
)
_M2_INV = torch.tensor(
    [
        [0.9999999985, 0.3963377922, 0.2158037573],
        [1.0000000089, -0.1055613423, -0.0638541758],
        [1.0000000547, -0.0894841821, -1.2914855480],
    ],
    dtype=torch.float32,
)

_SRGB_BREAK = 0.04045
_SRGB_LINEAR_BREAK = 0.0031308


def _device_dtype(t: torch.Tensor) -> tuple[torch.device, torch.dtype]:
    return t.device, t.dtype


def _matrix_on_device(m: torch.Tensor, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return m.to(device=device, dtype=dtype)


def _sign_cbrt(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.abs(x).pow(1.0 / 3.0)


def srgb_to_linear(rgb: torch.Tensor) -> torch.Tensor:
    """sRGB-encoded [...,3] → linear light. No clamp; values above 1.0 survive."""
    x = rgb.float()
    return torch.where(
        x <= _SRGB_BREAK,
        x / 12.92,
        ((x + 0.055) / 1.055).pow(2.4),
    )


def linear_to_srgb(linear: torch.Tensor) -> torch.Tensor:
    """Linear light [...,3] → sRGB-encoded.

    Values above 1.0 use the standard high segment (no clamp).
    Negative linear values use a sign-preserving extension of the EOTF so HDR
    round-trips stay invertible without silently discarding sub-black data.
    """
    x = linear.float()
    sign = torch.sign(x)
    ax = torch.abs(x)
    pos = torch.where(
        ax <= _SRGB_LINEAR_BREAK,
        ax * 12.92,
        1.055 * ax.pow(1.0 / 2.4) - 0.055,
    )
    return sign * pos


def linear_srgb_to_oklab(linear: torch.Tensor) -> torch.Tensor:
    """Linear sRGB [...,3] → Oklab [...,3] (L, a, b)."""
    dev, dt = _device_dtype(linear)
    m1 = _matrix_on_device(_M1, dev, dt)
    m2 = _matrix_on_device(_M2, dev, dt)
    lms = torch.matmul(linear.float(), m1.T)
    lms_c = _sign_cbrt(lms)
    return torch.matmul(lms_c, m2.T)


def oklab_to_linear_srgb(oklab: torch.Tensor) -> torch.Tensor:
    """Oklab [...,3] → linear sRGB [...,3]."""
    dev, dt = _device_dtype(oklab)
    m2_inv = _matrix_on_device(_M2_INV, dev, dt)
    m1_inv = _matrix_on_device(_M1_INV, dev, dt)
    lms_c = torch.matmul(oklab.float(), m2_inv.T)
    lms = lms_c ** 3
    return torch.matmul(lms, m1_inv.T)


def linear_luma(linear_rgb: torch.Tensor) -> torch.Tensor:
    """Rec.709 luma from linear RGB [...,3] → [...]."""
    r, g, b = linear_rgb[..., 0], linear_rgb[..., 1], linear_rgb[..., 2]
    return 0.2126 * r + 0.7152 * g + 0.0722 * b
