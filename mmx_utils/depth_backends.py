# Apache-2.0 — ComfyUI-CustomNodePacks
# PORTED FROM: ComfyUI-CustomNodePacks :: nodes/_control_backends.py @ HEAD

from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.nn.functional as F


class DepthBackendUnavailableError(RuntimeError):
    """Named error when a weight-gated depth backend cannot run."""


def _to_bhwc(image: torch.Tensor) -> torch.Tensor:
    if image.ndim == 3:
        image = image.unsqueeze(0)
    return image


def passthrough_depth_plate(image: torch.Tensor, passthrough_hints: torch.Tensor | None) -> torch.Tensor:
    """Offline-safe depth: external hints or luminance proxy in plate space."""
    if passthrough_hints is not None:
        hints = passthrough_hints
        if hints.ndim == 3:
            hints = hints.unsqueeze(0)
        return hints[..., :3] if hints.shape[-1] >= 3 else hints.repeat(1, 1, 1, 3)
    img = _to_bhwc(image)
    gray = img[..., :3].mean(dim=-1, keepdim=True)
    return gray.repeat(1, 1, 1, 3)


def run_depthcrafter(
    image_bhwc: torch.Tensor,
    *,
    num_inference_steps: int = 5,
    guidance_scale: float = 1.0,
    unet_path: str = "tencent/DepthCrafter",
    base: str = "stabilityai/stable-video-diffusion-img2vid-xt",
) -> torch.Tensor:
    raise DepthBackendUnavailableError(
        f"DepthCrafter backend unavailable — UNet weights at {unet_path!r} are not downloaded. "
        f"SVD base {base!r} may be present locally but DepthCrafter UNet is required. "
        "Use backend=passthrough for offline work."
    )


def run_dvd(
    image_bhwc: torch.Tensor,
    *,
    window_size: int = 81,
) -> torch.Tensor:
    _ = (image_bhwc, window_size)
    third_party = Path(__file__).resolve().parents[2] / "third_party" / "DVD"
    if not third_party.is_dir():
        raise DepthBackendUnavailableError(
            "DVD backend unavailable — third_party/DVD is not cloned. "
            "Clone EnVision-Research/DVD into third_party/DVD or use backend=passthrough."
        )
    raise DepthBackendUnavailableError(
        "DVD backend unavailable — third_party/DVD is not cloned or checkpoint is missing."
    )


def normals_from_depth(depth_bhwc: torch.Tensor) -> torch.Tensor:
    """Sobel-from-depth Lambertian normals — no model, always works."""
    d = _to_bhwc(depth_bhwc)
    g = d[..., 0].unsqueeze(1)
    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
    ky = kx.transpose(2, 3).contiguous()
    gx = F.conv2d(F.pad(g, (1, 1, 1, 1), mode="replicate"), kx)
    gy = F.conv2d(F.pad(g, (1, 1, 1, 1), mode="replicate"), ky)
    nz = torch.ones_like(gx)
    n = torch.cat([-gx, -gy, nz], dim=1)
    n = n / (n.norm(dim=1, keepdim=True) + 1e-6)
    n = (n * 0.5 + 0.5).permute(0, 2, 3, 1)
    return n.clamp(0, 1).contiguous()


def run_depth_backend(
    backend: str,
    image: torch.Tensor,
    *,
    passthrough_hints: torch.Tensor | None = None,
) -> tuple[torch.Tensor, str]:
    b = (backend or "passthrough").lower()
    if b == "passthrough":
        out = passthrough_depth_plate(image, passthrough_hints)
        note = "passthrough: warped hints" if passthrough_hints is not None else "passthrough: luminance proxy"
        return out, note
    if b == "depthcrafter":
        return run_depthcrafter(image), "depthcrafter"
    if b == "dvd":
        return run_dvd(image), "dvd"
    raise ValueError(f"Unknown depth backend {backend!r}")
