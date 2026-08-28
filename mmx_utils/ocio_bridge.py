# MIT License — nuke-nodes-comfyui
# Copyright (c) 2025 Sumit Chatterjee
# PORTED FROM: nuke-nodes-comfyui :: colorspace_nodes.py + io_nodes.py (bit-depth guard) @ HEAD

"""OCIO color transforms — HDR-safe, no silent clamp, no silent fallback on errors."""

from __future__ import annotations

from typing import Literal

import numpy as np
import torch

from .ocio_config import (
    OCIO_AVAILABLE,
    OCIOColorSpaceError,
    OCIOConfigUnavailableError,
    OCIOUnavailableError,
    get_ocio_config,
    validate_colorspace,
)

Direction = Literal["to_display", "to_scene_linear"]

# OCIO PackedImageDesc applies per 2-D plane; batch dimension is chunked in Python.
OCIO_FORCES_PER_FRAME_PLANES = True


class OCIOTransformError(RuntimeError):
    """OCIO processor failed to apply a transform."""


def _require_ocio_module():
    if not OCIO_AVAILABLE:
        raise OCIOUnavailableError("PyOpenColorIO is not installed.")
    import PyOpenColorIO as OCIO

    return OCIO


def _split_rgb_alpha(img: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
    if img.ndim != 3:
        raise ValueError(f"OCIO bridge expects HxWxC, got shape {img.shape}")
    channels = img.shape[2]
    if channels == 4:
        return img[:, :, :3].astype(np.float32, copy=True), img[:, :, 3:4].copy()
    if channels >= 3:
        return img[:, :, :3].astype(np.float32, copy=True), None
    rgb = np.stack([img[:, :, 0].astype(np.float32)] * 3, axis=-1)
    return rgb, None


def _merge_rgb_alpha(rgb: np.ndarray, alpha: np.ndarray | None) -> np.ndarray:
    if alpha is not None:
        return np.concatenate([rgb, alpha], axis=-1)
    return rgb


def _apply_cpu_processor(rgb: np.ndarray, cpu_processor) -> np.ndarray:
    """Vectorize across all pixels in one HxW plane (no per-pixel Python loop)."""
    OCIO = _require_ocio_module()
    height, width = rgb.shape[:2]
    rgb_flat = np.ascontiguousarray(rgb.reshape(-1, 3), dtype=np.float32)
    img_desc = OCIO.PackedImageDesc(
        rgb_flat,
        width,
        height,
        3,
        OCIO.BIT_DEPTH_F32,
        rgb_flat.strides[1],
        rgb_flat.strides[0],
        width * rgb_flat.strides[0],
    )
    cpu_processor.apply(img_desc)
    return rgb_flat.reshape(height, width, 3)


def apply_colorspace_transform(
    image: np.ndarray,
    src_colorspace: str,
    dst_colorspace: str,
    *,
    config=None,
) -> np.ndarray:
    if src_colorspace == dst_colorspace:
        return image
    OCIO = _require_ocio_module()
    cfg = config or get_ocio_config()
    if cfg is None:
        raise OCIOConfigUnavailableError("No OCIO config available for colorspace transform.")
    validate_colorspace(src_colorspace, config=cfg)
    validate_colorspace(dst_colorspace, config=cfg)
    rgb, alpha = _split_rgb_alpha(np.asarray(image, dtype=np.float32))
    try:
        processor = cfg.getProcessor(src_colorspace, dst_colorspace)
        cpu_processor = processor.getDefaultCPUProcessor()
        transformed = _apply_cpu_processor(rgb, cpu_processor)
    except OCIOColorSpaceError:
        raise
    except Exception as exc:
        raise OCIOTransformError(
            f"OCIO colorspace transform {src_colorspace!r} -> {dst_colorspace!r} failed: {exc}"
        ) from exc
    return _merge_rgb_alpha(transformed, alpha)


def apply_display_transform(
    image: np.ndarray,
    scene_colorspace: str,
    display: str,
    view: str,
    *,
    to_scene_linear: bool = False,
    config=None,
) -> np.ndarray:
    OCIO = _require_ocio_module()
    cfg = config or get_ocio_config()
    if cfg is None:
        raise OCIOConfigUnavailableError("No OCIO config available for display transform.")
    validate_colorspace(scene_colorspace, config=cfg)

    rgb, alpha = _split_rgb_alpha(np.asarray(image, dtype=np.float32))
    try:
        transform = OCIO.DisplayViewTransform()
        transform.setSrc(scene_colorspace)
        transform.setDisplay(display)
        transform.setView(view)
        direction = (
            OCIO.TRANSFORM_DIR_INVERSE if to_scene_linear else OCIO.TRANSFORM_DIR_FORWARD
        )
        processor = cfg.getProcessor(transform, direction)
        cpu_processor = processor.getDefaultCPUProcessor()
        transformed = _apply_cpu_processor(rgb, cpu_processor)
    except Exception as exc:
        mode = "to_scene_linear" if to_scene_linear else "to_display"
        raise OCIOTransformError(
            f"OCIO display transform ({mode}) scene={scene_colorspace!r} "
            f"display={display!r} view={view!r} failed: {exc}"
        ) from exc
    return _merge_rgb_alpha(transformed, alpha)


def apply_bridge_plane(
    image: np.ndarray,
    *,
    direction: Direction,
    source_colorspace: str,
    target_colorspace: str,
    display: str,
    view: str,
    config=None,
) -> np.ndarray:
    """One HxWxC plane — scene-linear ↔ display without clamping."""
    if direction == "to_display":
        work = image
        if source_colorspace != target_colorspace:
            work = apply_colorspace_transform(
                work, source_colorspace, target_colorspace, config=config
            )
        return apply_display_transform(
            work,
            target_colorspace,
            display,
            view,
            to_scene_linear=False,
            config=config,
        )

    if direction == "to_scene_linear":
        work = apply_display_transform(
            image,
            target_colorspace,
            display,
            view,
            to_scene_linear=True,
            config=config,
        )
        if source_colorspace != target_colorspace:
            work = apply_colorspace_transform(
                work, target_colorspace, source_colorspace, config=config
            )
        return work

    raise ValueError(f"Unknown OCIO bridge direction {direction!r}")


def transform_image_batch(
    image: torch.Tensor,
    *,
    direction: Direction,
    source_colorspace: str,
    target_colorspace: str,
    display: str,
    view: str,
    chunk_size: int = 4,
    config=None,
) -> torch.Tensor:
    """Apply bridge to [B,H,W,C]. Chunks batch; one OCIO plane per frame (API limit)."""
    if image.ndim == 3:
        image = image.unsqueeze(0)
    if image.ndim != 4:
        raise ValueError(f"Expected IMAGE [B,H,W,C], got {tuple(image.shape)}")

    b = int(image.shape[0])
    outputs: list[np.ndarray] = []
    for start in range(0, b, chunk_size):
        end = min(start + chunk_size, b)
        for i in range(start, end):
            plane = image[i].detach().cpu().numpy()
            outputs.append(
                apply_bridge_plane(
                    plane,
                    direction=direction,
                    source_colorspace=source_colorspace,
                    target_colorspace=target_colorspace,
                    display=display,
                    view=view,
                    config=config,
                )
            )
    stacked = np.stack(outputs, axis=0)
    return torch.from_numpy(stacked).to(device=image.device, dtype=image.dtype)


def roundtrip_plane(
    image: np.ndarray,
    *,
    scene_colorspace: str,
    display: str,
    view: str,
    config=None,
) -> np.ndarray:
    """scene-linear -> display -> scene-linear on one plane."""
    displayed = apply_display_transform(
        image,
        scene_colorspace,
        display,
        view,
        to_scene_linear=False,
        config=config,
    )
    return apply_display_transform(
        displayed,
        scene_colorspace,
        display,
        view,
        to_scene_linear=True,
        config=config,
    )
