"""HDR round-trip helpers — tag propagation, no clamp audit."""

from __future__ import annotations

from typing import Literal

import torch

from .ocio_bridge import transform_image_batch

Direction = Literal["to_view", "to_scene"]
Tag = Literal["HDR", "Rec709", "ACEScg"]


def tag_out_for_direction(direction: Direction, colorspace: str, tag_in: str) -> str:
    d = (direction or "to_view").lower()
    cs = (colorspace or "ACEScg").strip()
    tin = (tag_in or "HDR").strip()
    if d == "to_view":
        return "Rec709"
    if d == "to_scene":
        if cs == "ACEScg" or tin == "ACEScg":
            return "ACEScg"
        return "HDR"
    return tin


def map_direction(direction: str) -> str:
    d = (direction or "to_view").lower()
    if d == "to_view":
        return "to_display"
    if d == "to_scene":
        return "to_scene_linear"
    raise ValueError(f"direction must be to_view or to_scene, got {direction!r}")


def hdr_peak(image: torch.Tensor) -> float:
    x = image.detach().float()
    if x.numel() == 0:
        return 0.0
    return float(x.max())


def assert_hdr_not_clamped(before: torch.Tensor, after: torch.Tensor, *, tag: str) -> str:
    """Return report fragment; never clamps — verifies peak preserved on to_scene / round-trip."""
    if (tag or "").upper() not in ("HDR", "ACESCG"):
        return f"tag={tag}: non-HDR tag — peak audit skipped."
    peak_before = hdr_peak(before)
    peak_after = hdr_peak(after)
    if peak_before > 1.0 + 1e-5 and peak_after < peak_before - 1e-4:
        return (
            f"WARNING: HDR peak dropped {peak_before:.4g} -> {peak_after:.4g} — "
            "check OCIO view is invertible and no torch.clamp was applied."
        )
    return (
        f"HDR audit OK — peak {peak_after:.4g} (input {peak_before:.4g}), "
        "no clamp above 1.0 on scene-linear path."
    )


def transform_hdr_batch(
    image: torch.Tensor,
    *,
    direction: Direction,
    colorspace: str,
    display: str,
    view: str,
    chunk_size: int = 4,
    config=None,
) -> torch.Tensor:
    bridge_dir = map_direction(direction)
    return transform_image_batch(
        image,
        direction=bridge_dir,  # type: ignore[arg-type]
        source_colorspace=colorspace,
        target_colorspace=colorspace,
        display=display,
        view=view,
        chunk_size=chunk_size,
        config=config,
    )
