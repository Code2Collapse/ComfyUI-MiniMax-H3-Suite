# MIT License — nuke-nodes-comfyui
# Copyright (c) 2025 Sumit Chatterjee
# PORTED FROM: nuke-nodes-comfyui :: io_nodes.py write_image_oiio @ HEAD

"""OIIO bit-depth encoding — strict validation, no silent uint16 fallback."""

from __future__ import annotations

import numpy as np

SUPPORTED_BIT_DEPTHS: frozenset[str] = frozenset({"8", "16", "16f", "32f"})


class UnsupportedBitDepthError(ValueError):
    """bit_depth string is not one of the supported encodings."""


def encode_pixels_for_bit_depth(pixels: np.ndarray, bit_depth: str) -> tuple[np.ndarray, str]:
    """
    Prepare pixels for OIIO write.

    Returns (pixels_out, oiio_format_name) where format_name is a symbolic label
    for tests (UINT8 / UINT16 / HALF / FLOAT).
    """
    depth = str(bit_depth).strip()
    if depth not in SUPPORTED_BIT_DEPTHS:
        raise UnsupportedBitDepthError(
            f"Unsupported bit_depth {bit_depth!r}. "
            f"Expected one of {sorted(SUPPORTED_BIT_DEPTHS)} — "
            "refusing silent clamp to uint16."
        )

    px = np.asarray(pixels, dtype=np.float32)
    if depth == "8":
        return (np.clip(px, 0, 1) * 255).astype(np.uint8), "UINT8"
    if depth == "16":
        return (np.clip(px, 0, 1) * 65535).astype(np.uint16), "UINT16"
    if depth == "16f":
        return px.astype(np.float16), "HALF"
    return px.astype(np.float32), "FLOAT"
