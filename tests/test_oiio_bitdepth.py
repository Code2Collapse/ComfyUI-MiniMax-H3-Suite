import pytest

from mmx_utils.oiio_bitdepth import SUPPORTED_BIT_DEPTHS, UnsupportedBitDepthError, encode_pixels_for_bit_depth
import numpy as np


def test_encode_32f_preserves_hdr():
    px = np.array([[[0.0, 1.0, 16.0]]], dtype=np.float32)
    out, fmt = encode_pixels_for_bit_depth(px, "32f")
    assert fmt == "FLOAT"
    assert float(out[0, 0, 2]) == 16.0


def test_encode_16f_preserves_hdr():
    px = np.array([[[4.0, 8.0, 16.0]]], dtype=np.float32)
    out, fmt = encode_pixels_for_bit_depth(px, "16f")
    assert fmt == "HALF"
    assert float(out[0, 0, 2]) > 1.0


def test_encode_8_clamps_to_format_range():
    px = np.array([[[0.0, 1.0, 16.0]]], dtype=np.float32)
    out, fmt = encode_pixels_for_bit_depth(px, "8")
    assert fmt == "UINT8"
    assert int(out[0, 0, 2]) == 255


def test_unknown_bit_depth_raises_not_clamps():
    px = np.zeros((2, 2, 3), dtype=np.float32)
    with pytest.raises(UnsupportedBitDepthError, match="Unsupported bit_depth"):
        encode_pixels_for_bit_depth(px, "12")
    with pytest.raises(UnsupportedBitDepthError):
        encode_pixels_for_bit_depth(px, "bogus")


def test_supported_set_matches_nuke_port():
    assert SUPPORTED_BIT_DEPTHS == frozenset({"8", "16", "16f", "32f"})
