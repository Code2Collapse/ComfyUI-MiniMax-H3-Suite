import pytest
import torch

pytest.importorskip("PyOpenColorIO")

from mmx_utils.color_roundtrip_qc import build_hdr_probe_image
from mmx_utils.hdr_roundtrip import assert_hdr_not_clamped, tag_out_for_direction, transform_hdr_batch
from mmx_utils.ocio_config import get_ocio_config


@pytest.fixture
def ocio_ready():
    if get_ocio_config() is None:
        pytest.skip("No OCIO builtin config")


def test_tag_propagation():
    assert tag_out_for_direction("to_view", "ACEScg", "HDR") == "Rec709"
    assert tag_out_for_direction("to_scene", "ACEScg", "Rec709") == "ACEScg"


def test_hdr_peak_preserved_roundtrip(ocio_ready):
    probe = build_hdr_probe_image(width=5, height=1)
    displayed = transform_hdr_batch(
        probe,
        direction="to_view",
        colorspace="ACEScg",
        display="sRGB - Display",
        view="ACES 2.0 - SDR 100 nits (Rec.709)",
    )
    view_audit = assert_hdr_not_clamped(probe, displayed, tag="HDR")
    assert "skipped" in view_audit.lower() or "non-HDR" in view_audit.lower() or "peak" in view_audit.lower()
    back = transform_hdr_batch(
        displayed,
        direction="to_scene",
        colorspace="ACEScg",
        display="sRGB - Display",
        view="ACES 2.0 - SDR 100 nits (Rec.709)",
    )
    orig = probe[0, 0, :, 0]
    rec = back[0, 0, :, 0]
    assert torch.all(rec[orig > 1] > 1.0)
    audit = assert_hdr_not_clamped(probe, back, tag="HDR")
    assert "HDR audit OK" in audit or "peak" in audit.lower()
