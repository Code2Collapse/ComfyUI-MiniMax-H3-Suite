import torch
import pytest

pytest.importorskip("PyOpenColorIO")

from mmx_nodes.ocio_bridge import MiniMaxH3_OCIOBridge
from mmx_utils.color_roundtrip_qc import build_hdr_probe_image
from mmx_utils.ocio_config import get_ocio_config


@pytest.fixture
def ocio_ready():
    if get_ocio_config() is None:
        pytest.skip("No OCIO builtin config")


def test_ocio_bridge_node_roundtrip_hdr(ocio_ready):
    probe = build_hdr_probe_image()
    displayed = MiniMaxH3_OCIOBridge.execute(
        probe,
        "to_display",
        "ACEScg",
        "ACEScg",
        "sRGB - Display",
        "ACES 2.0 - SDR 100 nits (Rec.709)",
    )
    out_img, report = displayed[0], displayed[1]
    assert "OCIO" in report
    back = MiniMaxH3_OCIOBridge.execute(
        out_img,
        "to_scene_linear",
        "ACEScg",
        "ACEScg",
        "sRGB - Display",
        "ACES 2.0 - SDR 100 nits (Rec.709)",
    )
    recovered = back[0]
    orig = probe[0, 0, :, 0]
    rec = recovered[0, 0, :, 0]
    assert torch.all(rec[orig > 1] > 1.0)
