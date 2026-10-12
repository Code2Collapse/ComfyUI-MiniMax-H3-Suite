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


# L7.93 (owner's Linux log 2026-10-12): the node computes on the GPU but must hand IMAGE / MASK back on ComfyUI's
# intermediate device (the CPU), or core Batch Images fails with "Expected all tensors to be on the same device".
_needs_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU to compute on")


def _on_gpu(monkeypatch, module):
    """The node's own device choice, pinned to the GPU; ComfyUI's intermediate device stays the CPU."""
    import types

    stub = types.SimpleNamespace(get_torch_device=lambda: torch.device("cuda"),
                                 intermediate_device=lambda: torch.device("cpu"))
    monkeypatch.setattr(module, "mm", stub, raising=False)
    import mmx_utils.device as d

    monkeypatch.setattr(d, "mm", stub, raising=False)


@_needs_gpu
def test_ocio_bridge_hands_back_on_the_intermediate_device(ocio_ready, monkeypatch):
    import mmx_nodes.ocio_bridge as mod

    _on_gpu(monkeypatch, mod)
    out = MiniMaxH3_OCIOBridge.execute(build_hdr_probe_image(), "to_display", "ACEScg", "ACEScg", "sRGB - Display",
                                       "ACES 2.0 - SDR 100 nits (Rec.709)")
    assert out[0].device.type == "cpu"
