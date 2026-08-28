import numpy as np
import pytest
import torch

pytest.importorskip("PyOpenColorIO")

from mmx_utils.color_roundtrip_qc import (
    HDR_PROBE_VALUES,
    build_hdr_probe_image,
    clamp_for_vacuous_test,
    evaluate_roundtrip,
    run_display_roundtrip_qc,
)
from mmx_utils.ocio_bridge import apply_bridge_plane, roundtrip_plane
from mmx_utils.ocio_config import get_ocio_config


SCENE = "ACEScg"
DISPLAY = "sRGB - Display"
VIEW = "ACES 2.0 - SDR 100 nits (Rec.709)"


@pytest.fixture
def ocio_config():
    cfg = get_ocio_config()
    if cfg is None:
        pytest.skip("No OCIO builtin config available")
    return cfg


def test_hdr_probe_values_present():
    probe = build_hdr_probe_image()
    row = probe[0, 0, :, 0].tolist()
    assert row == list(HDR_PROBE_VALUES)


def test_roundtrip_preserves_hdr_highlights(ocio_config):
    probe = build_hdr_probe_image()
    plane = probe[0].numpy()
    recovered = roundtrip_plane(plane, scene_colorspace=SCENE, display=DISPLAY, view=VIEW, config=ocio_config)
    mask = plane > 1.0
    assert bool(mask.any())
    assert np.all(recovered[mask] > 1.0)


def test_bridge_to_display_and_back_roundtrip(ocio_config):
    probe = build_hdr_probe_image()
    plane = probe[0].numpy()
    displayed = apply_bridge_plane(
        plane,
        direction="to_display",
        source_colorspace=SCENE,
        target_colorspace=SCENE,
        display=DISPLAY,
        view=VIEW,
        config=ocio_config,
    )
    back = apply_bridge_plane(
        displayed,
        direction="to_scene_linear",
        source_colorspace=SCENE,
        target_colorspace=SCENE,
        display=DISPLAY,
        view=VIEW,
        config=ocio_config,
    )
    assert np.all(back[plane > 1.0] > 1.0)


def test_qc_fails_on_deliberately_clamped_input():
    probe = build_hdr_probe_image()[0].numpy()
    clamped = clamp_for_vacuous_test(probe)
    result = run_display_roundtrip_qc(
        torch.from_numpy(clamped).unsqueeze(0),
        scene_colorspace=SCENE,
        display=DISPLAY,
        view=VIEW,
        tolerance=1e-4,
    )
    assert not result.passed
    assert "clamped" in result.report.lower() or "HDR" in result.report


def test_run_display_roundtrip_qc_on_probe(ocio_config):
    probe = build_hdr_probe_image()
    result = run_display_roundtrip_qc(
        probe,
        scene_colorspace=SCENE,
        display=DISPLAY,
        view=VIEW,
        tolerance=1e-2,
        config=ocio_config,
    )
    assert result.hdr_preserved


def test_color_roundtrip_qc_node_passes_hdr_probe(ocio_config):
    from mmx_nodes.color_roundtrip_qc import MiniMaxH3_ColorRoundTripQC

    probe = build_hdr_probe_image()
    passed, max_err, report, err_map = MiniMaxH3_ColorRoundTripQC.execute(
        probe,
        SCENE,
        DISPLAY,
        VIEW,
        1e-2,
    )
    assert isinstance(passed, bool)
    assert "OCIO" in report
    assert err_map.shape[0] == 1


def test_color_roundtrip_qc_node_fails_on_clamped():
    from mmx_nodes.color_roundtrip_qc import MiniMaxH3_ColorRoundTripQC

    probe = build_hdr_probe_image()
    clamped = torch.from_numpy(clamp_for_vacuous_test(probe[0].numpy())).unsqueeze(0)
    passed, _max_err, report, _ = MiniMaxH3_ColorRoundTripQC.execute(
        clamped,
        SCENE,
        DISPLAY,
        VIEW,
        1e-4,
    )
    assert passed is False
    assert "FAILED" in report
    assert "clamped" in report.lower()
