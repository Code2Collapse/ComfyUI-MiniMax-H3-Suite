"""UI payload shape tests for MiniMax H3 plot widgets."""

from __future__ import annotations

import json

import torch

from mmx_nodes.drift_qc import MiniMaxH3_DriftQC
from mmx_utils.drift_qc import inject_pixel_shift
from mmx_utils.sigma_schedule import (
    TRAINED_SHIFT_AUDIO,
    TRAINED_SHIFT_VIDEO,
    build_sigma_inspector_json,
    synthesize_linear_sigmas,
)


def test_sigma_inspector_json_includes_trained_shifts():
    payload = json.loads(build_sigma_inspector_json(synthesize_linear_sigmas(5)))
    assert payload["trained_shift_video"] == TRAINED_SHIFT_VIDEO
    assert payload["trained_shift_audio"] == TRAINED_SHIFT_AUDIO


def test_drift_ui_json_includes_peak_and_threshold():
    torch.manual_seed(0)
    original = torch.rand(1, 48, 48, 3)
    edited = inject_pixel_shift(original, 0.0, 0.0)
    mask = torch.zeros(48, 48)
    mask[12:36, 12:36] = 1.0
    out = MiniMaxH3_DriftQC.execute(original, edited, mask, 2.0)
    assert out.ui is not None
    raw = out.ui["mmx_drift"][0]
    payload = json.loads(raw)
    assert "threshold_px" in payload
    assert "peak_px" in payload
    assert "peak_frame" in payload
    assert payload["threshold_px"] == 2.0
    assert payload["peak_frame"] >= 0


def test_drift_ui_json_empty_series_peak_defaults():
    from mmx_utils.drift_qc import measure_exterior_drift

    original = torch.rand(1, 32, 32, 3)
    mask = torch.ones(32, 32)
    result = measure_exterior_drift(original, original, mask, threshold_px=2.0)
    peak_px = max(result.drift_per_frame) if result.drift_per_frame else 0.0
    peak_frame = result.drift_per_frame.index(peak_px) if result.drift_per_frame else -1
    assert peak_px >= 0.0
    assert peak_frame == 0
