"""CPU-only tests for ComfyUI-Spectrum-MiniMax-H3 port (config + forecast helpers).

Attribution: ComfyUI-Spectrum-MiniMax-H3, author xmarre, GPL-3.0.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

PACK = Path(__file__).resolve().parents[1]
if str(PACK) not in sys.path:
    sys.path.insert(0, str(PACK))

from mmx_utils.spectrum_h3._copy_from_third_party import ensure_ported

ensure_ported()

from mmx_utils.spectrum_h3.config import SpectrumH3Config
from mmx_utils.spectrum_h3.forecast import HistoryWeightForecaster
from mmx_utils.spectrum_h3.nodes import _effective_bootstrap_first_forecast


def _direct_prediction(coords, features, target, degree, ridge_lambda, blend):
    design = HistoryWeightForecaster.chebyshev_design(coords, degree)
    phi = HistoryWeightForecaster.chebyshev_design(torch.tensor([target]), degree)
    flat = features.to(torch.float32).reshape(features.shape[0], -1)
    lhs = design.T @ design + ridge_lambda * torch.eye(degree + 1)
    coefficients = torch.linalg.solve(lhs, design.T @ flat)
    spectral = (phi @ coefficients).reshape(features.shape[1:])
    spacing = float(coords[-1] - coords[-2])
    ratio = (target - float(coords[-1])) / spacing
    linear = features[-1].float() + ratio * (features[-1].float() - features[-2].float())
    return blend * spectral + (1.0 - blend) * linear


def test_config_rejects_fractional_degree():
    # INVARIANT: degree must be integral — fractional values are rejected at validate().
    with pytest.raises(ValueError, match="integer"):
        SpectrumH3Config(degree=2.5).validate()


def test_config_defaults_match_upstream():
    # INVARIANT: ported defaults match upstream SpectrumH3Config baseline.
    config = SpectrumH3Config().validate()
    assert config.degree == 1
    assert config.warmup_steps == 1
    assert config.bootstrap_first_forecast is True
    assert config.offline_smoothing_replay is True
    assert config.model_aware_mode == "off"
    assert config.generic_correction_mode == "coordinate_rls"


def test_bootstrap_first_forecast_disabled_when_incompatible():
    # INVARIANT: bootstrap_first_forecast requires degree=1 and warmup_steps<=1.
    assert _effective_bootstrap_first_forecast(
        requested=True,
        degree=2,
        warmup_steps=1,
    ) is False
    assert _effective_bootstrap_first_forecast(
        requested=True,
        degree=1,
        warmup_steps=2,
    ) is False
    assert _effective_bootstrap_first_forecast(
        requested=True,
        degree=1,
        warmup_steps=1,
    ) is True


@pytest.mark.parametrize("feature_dtype", [torch.float16, torch.float32, torch.bfloat16])
def test_forecast_matches_direct_coefficients(feature_dtype):
    # INVARIANT: HistoryWeightForecaster matches direct Chebyshev ridge solve.
    torch.manual_seed(11)
    coords = torch.tensor([-1.0, -0.7, -0.3, 0.1, 0.35], dtype=torch.float32)
    features = torch.randn(5, 3, 4, 6).to(feature_dtype)
    forecaster = HistoryWeightForecaster(
        degree=3,
        ridge_lambda=0.1,
        max_history=8,
        chunk_bytes=4096,
    )
    for coordinate, feature in zip(coords, features, strict=True):
        forecaster.update(float(coordinate), feature)
    actual = forecaster.predict(0.55, blend_weight=0.6)
    expected = _direct_prediction(coords, features, 0.55, 3, 0.1, 0.6).to(feature_dtype)
    tolerance = 2e-2 if feature_dtype != torch.float32 else 2e-5
    torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)


def test_forecast_chunked_rows_match_full_prediction():
    # INVARIANT: row-chunked predict matches full-tensor predict on the same history.
    torch.manual_seed(21)
    forecaster = HistoryWeightForecaster(degree=2, ridge_lambda=0.1, max_history=5, chunk_bytes=4096)
    for index, coordinate in enumerate((-1.0, -0.4, 0.2)):
        forecaster.update(coordinate, torch.randn(4, 17, 19) + index)
    full = forecaster.predict(0.7, blend_weight=0.5)
    subset = forecaster.predict(0.7, blend_weight=0.5, rows=(3, 1))
    torch.testing.assert_close(subset, full[(3, 1), ...])
