"""Tests for H3 Color QC metrics."""

from __future__ import annotations

import torch

from mmx_utils.color_qc import colourfulness_band, hasler_susstrunk_display, measure_color_qc


def _solid_clip(r: float, g: float, b: float, frames: int = 4) -> torch.Tensor:
    rgb = torch.tensor([r, g, b], dtype=torch.float32).view(1, 1, 1, 3)
    return rgb.expand(frames, 64, 64, 3).clone()


def test_saturated_red_hasler_value():
    red = _solid_clip(1.0, 0.0, 0.0)
    cf_red, _ = hasler_susstrunk_display(red)
    assert abs(cf_red - 85.5) < 0.1

    grey = _solid_clip(0.5, 0.5, 0.5)
    cf_grey, _ = hasler_susstrunk_display(grey)
    assert cf_grey == 0.0


def test_saturated_scores_higher_than_greyscale():
    sat = _solid_clip(1.0, 0.0, 0.0)
    grey = _solid_clip(0.5, 0.5, 0.5)
    cf_sat, _ = hasler_susstrunk_display(sat)
    cf_grey, _ = hasler_susstrunk_display(grey)
    assert cf_sat > cf_grey


def test_colourfulness_bands():
    assert "not colourful" in colourfulness_band(10)
    assert "extremely colourful" in colourfulness_band(120)


def test_headroom_clipped_vs_scaled():
    blown = _solid_clip(1.0, 1.0, 1.0)
    scaled = blown * 0.8
    m_blown, _ = measure_color_qc(blown)
    m_scaled, _ = measure_color_qc(scaled)
    assert m_blown.clipped_fraction > m_scaled.clipped_fraction
    assert m_blown.headroom < m_scaled.headroom


def test_oklab_distance_zero_for_identical_reference():
    # The reference is a still image; a clip made of that image matches it.
    img = torch.rand(1, 32, 32, 3)
    clip = img.repeat(3, 1, 1, 1)
    m, _ = measure_color_qc(clip, img)
    assert m.oklab_mean_dist is not None
    assert m.oklab_mean_dist < 1e-3


def test_oklab_distance_increases_when_tinted_red():
    clip = torch.rand(2, 32, 32, 3)
    m_base, _ = measure_color_qc(clip, clip)
    tinted = clip.clone()
    tinted[..., 0] = (tinted[..., 0] + 0.3).clamp(0.0, 1.0)
    m_tinted, _ = measure_color_qc(tinted, clip)
    assert m_tinted.oklab_mean_dist is not None
    assert m_tinted.oklab_mean_dist > m_base.oklab_mean_dist
