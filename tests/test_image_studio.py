"""H3 Image Studio pure logic — CPU-only, no weights, no network."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

PACK = Path(__file__).resolve().parents[1]
if str(PACK) not in sys.path:
    sys.path.insert(0, str(PACK))

from mmx_utils.h3_constants import CANVAS_MULTIPLE, MAX_PIXELS  # noqa: E402
from mmx_utils.h3_grid import decoded_frames_for_latent_t, latent_t_for_frame_count  # noqa: E402
from mmx_utils.image_studio import (  # noqa: E402
    FRAME_PRESETS,
    calculate_custom_resolution,
    calculate_preset_resolution,
    detail_tone_lock,
    frame_profile_packet,
    prompt_warning,
    select_still_frame,
)


@pytest.mark.parametrize("aspect", ["16:9 landscape", "1:1 square", "9:16 portrait"])
def test_resolution_fitting_is_grid_aligned_and_capped(aspect):
    width, height, report = calculate_preset_resolution(
        aspect, "native detail | 0.98 MP", limit_to_native_area=True,
    )
    assert width % CANVAS_MULTIPLE == 0
    assert height % CANVAS_MULTIPLE == 0
    assert width * height <= MAX_PIXELS
    assert "×" in report
    assert str(width) in report


def test_custom_resolution_respects_native_cap():
    width, height, report = calculate_custom_resolution(
        "custom dimensions", 4.0, 32, True, 2048, 2048,
    )
    assert width % 32 == 0 and height % 32 == 0
    assert width * height <= MAX_PIXELS
    assert "native cap on" in report


@pytest.mark.parametrize("profile", list(FRAME_PRESETS.keys()))
def test_frame_profiles_map_to_valid_latent_packets(profile):
    requested, latent_t, natural = frame_profile_packet(profile)
    assert requested == FRAME_PRESETS[profile]
    assert latent_t >= 1
    assert natural >= requested
    assert decoded_frames_for_latent_t(latent_t) == natural
    # natural decode length must cover the requested profile
    assert natural >= FRAME_PRESETS[profile]


def test_latent_t_for_frame_count_matches_upstream_table():
    assert latent_t_for_frame_count(1) == (1, 1)
    assert latent_t_for_frame_count(5) == (2, 5)
    assert latent_t_for_frame_count(9) == (3, 9)
    assert latent_t_for_frame_count(20) == (7, 22)


def _synthetic_sharpest_sequence(n: int = 5, sharp_at: int = 2) -> torch.Tensor:
    """Blurry frames except one with a high-frequency checker."""
    frames = []
    for i in range(n):
        if i == sharp_at:
            checker = torch.zeros(64, 64, 3)
            checker[::2, ::2] = 1.0
            checker[1::2, 1::2] = 1.0
            frames.append(checker)
        else:
            frames.append(torch.full((64, 64, 3), 0.5))
    return torch.stack(frames)


def test_frame_selector_picks_sharpest_frame():
    frames = _synthetic_sharpest_sequence(sharp_at=2)
    _primary, _debug, index, _score, report = select_still_frame(frames, "sharpest")
    assert index == 2
    assert "frame 2" in report


def test_empty_prompt_has_no_warning():
    assert prompt_warning("") == ""
    assert prompt_warning("a portrait in soft light") == ""


def test_video_phrased_prompt_warns():
    warning = prompt_warning("camera pans left over 3 seconds with zoom")
    assert "WARNING" in warning
    assert "motion" in warning.lower() or "still" in warning.lower()


def test_select_still_frame_does_not_mutate_input():
    frames = _synthetic_sharpest_sequence()
    before = frames.clone()
    select_still_frame(frames, "stable_quality")
    assert torch.equal(frames, before)


def test_detail_tone_lock_does_not_mutate_inputs():
    src = torch.rand(1, 32, 32, 3)
    ref = torch.rand(1, 32, 32, 3)
    src_before = src.clone()
    ref_before = ref.clone()
    detail_tone_lock(src, ref, 0.5, 0.5, 8)
    assert torch.equal(src, src_before)
    assert torch.equal(ref, ref_before)


def test_preset_resolution_report_names_dimensions():
    width, height, report = calculate_preset_resolution("16:9 landscape", "balanced | 0.70 MP")
    assert f"{width}×{height}" in report


def test_frame_selector_report_names_selected_index():
    frames = _synthetic_sharpest_sequence(sharp_at=3)
    _primary, _debug, index, _score, report = select_still_frame(frames, "sharpest")
    assert f"frame {index}" in report
