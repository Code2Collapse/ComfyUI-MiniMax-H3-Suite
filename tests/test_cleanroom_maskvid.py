"""Clean-room Phase B tests — differential denoise + frame-range mask.

Spec: docs/cleanroom_maskvid.md. CPU-only, weight-free. Every test carries an
INVARIANT line naming what it pins.

The endpoint tests matter most. A ramp is easy to write in a way that looks
right and quietly breaks the contract with stock differential diffusion at
mask==1 or mask==0, and the picture would still look plausible.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

PACK = Path(__file__).resolve().parents[1]
if str(PACK) not in sys.path:
    sys.path.insert(0, str(PACK))

from mmx_utils.differential_denoise import (  # noqa: E402
    band_width_in_pixels,
    schedule_threshold,
    soft_step_mask,
)
from mmx_utils.frame_ranges import FrameRangeError, parse_frame_ranges  # noqa: E402


# ── A1: differential denoise ────────────────────────────────────────────────

def test_softness_zero_is_exactly_the_hard_threshold():
    # INVARIANT: softness=0 must reproduce stock differential diffusion bit for
    # bit, so the node is a safe drop-in and can be A/B'd against it.
    m = torch.linspace(0, 1, 11)
    for threshold in (0.0, 0.25, 0.5, 0.75, 1.0):
        assert torch.equal(soft_step_mask(m, threshold, 0.0), (m >= threshold).float())


def test_ramp_endpoints_match_stock_at_both_ends():
    # INVARIANT: the reason the ramp centre traverses [s/2, 1-s/2] and not [0,1].
    # A mask value of 1 must be fully open on the FIRST step (threshold 1.0), and
    # a value of 0 must stay shut on the LAST (threshold 0.0). Centring on [0,1]
    # breaks both, and the output still looks like a picture.
    for softness in (0.1, 0.2, 0.5, 0.9):
        first = soft_step_mask(torch.tensor([1.0, 0.0]), 1.0, softness)
        last = soft_step_mask(torch.tensor([1.0, 0.0]), 0.0, softness)
        assert first[0].item() == pytest.approx(1.0), f"mask=1 not open at step 0 (s={softness})"
        assert last[1].item() == pytest.approx(0.0), f"mask=0 leaked at the last step (s={softness})"


def test_output_is_monotonic_in_mask_value():
    # INVARIANT: a more-masked pixel is never denoised LESS than a less-masked
    # one. A sign slip would invert the mask and still produce output.
    m = torch.linspace(0, 1, 64)
    out = soft_step_mask(m, 0.5, 0.25)
    assert torch.all(out[1:] >= out[:-1] - 1e-6)


def test_soft_edge_is_wider_than_hard_edge():
    # INVARIANT: softness must actually soften. Counts partial (non-binary)
    # values - the hard path has none by construction.
    m = torch.linspace(0, 1, 256)
    hard = soft_step_mask(m, 0.5, 0.0)
    soft = soft_step_mask(m, 0.5, 0.3)
    partial = lambda t: int(((t > 1e-6) & (t < 1 - 1e-6)).sum())
    assert partial(hard) == 0
    assert partial(soft) > 10


def test_band_width_scales_inversely_with_mask_gradient():
    # INVARIANT: the property that justifies the whole design - a feathered mask
    # (small gradient) keeps a WIDE soft edge, a sharp mask keeps a narrow one.
    feathered = band_width_in_pixels(0.2, 0.01)   # gentle gradient
    sharp = band_width_in_pixels(0.2, 1.0)        # steep gradient
    assert feathered > sharp * 10
    assert band_width_in_pixels(0.2, 0.0) == float("inf")


def test_softness_one_disables_the_schedule():
    # INVARIANT: documented degenerate case - blend by the raw mask, no schedule.
    m = torch.rand(32)
    for threshold in (0.0, 0.5, 1.0):
        assert torch.allclose(soft_step_mask(m, threshold, 1.0), m, atol=1e-6)


def test_strength_blends_toward_the_raw_mask():
    # INVARIANT: strength=0 ignores the schedule; 1 applies it fully.
    m = torch.linspace(0, 1, 16)
    assert torch.allclose(soft_step_mask(m, 0.5, 0.0, strength=0.0), m, atol=1e-6)
    assert torch.equal(soft_step_mask(m, 0.5, 0.0, strength=1.0), (m >= 0.5).float())


def test_output_never_leaves_zero_one():
    # INVARIANT: the result is a mask. Values outside [0,1] would scale the
    # denoise beyond its intended range.
    m = torch.linspace(-0.5, 1.5, 128)      # hostile: out-of-range input
    out = soft_step_mask(m, 0.4, 0.3)
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_schedule_runs_from_one_to_zero():
    # INVARIANT: 1.0 at the first step falling to 0.0 at the last, in TIMESTEP
    # space. Identity timestep keeps the test about the normalisation.
    ident = lambda s: float(s)
    assert schedule_threshold(1.0, 1.0, 0.0, ident) == pytest.approx(1.0)
    assert schedule_threshold(0.0, 1.0, 0.0, ident) == pytest.approx(0.0)
    assert schedule_threshold(0.5, 1.0, 0.0, ident) == pytest.approx(0.5)


def test_schedule_uses_the_higher_of_floor_and_last_sigma():
    # INVARIANT: a partial-denoise run that stops early must still span a full
    # 1 -> 0 sweep, not compress into a fraction of it.
    ident = lambda s: float(s)
    assert schedule_threshold(0.4, 1.0, 0.4, ident, sigma_floor=0.0) == pytest.approx(0.0)


def test_degenerate_single_step_schedule_does_not_divide_by_zero():
    # INVARIANT: first == last must not produce inf/nan.
    val = schedule_threshold(1.0, 1.0, 1.0, lambda s: float(s))
    assert val == 0.0


# ── A2: frame ranges ────────────────────────────────────────────────────────

def test_slice_stop_is_excluded():
    # INVARIANT: Python semantics. Off-by-one here silently shifts every edit.
    assert parse_frame_ranges("0:5", 24) == [0, 1, 2, 3, 4]


def test_single_frame_and_step_and_open_bounds():
    # INVARIANT: the documented syntax actually parses.
    assert parse_frame_ranges("7", 24) == [7]
    assert parse_frame_ranges("0:6:2", 24) == [0, 2, 4]
    assert parse_frame_ranges(":3", 24) == [0, 1, 2]
    assert parse_frame_ranges("21:", 24) == [21, 22, 23]
    assert parse_frame_ranges("21:end", 24) == [21, 22, 23]


def test_negative_indices_count_from_the_end():
    # INVARIANT: -1 is the last frame, as in list indexing.
    assert parse_frame_ranges("-1", 24) == [23]
    assert parse_frame_ranges("-3:", 24) == [21, 22, 23]


def test_overlapping_segments_collapse_and_sort():
    # INVARIANT: the result is a set, so overlapping ranges cannot double-count.
    assert parse_frame_ranges("5:8, 6:10, 2", 24) == [2, 5, 6, 7, 8, 9]


def test_whitespace_and_newlines_are_insignificant():
    # INVARIANT: a multiline box means users WILL use newlines and spaces.
    assert parse_frame_ranges(" 0 : 3 ,\n 10 \n", 24) == [0, 1, 2, 10]


def test_out_of_range_slice_clamps_but_single_frame_raises():
    # INVARIANT: the deliberate asymmetry. '0:9999' plainly means "all of them";
    # 'frame 9999' on a 24-frame clip is a typo the author needs told about.
    assert parse_frame_ranges("0:9999", 24) == list(range(24))
    with pytest.raises(FrameRangeError, match="outside a clip"):
        parse_frame_ranges("9999", 24)


def test_authoring_mistakes_are_named_not_swallowed():
    # INVARIANT: every error names the offending segment and shows the syntax.
    for bad, needle in (("0:2:0", "step of zero"),
                        ("end:5", "END of a range"),
                        ("1:2:3:4", "could not read"),
                        ("abc", "not a frame number")):
        with pytest.raises(FrameRangeError, match=needle):
            parse_frame_ranges(bad, 24)


def test_zero_frame_count_is_rejected_before_parsing():
    # INVARIANT: caught early - the negative-index modulo would otherwise raise
    # ZeroDivisionError, which tells the user nothing.
    with pytest.raises(FrameRangeError, match="at least 1"):
        parse_frame_ranges("0", 0)


def test_empty_text_selects_nothing():
    # INVARIANT: empty input is not an error, it is an empty selection.
    assert parse_frame_ranges("", 24) == []
    assert parse_frame_ranges("  ,\n , ", 24) == []
