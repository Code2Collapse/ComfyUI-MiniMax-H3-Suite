"""Context-window planning for long-form H3.

Every test pins a piece of H3 arithmetic that, if wrong, produces a plan that
LOOKS fine and corrupts the latent layout at sample time. CPU-only, no weights.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PACK = Path(__file__).resolve().parents[1]
if str(PACK) not in sys.path:
    sys.path.insert(0, str(PACK))

from mmx_utils.context_windows import (  # noqa: E402
    CYCLE,
    H3_ROPE_UNITS_PER_FRAME,
    ContextWindowError,
    is_legal_frame_count,
    latent_rows,
    plan_context_windows,
    plan_report,
    rope_offset_for_frame,
    snap_frame_count,
)


# ── the grid ────────────────────────────────────────────────────────────────

def test_legal_frame_counts_are_exactly_the_h3_grid():
    # INVARIANT: n % 17 == 5. This is THE constraint - a window off the grid
    # does not tile onto the latent rows and nothing downstream can fix it.
    assert is_legal_frame_count(5)          # head only
    assert is_legal_frame_count(22)         # head + 1 cycle
    assert is_legal_frame_count(39)         # head + 2 cycles
    assert is_legal_frame_count(396)        # head + 23 cycles
    assert not is_legal_frame_count(81)     # the popular default is NOT legal
    assert not is_legal_frame_count(409)    # nor is a 17s clip length
    assert not is_legal_frame_count(0)
    # only the WINDOW must sit on the grid; total_frames is free, which is why
    # the node's default total of 409 is fine while a window of 409 would not be.
    assert all((5 + 17 * k) % 17 == 5 for k in range(40))


def test_snap_up_never_shortens_the_clip():
    # INVARIANT: 'up' is the default because silently dropping requested frames
    # is worse than using slightly more memory.
    for n in (6, 20, 81, 100):
        s = snap_frame_count(n, "up")
        assert s >= n
        assert is_legal_frame_count(s)


def test_snap_down_never_lengthens_the_clip():
    # INVARIANT: 'down' exists to cap VRAM, so it must never exceed the request.
    for n in (6, 20, 81, 100):
        s = snap_frame_count(n, "down")
        assert s <= n
        assert is_legal_frame_count(s)


def test_latent_rows_matches_the_suite_formula():
    # INVARIANT: must agree with h3_constants.video_latent_t, which mirrors
    # comfy/ldm/minimax/model.py. Two implementations drifting apart is how a
    # layout silently goes wrong.
    for fc in (5, 22, 39, 90, 409):
        assert latent_rows(fc) == (2 if fc <= 5 else ((fc - 5) // 17) * 5 + 2)


def test_rope_offset_is_40hz_against_24fps():
    # INVARIANT: H3 carries position in a 40 Hz temporal RoPE while output is
    # 24 fps. Get this ratio wrong and every window believes it starts at t=0,
    # so the seams drift - the failure looks like "the model forgot the shot".
    assert H3_ROPE_UNITS_PER_FRAME == pytest.approx(40.0 / 24.0)
    assert rope_offset_for_frame(0) == 0.0
    assert rope_offset_for_frame(24) == pytest.approx(40.0)
    with pytest.raises(ContextWindowError):
        rope_offset_for_frame(-1)


# ── the tiling ──────────────────────────────────────────────────────────────

def test_every_window_starts_on_a_cycle_boundary():
    # INVARIANT: the reason the stride is forced to a multiple of 17. Window k
    # must share window k-1's internal latent structure or the passes cannot be
    # stitched at all.
    p = plan_context_windows(409, 81, 17)
    assert p["stride_frames"] % CYCLE == 0
    for w in p["windows"][1:]:
        assert w["start"] % CYCLE == 0, w


def test_plan_covers_the_whole_request_and_never_overruns():
    # INVARIANT: the tiling must reach total_frames exactly - not short (missing
    # footage) and not past it (frames nobody asked for).
    for total in (60, 409, 1000):
        p = plan_context_windows(total, 90, 22)
        assert p["covered_frames"] == total
        assert all(w["end"] <= total for w in p["windows"])
        assert p["windows"][0]["start"] == 0


def test_windows_actually_overlap_by_the_reported_amount():
    # INVARIANT: the report's overlap number must be the real one, because it is
    # what the stitch node cross-fades over.
    p = plan_context_windows(409, 90, 22)
    ov = p["overlap_frames"]
    for a, b in zip(p["windows"], p["windows"][1:]):
        assert a["end"] - b["start"] == ov
        assert b["overlap_prev"] == ov


def test_rope_offsets_increase_with_the_window_start():
    # INVARIANT: monotonic and proportional. A repeated or out-of-order offset
    # would place two passes at the same point on the timeline.
    p = plan_context_windows(409, 90, 22)
    offs = [w["rope_offset"] for w in p["windows"]]
    assert offs == sorted(offs)
    for w in p["windows"]:
        assert w["rope_offset"] == pytest.approx(w["start"] * H3_ROPE_UNITS_PER_FRAME)


def test_single_window_when_the_clip_already_fits():
    # INVARIANT: no pointless tiling. A clip shorter than one window is one pass.
    p = plan_context_windows(40, 90, 22)
    assert p["window_count"] == 1
    assert p["windows"][0]["rope_offset"] == 0.0
    assert p["windows"][0]["short_tail"] is True


# ── refusals ────────────────────────────────────────────────────────────────

def test_overlap_at_or_above_the_window_is_refused():
    # INVARIANT: the windows would never advance - an infinite plan. Refuse with
    # a sentence rather than looping or silently clamping.
    with pytest.raises(ContextWindowError, match="never advance"):
        plan_context_windows(409, 90, 90)


def test_impossible_requests_are_named_not_swallowed():
    # INVARIANT: every refusal says which widget is wrong.
    with pytest.raises(ContextWindowError, match="total_frames"):
        plan_context_windows(0, 90, 22)
    with pytest.raises(ContextWindowError, match="window_frames"):
        plan_context_windows(409, 2, 0)
    with pytest.raises(ContextWindowError, match="negative"):
        plan_context_windows(409, 90, -1)


# ── the report ──────────────────────────────────────────────────────────────

def test_report_says_what_it_changed_and_why():
    # INVARIANT: snapping 81 -> 90 behind the artist's back, unexplained, is how
    # someone spends an afternoon wondering why their frame count moved.
    p = plan_context_windows(409, 81, 17)
    r = plan_report(p)
    assert "81" in r and "90" in r
    assert "17" in r
    assert "snapped" in r.lower()
    assert "cycle" in r.lower()


def test_report_flags_a_short_tail():
    # INVARIANT: the last window is usually short; saying so prevents a confused
    # "why is my last clip a different length" report.
    p = plan_context_windows(200, 90, 22)
    if p["windows"][-1]["short_tail"]:
        assert "tail" in plan_report(p).lower()
