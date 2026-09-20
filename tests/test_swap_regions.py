"""The landmark groups that drive a swap, and the one that must never.

H3 has no face conditioning path - expression can only reach it through the
control video. So the control carries the dupe's performance, and the trap is
that it then also carries the dupe's ANATOMY: put the dupe's face contour in
the control and the swap comes back with the actor's texture on the dupe's
skull.

DWPose emits COCO-WholeBody 133, whose face block is the dlib-68 layout, and
the jaw is a contiguous run at the front of it (face 0-16). Excluding exactly
that run is what lets the dupe drive the performance while the reference
actor keeps their own head shape. These tests exist to keep it excluded.

CPU-only, no detector, no weights.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PACK = Path(__file__).resolve().parents[1]
if str(PACK) not in sys.path:
    sys.path.insert(0, str(PACK))

from mmx_utils.swap_regions import (  # noqa: E402
    EXPRESSION,
    FACE,
    GROUPS,
    N_WHOLEBODY,
    NEVER_RENDERED,
    SWAP_SCOPES,
    SwapRegionError,
    describe,
    group_indices,
    scope_groups,
    scope_indices,
    select,
)


# ── the layout is the one DWPose actually emits ─────────────────────────────

def test_the_groups_tile_the_133_points_without_overlap():
    """A gap or an overlap here silently mis-selects landmarks."""
    seen: list[int] = []
    for name in GROUPS:
        lo, hi = GROUPS[name]
        seen.extend(range(lo, hi))
    assert len(seen) == len(set(seen)), "groups overlap"
    assert sorted(seen) == list(range(N_WHOLEBODY)), "groups do not tile 0..132"


def test_the_face_block_is_the_dlib_68():
    lo, hi = FACE
    assert hi - lo == 68


def test_the_jaw_is_the_first_seventeen_face_points():
    lo, hi = GROUPS["jaw"]
    assert (lo - FACE[0], hi - FACE[0]) == (0, 17)


def test_the_mouth_is_the_last_twenty():
    lo, hi = GROUPS["mouth"]
    assert (lo - FACE[0], hi - FACE[0]) == (48, 68)


# ── the rule the whole design rests on ──────────────────────────────────────

@pytest.mark.parametrize("scope", sorted(SWAP_SCOPES))
def test_no_scope_ever_renders_the_jaw(scope):
    """THE rule. Rendering the jaw transfers the dupe's skull shape, which is
    the one thing a faceswap must not do - the head must stay the actor's."""
    for banned in NEVER_RENDERED:
        assert banned not in scope_groups(scope), (
            f"scope {scope!r} would draw the dupe's {banned} into the control")
    jaw = set(range(*GROUPS["jaw"]))
    assert not (set(scope_indices(scope)) & jaw)


def test_the_face_scope_still_carries_expression_and_lips():
    """Excluding the jaw must not have thrown the performance out with it."""
    groups = scope_groups("face")
    for needed in ("brows", "eyes", "mouth", "nose"):
        assert needed in groups, f"face scope dropped {needed}"


def test_lips_scope_is_the_mouth_and_nothing_else():
    assert scope_groups("lips") == ("mouth",)
    assert len(scope_indices("lips")) == 20


def test_person_scope_drives_body_and_face_but_still_not_the_jaw():
    groups = scope_groups("person")
    assert "body" in groups and "mouth" in groups
    assert "jaw" not in groups


def test_the_vfx_scopes_all_exist():
    """The named operations: lip-sync only, face, head, body, whole person."""
    assert set(SWAP_SCOPES) == {"lips", "face", "head", "body", "person"}


# ── selection ───────────────────────────────────────────────────────────────

def _kps(n=1):
    a = np.zeros((n, N_WHOLEBODY, 3), dtype=np.float32)
    a[..., 0] = np.arange(N_WHOLEBODY)          # x, so identity is checkable
    a[..., 1] = 100.0
    a[..., 2] = 1.0                              # all confident to start
    return a


def test_select_suppresses_everything_outside_the_scope():
    out = select(_kps(), "lips")
    kept = np.where(out[0, :, 2] > 0)[0]
    assert kept.tolist() == scope_indices("lips")


def test_select_keeps_the_133_point_layout():
    """Downstream renderers and detectors index by position; dropping rows
    would silently shift every landmark after the gap."""
    out = select(_kps(), "face")
    assert out.shape == (1, N_WHOLEBODY, 3)


def test_select_does_not_move_any_coordinate():
    src = _kps()
    out = select(src, "face")
    assert np.array_equal(out[..., :2], src[..., :2]), (
        "selection must suppress, never reposition - the whole point is that "
        "feature positions come through unchanged")


def test_select_does_not_mutate_its_input():
    src = _kps()
    before = src.copy()
    select(src, "face")
    assert np.array_equal(src, before)


def test_the_jaw_is_suppressed_by_every_scope():
    jaw = list(range(*GROUPS["jaw"]))
    for scope in SWAP_SCOPES:
        out = select(_kps(), scope)
        assert out[0, jaw, 2].max() == 0.0, f"{scope} left the jaw confident"


def test_a_body_only_pose_is_refused_with_the_reason():
    """17-point COCO has no face at all, so it cannot carry expression. Saying
    that is more useful than an index error 200 lines later."""
    with pytest.raises(SwapRegionError, match="no face landmarks"):
        select(np.zeros((1, 17, 3)), "face")


def test_keypoints_without_scores_are_refused():
    with pytest.raises(SwapRegionError, match="x, y, score"):
        select(np.zeros((1, N_WHOLEBODY, 2)), "face")


def test_an_unknown_scope_lists_the_real_ones():
    with pytest.raises(SwapRegionError, match="lips"):
        scope_groups("eyebrows_only")


def test_an_unknown_group_lists_the_real_ones():
    with pytest.raises(SwapRegionError, match="mouth"):
        group_indices(["forehead"])


# ── the report ──────────────────────────────────────────────────────────────

def test_the_report_says_the_jaw_is_excluded():
    text = describe("face")
    assert "jaw" in text.lower()
    assert "skull" in text.lower() or "shape" in text.lower()


def test_the_report_counts_the_landmarks_actually_driving():
    text = describe("lips")
    assert "20 of 133" in text
