"""The landmark groups that drive a swap.

H3 has no face conditioning path - expression can only reach it through the
control video, so the control carries the dupe's whole performance.

The jaw (face points 0-16) is the awkward group, and it is a trade rather than
a bug. Those points carry head POSE and the CHIN DROP that lets the mouth
open, as well as the dupe's skull SHAPE. Driving them brings the first two and
risks the third; the region is masked as well, so the reference actor's skull
can still come through, and ControlNet strength decides who wins.
`drive_jaw=False` is the lever for when identity beats pose, and both paths
are pinned here.

The array is 135: COCO-WholeBody's 133 plus the two pupils, which are the only
eye-DIRECTION signal - the 68-point eye contours give the lid, never the gaze.

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
    N_EXTENDED,
    N_WHOLEBODY,
    SWAP_SCOPES,
    SwapRegionError,
    describe,
    group_indices,
    scope_groups,
    scope_indices,
    select,
)


# ── the layout is the one DWPose actually emits ─────────────────────────────

def test_the_groups_tile_every_point_without_overlap():
    """A gap or an overlap here silently mis-selects landmarks."""
    seen: list[int] = []
    for name in GROUPS:
        lo, hi = GROUPS[name]
        seen.extend(range(lo, hi))
    assert len(seen) == len(set(seen)), "groups overlap"
    assert sorted(seen) == list(range(N_EXTENDED)), "groups do not tile 0..134"


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

@pytest.mark.parametrize("scope", ["face", "head", "person", "lips"])
def test_every_face_scope_drives_the_jaw_by_default(scope):
    jaw = set(range(*GROUPS["jaw"]))
    assert set(scope_indices(scope)) & jaw, f"{scope} drives no jaw point"


@pytest.mark.parametrize("scope", ["face", "head", "person", "lips"])
def test_drive_jaw_false_removes_it(scope):
    jaw = set(range(*GROUPS["jaw"]))
    assert not (set(scope_indices(scope, drive_jaw=False)) & jaw)


def test_the_face_scope_carries_the_whole_performance():
    groups = scope_groups("face")
    for needed in ("jaw", "brows", "eyes", "mouth", "nose", "pupils"):
        assert needed in groups, f"face scope dropped {needed}"


def test_lips_scope_is_mouth_plus_the_jaw_that_opens_it():
    assert set(scope_groups("lips")) == {"mouth", "jaw"}


def test_person_scope_drives_body_and_face_together():
    groups = scope_groups("person")
    assert "body" in groups and "mouth" in groups and "jaw" in groups


def test_pupils_are_a_group_and_are_driven_for_gaze():
    from mmx_utils.swap_regions import PUPILS

    assert GROUPS["pupils"] == PUPILS
    assert "pupils" in scope_groups("face")


def test_the_vfx_scopes_all_exist():
    """The named operations: lip-sync only, face, head, body, whole person."""
    assert set(SWAP_SCOPES) == {"lips", "face", "head", "body", "person"}


# ── selection ───────────────────────────────────────────────────────────────

def _kps(n=1):
    a = np.zeros((n, N_EXTENDED, 3), dtype=np.float32)
    a[..., 0] = np.arange(N_EXTENDED)          # x, so identity is checkable
    a[..., 1] = 100.0
    a[..., 2] = 1.0                              # all confident to start
    return a


def test_select_suppresses_everything_outside_the_scope():
    out = select(_kps(), "lips")
    kept = np.where(out[0, :, 2] > 0)[0]
    assert kept.tolist() == scope_indices("lips")


def test_select_keeps_the_full_point_layout():
    """Downstream renderers and detectors index by position; dropping rows
    would silently shift every landmark after the gap."""
    out = select(_kps(), "face")
    assert out.shape == (1, N_EXTENDED, 3)


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


def test_the_jaw_survives_selection_for_a_face_scope():
    jaw = list(range(*GROUPS["jaw"]))
    assert select(_kps(), "face")[0, jaw, 2].max() > 0.0


def test_drive_jaw_false_suppresses_it_in_selection():
    jaw = list(range(*GROUPS["jaw"]))
    out = select(_kps(), "face", drive_jaw=False)
    assert out[0, jaw, 2].max() == 0.0


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

def test_the_report_names_the_jaw_and_the_knob():
    text = describe("face").lower()
    assert "jaw" in text
    assert "strength" in text, "a user seeing the dupe's head needs the knob named"


def test_the_report_counts_the_landmarks_actually_driving():
    """lips drives mouth (20) + jaw (17) = 37 of the 135-point array."""
    assert "37 of 135" in describe("lips")


def test_the_report_counts_fewer_when_the_jaw_is_dropped():
    assert "20 of 135" in describe("lips", drive_jaw=False)


# ── drive_mouth: who decides the lip shape ──────────────────────────────────
#
# The three cases the swap has to serve, and they are genuinely different:
#
#   1. lips masked, no audio      -> control drives the lips (the dupe's own
#                                    performance is copied)
#   2. face masked, audio locked  -> BOTH can drive the mouth. Same take: they
#                                    agree. A dub: they fight, control wins,
#                                    and you get old lips over new words.
#   3. face masked, no audio      -> control drives the lips, as case 1
#
# drive_mouth is the lever for case 2 only. Off, the mouth is MASKED but NOT
# CONTROLLED, which is the one configuration where the audio decides.

def test_the_mouth_is_driven_by_default_so_silent_lip_movement_works():
    """The user's case 3: no audio, face masked, lips must still move with the
    dupe. That only happens if the control draws them."""
    for scope in ("lips", "face", "head", "person"):
        assert "mouth" in scope_groups(scope), scope


def test_drive_mouth_false_removes_the_lips_from_the_control():
    mouth = set(range(*GROUPS["mouth"]))
    for scope in ("lips", "face", "head", "person"):
        assert "mouth" not in scope_groups(scope, drive_mouth=False), scope
        assert not (set(scope_indices(scope, drive_mouth=False)) & mouth), scope


def test_drive_mouth_false_still_leaves_the_mouth_masked():
    """This is the whole point. The lips must stay free to CHANGE - otherwise
    the audio has nothing to change - while nothing tells them what to be."""
    from mmx_utils.swap_regions import MASK_GROUPS

    for scope in ("lips", "face", "head", "person"):
        assert "mouth" in MASK_GROUPS[scope], (
            f"{scope} stopped masking the mouth; audio could not move the lips")


def test_drive_mouth_does_not_disturb_the_other_expression_groups():
    kept = set(scope_groups("face", drive_mouth=False))
    for group in ("jaw", "brows", "nose", "eyes", "pupils"):
        assert group in kept, f"{group} was lost with the mouth"


def test_the_two_levers_are_independent():
    both_off = set(scope_groups("face", drive_jaw=False, drive_mouth=False))
    assert "jaw" not in both_off and "mouth" not in both_off
    assert {"brows", "nose", "eyes", "pupils"} <= both_off


def test_selection_drops_the_mouth_points_too():
    out = select(_kps(), "face", drive_mouth=False)
    lo, hi = GROUPS["mouth"]
    assert out[:, lo:hi, 2].max() == 0.0, "mouth points still carry confidence"


def test_describe_says_the_audio_is_deciding():
    text = describe("face", drive_mouth=False)
    assert "masked but NOT driven" in text
    assert "audio" in text
    # and it must warn about the configuration that does nothing at all
    assert "no audio" in text.lower()


def test_describe_stays_quiet_when_the_mouth_is_driven():
    assert "masked but NOT driven" not in describe("face")
