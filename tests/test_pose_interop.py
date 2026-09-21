"""POSE_KEYPOINT (OpenPose JSON) -> COCO-WholeBody 133.

Every pose node in the ecosystem hands over POSE_KEYPOINT, which is OpenPose
JSON, and that is not the layout the swap tables index. The differences are
the silent kind:

  * OpenPose body is COCO-18 with a NECK at index 1 and a different head
    order. Mapping it as a slice swaps left and right, which on a face swap
    MIRRORS the expression - a wink on the wrong eye, and nothing raises.
  * OpenPose face is 70 points: dlib-68 plus two pupils with no COCO slot.
  * ComfyUI's POSE_KEYPOINT is NORMALISED 0..1; the renderer wants pixels.

Each of those is pinned below, because each fails silently.

CPU-only, numpy only.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PACK = Path(__file__).resolve().parents[1]
if str(PACK) not in sys.path:
    sys.path.insert(0, str(PACK))

from mmx_utils.pose_interop import (  # noqa: E402
    PoseInteropError,
    describe_source,
    has_face,
    person_to_wholebody,
    pose_keypoint_to_wholebody,
)
from mmx_utils.swap_regions import (  # noqa: E402
    FACE,
    GROUPS,
    N_EXTENDED,
    PUPILS,
)

W, H = 640, 480


def op_person(body=True, face=True, hands=True):
    """An OpenPose person whose every value encodes its own index, so a
    permutation bug shows up as a wrong number rather than a wrong picture."""
    p = {}
    if body:
        # x = index/100 so it survives the normalised->pixel scaling readably
        p["pose_keypoints_2d"] = [v for i in range(18)
                                  for v in (i / 100.0, 0.5, 1.0)]
    if face:
        p["face_keypoints_2d"] = [v for i in range(70)
                                  for v in (i / 100.0, 0.25, 1.0)]
    if hands:
        p["hand_left_keypoints_2d"] = [v for i in range(21)
                                       for v in (i / 100.0, 0.75, 1.0)]
        p["hand_right_keypoints_2d"] = [v for i in range(21)
                                        for v in (i / 100.0, 0.9, 1.0)]
    return p


def frame(**kw):
    return {"people": [op_person(**kw)], "canvas_width": W, "canvas_height": H}


# ── the body permutation, which is the one that mirrors a face ──────────────

def test_the_body_is_permuted_not_sliced():
    """COCO-17 slot 1 is the LEFT eye, which is OpenPose slot 15 - not slot 1,
    which is the neck. A slice here mirrors the skeleton."""
    kp = person_to_wholebody(op_person(), (W, H))
    # x was index/100, then scaled by W
    assert round(kp[1, 0] / W * 100) == 15, "COCO left-eye did not come from OP 15"
    assert round(kp[2, 0] / W * 100) == 14, "COCO right-eye did not come from OP 14"


def test_left_and_right_are_not_swapped():
    kp = person_to_wholebody(op_person(), (W, H))
    pairs = {5: 5, 6: 2, 7: 6, 8: 3, 9: 7, 10: 4,      # shoulders/elbows/wrists
             11: 11, 12: 8, 13: 12, 14: 9, 15: 13, 16: 10}
    for coco, op in pairs.items():
        assert round(kp[coco, 0] / W * 100) == op, f"COCO {coco} != OpenPose {op}"


def test_the_neck_is_dropped():
    """OpenPose slot 1 is a neck; COCO-WholeBody has no neck. If it leaked in
    it would land on a real landmark and drag a line to the wrong place."""
    kp = person_to_wholebody(op_person(), (W, H))
    body_sources = {round(kp[i, 0] / W * 100) for i in range(17)}
    assert 1 not in body_sources


# ── the face block ──────────────────────────────────────────────────────────

def test_the_first_68_face_points_land_in_the_face_block():
    kp = person_to_wholebody(op_person(), (W, H))
    for i in (0, 17, 48, 67):
        assert round(kp[FACE[0] + i, 0] / W * 100) == i


def test_the_two_pupils_are_kept_at_the_end():
    """They are the only eye-DIRECTION signal - the 68-point eye contours give
    the lid, never the gaze - so they are appended at 133,134 rather than
    dropped. Appending, not inserting: inserting would shift every hand index.
    """
    kp = person_to_wholebody(op_person(), (W, H))
    assert FACE[1] - FACE[0] == 68, "the face block must stay dlib-68"
    assert round(kp[PUPILS[0], 0] / W * 100) == 68
    assert round(kp[PUPILS[1] - 1, 0] / W * 100) == 69
    # the hand block must still start exactly where the table says
    lo, _ = GROUPS["left_hand"]
    assert round(kp[lo, 0] / W * 100) == 0, "left hand did not start at its slot"


def test_a_pose_with_pupils_reports_gaze_as_available():
    from mmx_utils.pose_interop import has_pupils

    kp, canvas = pose_keypoint_to_wholebody([frame()])
    assert has_pupils(kp)
    assert "No pupils" not in describe_source(kp, canvas)


def test_a_pose_without_pupils_says_gaze_cannot_be_driven():
    from mmx_utils.pose_interop import has_pupils

    kp, canvas = pose_keypoint_to_wholebody([frame(face=False)])
    assert not has_pupils(kp)
    text = describe_source(kp, canvas)
    assert "eye DIRECTION cannot be driven" in text


def test_hands_are_copied_in_order():
    kp = person_to_wholebody(op_person(), (W, H))
    for group in ("left_hand", "right_hand"):
        lo, hi = GROUPS[group]
        assert hi - lo == 21
        for i in (0, 10, 20):
            assert round(kp[lo + i, 0] / W * 100) == i


def test_feet_stay_unconfident_because_openpose_coco18_has_none():
    """Inventing feet would draw lines to (0,0)."""
    kp = person_to_wholebody(op_person(), (W, H))
    lo, hi = GROUPS["feet"]
    assert kp[lo:hi, 2].max() == 0.0


# ── normalised vs pixel ─────────────────────────────────────────────────────

def test_normalised_coordinates_are_scaled_to_pixels():
    kp = person_to_wholebody(op_person(), (W, H))
    assert kp[:, 0].max() > 1.5, "still normalised - the skeleton would be 1px"
    assert kp[FACE[0], 1] == pytest.approx(0.25 * H)


def test_pixel_coordinates_are_left_alone():
    """A detector that already emits pixels must not be scaled again."""
    p = {"pose_keypoints_2d": [v for i in range(18) for v in (i * 20.0, 100.0, 1.0)]}
    kp = person_to_wholebody(p, (W, H))
    assert kp[0, 1] == pytest.approx(100.0), "pixel input was rescaled"


# ── the whole clip ──────────────────────────────────────────────────────────

def test_a_clip_becomes_a_stacked_array():
    kp, canvas = pose_keypoint_to_wholebody([frame(), frame(), frame()])
    assert kp.shape == (3, N_EXTENDED, 3)
    assert canvas == (W, H)


def test_a_frame_with_nobody_in_it_is_blank_not_an_error():
    """Detection drops out mid-shot; that is a gap, not a failure."""
    kp, _ = pose_keypoint_to_wholebody(
        [frame(), {"people": [], "canvas_width": W, "canvas_height": H}])
    assert kp[1, :, 2].max() == 0.0


def test_a_missing_canvas_is_named():
    with pytest.raises(PoseInteropError, match="canvas_width"):
        pose_keypoint_to_wholebody([{"people": [op_person()]}])


def test_an_empty_pose_is_named():
    with pytest.raises(PoseInteropError, match="body-only"):
        pose_keypoint_to_wholebody([])


def test_a_wrong_point_count_names_the_field():
    bad = {"people": [{"face_keypoints_2d": [0.0] * 30}],
           "canvas_width": W, "canvas_height": H}
    with pytest.raises(PoseInteropError, match="face_keypoints_2d"):
        pose_keypoint_to_wholebody([bad])


def test_an_out_of_range_person_is_named():
    with pytest.raises(PoseInteropError, match="out of range"):
        pose_keypoint_to_wholebody([frame()], person_index=3)


# ── saying when expression cannot work ──────────────────────────────────────

def test_a_body_only_pose_is_reported_as_unable_to_drive_expression():
    kp, canvas = pose_keypoint_to_wholebody([frame(face=False, hands=False)])
    assert not has_face(kp)
    text = describe_source(kp, canvas)
    assert "NO face landmarks" in text
    assert "SDPose" in text or "DWPose" in text


def test_a_wholebody_pose_reports_face_landmarks():
    kp, canvas = pose_keypoint_to_wholebody([frame()])
    assert has_face(kp)
    assert "NO face landmarks" not in describe_source(kp, canvas)
