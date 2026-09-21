"""The swap's region mask, and the one place it must disagree with the control.

Mask says what may CHANGE. Control says what DRIVES. For the jaw they now
BOTH apply, and the two pull against each other on purpose:

    jaw DRIVEN -> the dupe's head pose and chin drop transfer
    jaw MASKED -> the region can still move toward the reference actor's skull

ControlNet strength decides who wins. `lips` is the exception: it drives the
jaw so the chin can drop, but masks only the mouth, because a lip-sync must
never reshape the chin.

Both are derived from ONE landmark array so they cannot drift apart frame to
frame, which is what keeps the boundary offset at zero without per-shot tuning.

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

from mmx_utils.swap_control import scope_edges  # noqa: E402
from mmx_utils.swap_mask import build_swap_mask, describe_mask  # noqa: E402
from mmx_utils.swap_regions import (  # noqa: E402
    FACE,
    GROUPS,
    MASK_GROUPS,
    N_EXTENDED,
    N_WHOLEBODY,
    SWAP_SCOPES,
    SwapRegionError,
    describe,
    mask_groups,
    scope_groups,
)

_F = FACE[0]
W = H = 256


def face_kps(frames=1, cx=128.0, cy=128.0, r=50.0):
    """A ring of face landmarks, so the hull is a predictable disc."""
    a = np.zeros((frames, N_EXTENDED, 3), dtype=np.float32)
    n = FACE[1] - FACE[0]
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    a[:, FACE[0]:FACE[1], 0] = cx + r * np.cos(ang)
    a[:, FACE[0]:FACE[1], 1] = cy + r * np.sin(ang)
    a[:, FACE[0]:FACE[1], 2] = 1.0
    return a


# ── the opposition ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("scope", ["face", "head", "person"])
def test_the_jaw_is_both_masked_and_driven(scope):
    """Both halves asserted together, because either one alone is a different
    result: driven-only imposes the dupe's skull, masked-only loses head pose
    and chin drop."""
    assert "jaw" in mask_groups(scope), (
        f"{scope}: the jaw is not masked, so the actor's jaw can never be "
        "generated - the dupe's jaw pixels survive untouched")
    assert "jaw" in scope_groups(scope), (
        f"{scope}: the jaw is not driven, so head pose and chin drop are lost")


def test_lips_drives_the_jaw_but_does_not_mask_it():
    """The one scope where they must differ. The chin has to DROP for the
    mouth to open, but a lip-sync that reshapes the chin is a broken shot."""
    assert "jaw" in scope_groups("lips"), "the chin cannot drop"
    assert mask_groups("lips") == ("mouth",), "a lip-sync must not reshape the chin"


def test_drive_jaw_off_removes_it_from_the_control_but_not_the_mask():
    """The identity lever leaves the region free while removing the contour."""
    for scope in ("face", "head", "person"):
        assert "jaw" not in scope_groups(scope, drive_jaw=False)
        assert "jaw" in mask_groups(scope), (
            "turning the control off must not also stop the region changing")


def test_body_scope_does_not_touch_the_face_either_way():
    assert "jaw" not in mask_groups("body")
    assert "jaw" not in scope_groups("body")


def test_the_report_explains_the_tension():
    """A user who sees the head drifting toward the dupe needs to be told the
    knob, not left to guess."""
    text = describe("face")
    assert "driven and masked" in text.lower()
    assert "strength" in text.lower()


def test_the_report_explains_the_other_setting_too():
    text = describe("face", drive_jaw=False)
    assert "masked but not driven" in text.lower()


def test_the_report_mentions_gaze_when_pupils_are_driven():
    assert "direction" in describe("face").lower()


# ── the mask itself ─────────────────────────────────────────────────────────

def test_a_face_mask_covers_the_landmarks_it_was_built_from():
    m = build_swap_mask(face_kps(), "face", (W, H), pad=0.0)
    assert m.shape == (1, H, W)
    assert m[0, 128, 128] > 0.5, "the centre of the hull is not filled"


def test_the_mask_does_not_cover_the_whole_frame():
    m = build_swap_mask(face_kps(), "face", (W, H))
    assert m[0].mean() < 0.5, "a face mask should not be most of the frame"
    assert m[0, 2, 2] == 0.0, "the far corner should be outside the hull"


def test_padding_grows_the_mask_without_moving_its_centre():
    """Scaling about the centroid, not offsetting edges - the same rule as
    token-space mask growth. A mask that drifts when you pad it puts the
    boundary somewhere new relative to the face."""
    def centroid(m):
        w = m[0]
        ys = (w.sum(1) * np.arange(w.shape[0])).sum() / w.sum()
        xs = (w.sum(0) * np.arange(w.shape[1])).sum() / w.sum()
        return ys, xs

    tight = build_swap_mask(face_kps(), "face", (W, H), pad=0.0)
    loose = build_swap_mask(face_kps(), "face", (W, H), pad=0.4)
    assert loose[0].sum() > tight[0].sum(), "padding did not grow it"
    ty, tx = centroid(tight)
    ly, lx = centroid(loose)
    assert abs(ty - ly) < 1.0 and abs(tx - lx) < 1.0, (
        f"padding moved the centre by ({ly - ty:.2f}, {lx - tx:.2f}) px")


def test_head_pads_further_than_face_because_hair_has_no_landmarks():
    kp = face_kps()
    face = build_swap_mask(kp, "face", (W, H))
    head = build_swap_mask(kp, "head", (W, H))
    assert head[0].sum() > face[0].sum()


def test_lips_masks_a_far_smaller_area_than_face():
    a = np.zeros((1, N_EXTENDED, 3), dtype=np.float32)
    a[..., 2] = 1.0
    ang = np.linspace(0, 2 * np.pi, 68, endpoint=False)
    a[0, FACE[0]:FACE[1], 0] = 128 + 50 * np.cos(ang)
    a[0, FACE[0]:FACE[1], 1] = 128 + 50 * np.sin(ang)
    lips = build_swap_mask(a, "lips", (W, H))
    face = build_swap_mask(a, "face", (W, H))
    assert lips[0].sum() < face[0].sum()


def test_a_frame_with_no_detection_is_empty_not_guessed():
    """A hole is honest; an interpolated mask silently regenerates the wrong
    region for however long the detection was out."""
    kp = face_kps(frames=2)
    kp[1, :, 2] = 0.0
    m = build_swap_mask(kp, "face", (W, H))
    assert m[0].max() > 0
    assert m[1].max() == 0.0


def test_the_report_names_frames_with_no_mask():
    kp = face_kps(frames=3)
    kp[2, :, 2] = 0.0
    text = describe_mask(build_swap_mask(kp, "face", (W, H)), "face")
    assert "1 frame(s) have NO mask" in text


def test_feathering_softens_the_edge_without_emptying_it():
    kp = face_kps()
    hard = build_swap_mask(kp, "face", (W, H), feather=0)
    soft = build_swap_mask(kp, "face", (W, H), feather=5)
    assert soft[0].max() > 0.5
    mid = (soft[0] > 0.05) & (soft[0] < 0.95)
    assert mid.sum() > 0, "feathering produced no partial values"
    assert hard[0][(hard[0] > 0.05) & (hard[0] < 0.95)].size == 0


def test_the_mask_stays_in_range():
    m = build_swap_mask(face_kps(), "head", (W, H), feather=4)
    assert m.min() >= 0.0 and m.max() <= 1.0


def test_landmarks_off_canvas_do_not_crash():
    kp = face_kps()
    kp[0, :, 0] += 5000
    m = build_swap_mask(kp, "face", (W, H))
    assert m[0].max() == 0.0


def test_a_bad_canvas_is_named():
    with pytest.raises(SwapRegionError, match="positive"):
        build_swap_mask(face_kps(), "face", (0, H))


def test_the_builder_does_not_mutate_its_keypoints():
    kp = face_kps()
    before = kp.copy()
    build_swap_mask(kp, "face", (W, H))
    assert np.array_equal(kp, before)


def test_every_scope_has_a_pad_value():
    from mmx_utils.swap_regions import MASK_PAD

    assert set(MASK_PAD) == set(MASK_GROUPS) == set(SWAP_SCOPES)


# ── the node emits both, from one landmark array ────────────────────────────

def test_the_node_returns_control_mask_and_report():
    from mmx_nodes.swap_control import MiniMaxH3_SwapControl

    names = [o.display_name for o in MiniMaxH3_SwapControl.define_schema().outputs]
    assert names == ["control_video", "region_mask", "report"]


def test_the_node_builds_both_from_the_same_pose():
    """One pose in, a control and a mask out that are consistent by
    construction - not two nodes that can be wired to different sources."""
    from mmx_nodes.swap_control import MiniMaxH3_SwapControl

    ang = np.linspace(0, 2 * np.pi, 68, endpoint=False)
    person = {
        "face_keypoints_2d": [v for i in range(70) for v in (
            (0.5 + 0.15 * np.cos(ang[min(i, 67)])),
            (0.5 + 0.15 * np.sin(ang[min(i, 67)])), 1.0)],
    }
    pose = [{"people": [person], "canvas_width": 256, "canvas_height": 256}]

    control, mask, report = MiniMaxH3_SwapControl.execute(
        pose_keypoint=pose, swap_scope="face", width=256, height=256,
        confidence_gate=0.3, line_width=4, face_line_width=0)

    assert control.shape == (1, 256, 256, 3)
    assert mask.shape == (1, 256, 256)
    assert float(mask.max()) > 0.5, "no mask was produced"
    assert float(control.max()) > 0.0, "no control was drawn"
    assert "jaw" in report.lower()
