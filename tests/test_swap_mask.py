"""The swap's region mask, and the one place it must disagree with the control.

Mask says what may CHANGE. Control says what DRIVES. For the jaw they are
opposites, and that opposition is the whole mechanism:

    jaw NOT controlled -> the dupe's skull shape is never imposed
    jaw IS masked      -> the region can be regenerated, so the reference
                          actor's jaw can appear there

Leave the jaw out of both and you keep the dupe's original jaw pixels - the
same failure by a different route. Put it in both and you are back to the
dupe's skull.

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
    a = np.zeros((frames, N_WHOLEBODY, 3), dtype=np.float32)
    n = FACE[1] - FACE[0]
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    a[:, FACE[0]:FACE[1], 0] = cx + r * np.cos(ang)
    a[:, FACE[0]:FACE[1], 1] = cy + r * np.sin(ang)
    a[:, FACE[0]:FACE[1], 2] = 1.0
    return a


# ── the opposition ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("scope", ["face", "head", "person"])
def test_the_jaw_is_masked_but_not_controlled(scope):
    """THE mechanism. Both halves asserted together, because either one alone
    is a different bug."""
    assert "jaw" in mask_groups(scope), (
        f"{scope}: the jaw is not masked, so the actor's jaw can never be "
        "generated - the dupe's jaw pixels survive untouched")
    assert "jaw" not in scope_groups(scope), (
        f"{scope}: the jaw is controlled, so the dupe's skull shape is imposed")


def test_no_scope_both_masks_and_controls_the_jaw():
    for scope in SWAP_SCOPES:
        assert not ("jaw" in mask_groups(scope) and "jaw" in scope_groups(scope))


def test_no_control_edge_touches_the_jaw_even_though_it_is_masked():
    """The mask growing to include the jaw must not quietly pull jaw edges
    into the control alongside it."""
    jaw = set(range(*GROUPS["jaw"]))
    for scope in SWAP_SCOPES:
        for (a, b), _ in scope_edges(scope):
            assert a not in jaw and b not in jaw, scope


def test_lips_scope_masks_only_the_mouth():
    """Lip-sync only: the jaw must NOT be masked, or the chin changes shape
    on every syllable."""
    assert mask_groups("lips") == ("mouth",)
    assert "jaw" not in mask_groups("lips")


def test_body_scope_does_not_mask_the_face():
    assert "jaw" not in mask_groups("body")
    assert "mouth" not in mask_groups("body")


def test_the_report_explains_both_halves():
    text = describe("face")
    assert "NOT in the control" in text
    assert "IS inside the mask" in text


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
    a = np.zeros((1, N_WHOLEBODY, 3), dtype=np.float32)
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
