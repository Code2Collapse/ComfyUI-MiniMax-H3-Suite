"""Rendering the driving landmarks into a control video.

Detector-agnostic on purpose: DWPose, SDPose (arXiv 2509.24980) and
ViTPose-wholebody all emit the same COCO-WholeBody 133 array, so swapping to
the stronger detector is a change of source, not of contract. SDPose is the
stronger one - 72.8 AP on COCO-WholeBody against DWPose's ~66, and better out
of domain - which is why these tests pin the LAYOUT rather than a detector.

The jaw IS drawn by default. Those 17 points carry head POSE and the CHIN
DROP that lets the mouth open, not only skull shape, and dropping them to
protect face width also drops those. `drive_jaw=False` removes them for the
cases where identity beats pose, and both paths are pinned below.

CPU-only, numpy only, no detector, no weights.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PACK = Path(__file__).resolve().parents[1]
if str(PACK) not in sys.path:
    sys.path.insert(0, str(PACK))

from mmx_utils.swap_control import (  # noqa: E402
    GROUP_COLOUR,
    GROUP_EDGES,
    render_swap_control,
    scope_edges,
)
from mmx_utils.swap_regions import (  # noqa: E402
    FACE,
    GROUPS,
    N_EXTENDED,
    N_WHOLEBODY,
    SWAP_SCOPES,
    SwapRegionError,
)

_F = FACE[0]


def kps(frames=1, w=256, h=256):
    """Every landmark confident and inside the canvas, spread out enough that
    drawn lines are distinguishable."""
    a = np.zeros((frames, N_EXTENDED, 3), dtype=np.float32)
    rng = np.random.default_rng(0)
    a[..., 0] = rng.uniform(20, w - 20, size=(frames, N_EXTENDED))
    a[..., 1] = rng.uniform(20, h - 20, size=(frames, N_EXTENDED))
    a[..., 2] = 1.0
    return a


# ── the rule, at the edge-table level ───────────────────────────────────────

def test_the_jaw_chain_is_an_open_arc_not_a_loop():
    """Ear to chin to ear. Closing it draws a line straight across the eyes."""
    edges = {(a - _F, b - _F) for a, b in GROUP_EDGES["jaw"]}
    assert (16, 0) not in edges, "the jaw contour was closed into a loop"
    assert len(edges) == 16


@pytest.mark.parametrize("scope", ["face", "head", "person", "lips"])
def test_the_jaw_is_drawn_by_default(scope):
    """Head pose and chin drop live here; without them a profile reads as a
    front-on face and the mouth cannot open far."""
    jaw = set(range(*GROUPS["jaw"]))
    touched = {i for (a, b), _ in scope_edges(scope) for i in (a, b)}
    assert touched & jaw, f"scope {scope!r} drew no jaw edge"


@pytest.mark.parametrize("scope", ["face", "head", "person", "lips"])
def test_drive_jaw_false_removes_every_jaw_edge(scope):
    """The identity lever. When it is off, the dupe's face width has no route
    into the control at all."""
    jaw = set(range(*GROUPS["jaw"]))
    for (a, b), _ in scope_edges(scope, drive_jaw=False):
        assert a not in jaw and b not in jaw, (
            f"scope {scope!r} still draws {(a, b)} with drive_jaw off")


def test_drive_jaw_false_keeps_the_expression_groups():
    """Dropping the contour must not take the performance with it."""
    kept = {i for (a, b), _ in scope_edges("face", drive_jaw=False) for i in (a, b)}
    mouth = set(range(*GROUPS["mouth"]))
    eyes = set(range(*GROUPS["eyes"]))
    assert kept & mouth and kept & eyes


def test_pupils_are_drawn_for_gaze():
    """The only eye-DIRECTION signal. The 68-point eye contours give the lid
    opening and never where the eye is looking."""
    from mmx_utils.swap_regions import PUPILS

    a = np.zeros((1, N_EXTENDED, 3), dtype=np.float32)
    a[0, PUPILS[0]:PUPILS[1], 0] = [80.0, 160.0]
    a[0, PUPILS[0]:PUPILS[1], 1] = 120.0
    a[0, PUPILS[0]:PUPILS[1], 2] = 1.0
    out = render_swap_control(a, "face", (256, 256))
    assert out.max() > 0.0, "pupils were not drawn"
    ys, xs = np.nonzero(out[0].sum(-1) > 0)
    assert 70 <= xs.min() and xs.max() <= 170


def test_a_pose_without_pupils_still_renders():
    """133-point detectors have no pupils; that is a missing bonus, not an
    error."""
    a = np.zeros((1, N_WHOLEBODY, 3), dtype=np.float32)
    a[..., 2] = 1.0
    a[0, :, :2] = 64.0
    render_swap_control(a, "face", (128, 128))


def test_every_drawn_group_has_a_colour():
    """An edge with no colour would render black on black and vanish."""
    for group in GROUP_EDGES:
        assert group in GROUP_COLOUR, f"{group} has edges but no colour"


def test_every_edge_index_is_in_range():
    for group, edges in GROUP_EDGES.items():
        for a, b in edges:
            assert 0 <= a < N_EXTENDED and 0 <= b < N_EXTENDED, (group, a, b)


# ── the face chains are the dlib-68 ones ────────────────────────────────────

def test_the_mouth_is_two_closed_loops():
    """Outer 48-59 and inner 60-67, each closed. An unclosed lip contour leaves
    a gap the model reads as a hole in the mouth."""
    edges = {(a - _F, b - _F) for a, b in GROUP_EDGES["mouth"]}
    assert (59, 48) in edges, "outer lip loop is not closed"
    assert (67, 60) in edges, "inner lip loop is not closed"
    assert len(edges) == 20


def test_the_eyes_are_two_closed_loops():
    edges = {(a - _F, b - _F) for a, b in GROUP_EDGES["eyes"]}
    assert (41, 36) in edges and (47, 42) in edges
    assert len(edges) == 12


def test_the_brows_are_two_open_chains():
    """Brows are arcs, not loops - closing them draws a line across the brow."""
    edges = {(a - _F, b - _F) for a, b in GROUP_EDGES["brows"]}
    assert (21, 17) not in edges and (26, 22) not in edges
    assert len(edges) == 8


def test_the_body_omits_the_coarse_coco_head_points():
    """COCO 0-4 are nose/eyes/ears at body resolution. The 68-point face block
    sits right beside them and is far finer; drawing both puts two different
    noses in the control."""
    touched = {i for e in GROUP_EDGES["body"] for i in e}
    assert not (touched & {0, 1, 2, 3, 4})


# ── rendering ───────────────────────────────────────────────────────────────

def test_lips_scope_with_the_jaw_off_draws_only_the_mouth():
    """A tight, checkable claim: put the mouth in a known box, park everything
    else far away, and assert nothing is drawn outside the box.

    drive_jaw is off here because `lips` DOES drive the jaw by default - the
    chin has to drop for the mouth to open - so the default would legitimately
    draw outside the mouth box.
    """
    a = np.zeros((1, N_EXTENDED, 3), dtype=np.float32)
    a[..., 2] = 1.0
    a[0, :, 0] = 10.0          # everything else parked top-left
    a[0, :, 1] = 10.0
    mouth = slice(_F + 48, _F + 68)
    a[0, mouth, 0] = np.linspace(100, 150, 20)
    a[0, mouth, 1] = np.linspace(180, 200, 20)

    out = render_swap_control(a, "lips", (256, 256), line_width=2, drive_jaw=False)
    ink = out[0].sum(-1) > 0
    ys, xs = np.nonzero(ink)
    assert ys.size > 0, "nothing was drawn"
    assert xs.min() >= 90 and xs.max() <= 160
    assert ys.min() >= 170 and ys.max() <= 210


def test_lips_scope_draws_the_jaw_by_default():
    """Because the chin drops when the mouth opens."""
    a = np.zeros((1, N_EXTENDED, 3), dtype=np.float32)
    a[..., 2] = 1.0
    a[0, :, :2] = 128.0
    jaw = slice(_F + 0, _F + 17)
    a[0, jaw, 0] = np.linspace(40, 200, 17)
    a[0, jaw, 1] = 220.0
    out = render_swap_control(a, "lips", (256, 256), line_width=2)
    assert (out[0, 215:225].sum(-1) > 0).any(), "no jaw line was drawn"


def test_face_scope_draws_more_than_lips():
    a = kps()
    lips = render_swap_control(a, "lips", (256, 256))
    face = render_swap_control(a, "face", (256, 256))
    assert (face.sum(-1) > 0).sum() > (lips.sum(-1) > 0).sum()


def test_low_confidence_landmarks_are_not_drawn():
    a = kps()
    a[0, :, 2] = 0.0
    out = render_swap_control(a, "person", (128, 128))
    assert out.max() == 0.0, "a landmark below the gate was still drawn"


def test_the_output_shape_and_range():
    out = render_swap_control(kps(frames=3), "face", (96, 64))
    assert out.shape == (3, 64, 96, 3)
    assert out.dtype == np.float32
    assert 0.0 <= out.min() and out.max() <= 1.0


def test_face_lines_are_thinner_than_body_lines_by_default():
    """A 4px mouth line on a small frame closes the lip gap, and the model can
    then no longer tell an open mouth from a shut one."""
    a = np.zeros((1, N_EXTENDED, 3), dtype=np.float32)
    a[..., 2] = 1.0
    a[0, :, :2] = 8.0
    mouth = slice(_F + 48, _F + 68)
    a[0, mouth, 0] = np.linspace(40, 200, 20)
    a[0, mouth, 1] = 128.0
    thin = (render_swap_control(a, "lips", (256, 256), line_width=8).sum(-1) > 0).sum()
    thick = (render_swap_control(a, "lips", (256, 256), line_width=8,
                                 face_line_width=8).sum(-1) > 0).sum()
    assert thin < thick


def test_a_non_133_pose_is_refused_with_the_reason():
    with pytest.raises(SwapRegionError, match="no face landmarks"):
        render_swap_control(np.zeros((1, 17, 3)), "face", (64, 64))


def test_a_bad_canvas_is_named():
    with pytest.raises(SwapRegionError, match="positive"):
        render_swap_control(kps(), "face", (0, 64))


def test_the_renderer_does_not_mutate_its_keypoints():
    a = kps()
    before = a.copy()
    render_swap_control(a, "person", (64, 64))
    assert np.array_equal(a, before)


def test_coordinates_off_canvas_do_not_crash():
    """Detectors return points outside the frame on partial occlusion."""
    a = kps()
    a[0, :, 0] -= 5000
    render_swap_control(a, "person", (64, 64))       # must simply draw nothing
