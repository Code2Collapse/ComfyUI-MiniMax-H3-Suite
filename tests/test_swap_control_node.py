"""The three swap cases, exercised through the node itself.

The pieces are unit-tested elsewhere; this file exists because the cases the
user actually builds are combinations, and a combination can be wrong while
every part of it is right:

  1. LIP-SYNC ONLY      scope=lips, no audio lock
                        -> only the mouth region regenerates, and the dupe's
                           own lips drive it

  2. SWAP + LIP-SYNC    scope=face, audio locked
                        -> the face region regenerates. BOTH the control's
                           mouth and the locked audio can shape the lips. Same
                           take: they agree. A dub: they fight and the control
                           wins, which is what drive_mouth=False is for.

  3. SWAP, NO AUDIO     scope=face, no audio lock
                        -> the lips still move, because the control draws
                           them. This is the case that breaks if anyone
                           "tidies" the mouth out of the default scope.

CPU-only, numpy only, no detector, no weights, no model.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PACK = Path(__file__).resolve().parents[1]
if str(PACK) not in sys.path:
    sys.path.insert(0, str(PACK))

from mmx_nodes.swap_control import MiniMaxH3_SwapControl  # noqa: E402
from mmx_utils.swap_regions import GROUPS  # noqa: E402

W, H = 256, 256


def op_frame(w=W, h=H):
    """One OpenPose person with a full 70-point face, laid out so the mouth
    sits in a corner nothing else occupies - see test_swap_control.py."""
    rng = np.random.default_rng(3)

    def block(n, lo, hi):
        out = []
        for _ in range(n):
            out += [float(rng.uniform(lo, hi)) / w,
                    float(rng.uniform(lo, hi)) / h, 1.0]
        return out

    face = []
    for i in range(70):
        if 48 <= i < 68:                       # dlib mouth -> the corner
            x, y = rng.uniform(14, 52), rng.uniform(14, 52)
        else:
            x, y = rng.uniform(92, w - 20), rng.uniform(92, h - 20)
        face += [float(x) / w, float(y) / h, 1.0]

    return {
        "people": [{
            "pose_keypoints_2d": block(18, 92, w - 20),
            "face_keypoints_2d": face,
            "hand_left_keypoints_2d": block(21, 92, w - 20),
            "hand_right_keypoints_2d": block(21, 92, w - 20),
        }],
        "canvas_width": w,
        "canvas_height": h,
    }


def run(**kw):
    kw.setdefault("pose_keypoint", [op_frame()])
    kw.setdefault("swap_scope", "face")
    kw.setdefault("width", W)
    kw.setdefault("height", H)
    kw.setdefault("confidence_gate", 0.3)
    kw.setdefault("line_width", 2)
    kw.setdefault("face_line_width", 0)
    out = MiniMaxH3_SwapControl.execute(**kw)
    control, mask, report = out[0], out[1], out[2]
    return np.asarray(control), np.asarray(mask), str(report)


CORNER = (slice(0, 1), slice(0, 62), slice(0, 62))


# ── case 1: lip-sync only ───────────────────────────────────────────────────

def test_lips_scope_masks_only_the_mouth():
    """A lip-sync must not be free to reshape the chin, whatever drives it."""
    _, mask, _ = run(swap_scope="lips")
    assert mask[CORNER].max() > 0.5, "the mouth is not in the mask"
    # the rest of the face sits in the far corner of this fixture
    assert mask[0, 150:, 150:].max() == 0.0, "lips scope masked the whole face"


def test_lips_scope_still_drives_the_jaw_so_the_chin_can_drop():
    """Driven and masked are different sets. The jaw drives (the chin has to
    drop for the mouth to open) but is not masked (it must not be reshaped)."""
    control, mask, _ = run(swap_scope="lips")
    assert control[0, 150:, 150:].max() > 0.0, "no jaw ink outside the mouth"
    assert mask[0, 150:, 150:].max() == 0.0, "the jaw was masked by a lip-sync"


# ── case 3: a swap with no audio at all ─────────────────────────────────────

def test_a_face_swap_with_no_audio_still_moves_the_lips():
    """The user's stated requirement: expression and lip movement WITHOUT
    audio. They come from the control, so the mouth must be drawn."""
    control, _, _ = run(swap_scope="face")
    assert control[CORNER].max() > 0.05, (
        "no lip ink: a silent swap would produce a still mouth")


# ── case 2: the lever for a dub ─────────────────────────────────────────────

def test_drive_mouth_off_stops_drawing_the_lips_but_keeps_masking_them():
    control, mask, _ = run(swap_scope="face", drive_mouth=False)
    assert control[CORNER].max() == 0.0, "lip ink survived drive_mouth=False"
    assert mask[CORNER].max() > 0.5, (
        "the mouth left the mask too - a locked audio track could not move "
        "the lips at all")


def test_the_report_says_who_is_driving_the_mouth():
    _, _, on = run(swap_scope="face")
    _, _, off = run(swap_scope="face", drive_mouth=False)
    assert "masked but NOT driven" not in on
    assert "masked but NOT driven" in off
    assert "audio" in off


# ── the levers do not quietly take each other's groups ──────────────────────

def test_drive_jaw_and_drive_mouth_are_independent_through_the_node():
    control, mask, _ = run(swap_scope="face", drive_jaw=False, drive_mouth=False)
    assert control.max() > 0.05, "brows, nose, eyes and pupils went too"
    assert mask.max() > 0.5, "the mask followed the control off a cliff"


@pytest.mark.parametrize("scope", ["lips", "face", "head", "person"])
def test_every_scope_produces_a_control_and_a_matching_mask(scope):
    control, mask, report = run(swap_scope=scope)
    assert control.shape[:3] == (1, H, W)
    assert mask.shape == (1, H, W)
    assert report.strip(), "no report"
