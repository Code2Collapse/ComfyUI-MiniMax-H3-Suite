"""mmx_utils/performance_metrics.py - performance-preservation scores (C2C L8.25, owner A12 s1.3). CPU-only."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

PACK = Path(__file__).resolve().parents[1]
if str(PACK) not in sys.path:
    sys.path.insert(0, str(PACK))

from mmx_utils import performance_metrics as PM  # noqa: E402

W, H = 640, 480


def _face(yaw=0.0, pitch=0.0, roll=0.0, mouth=0.1, eye=0.3, gaze=0.0) -> np.ndarray:
    """A 478 x 3 landmark array (normalised) whose pose points are the face model seen at the given angles."""
    import cv2

    p = np.zeros((478, 3), np.float32)
    rx, ry, rz = (math.radians(a) for a in (pitch, yaw, roll))
    Rx = np.array([[1, 0, 0], [0, math.cos(rx), -math.sin(rx)], [0, math.sin(rx), math.cos(rx)]])
    Ry = np.array([[math.cos(ry), 0, math.sin(ry)], [0, 1, 0], [-math.sin(ry), 0, math.cos(ry)]])
    Rz = np.array([[math.cos(rz), -math.sin(rz), 0], [math.sin(rz), math.cos(rz), 0], [0, 0, 1]])
    rvec, _ = cv2.Rodrigues(Rz @ Ry @ Rx)
    cam = np.array([[W, 0, W / 2], [0, W, H / 2], [0, 0, 1]], dtype=np.float64)
    img, _ = cv2.projectPoints(PM.POSE_3D, rvec, np.array([0, 0, 600.0]), cam, np.zeros(4))
    for k, i in enumerate(PM.POSE_IDX):
        p[i, :2] = img[k, 0] / [W, H]
    cx, cy = 0.5, 0.6
    p[78, :2], p[308, :2] = (cx - 0.05, cy), (cx + 0.05, cy)            # mouth width 0.1
    p[13, :2], p[14, :2] = (cx, cy - mouth * 0.05), (cx, cy + mouth * 0.05)
    # eyes hang off the PROJECTED outer corners 33 / 263 (pose points too - overwriting them corrupts the pose)
    for (o, i_, up, lo, iris, dx) in ((33, 133, 159, 145, 468, 0.06), (263, 362, 386, 374, 473, -0.06)):
        p[i_, :2] = p[o, :2] + (dx, 0.0)
        mx, my = (p[o, :2] + p[i_, :2]) / 2.0
        p[up, :2], p[lo, :2] = (mx, my - eye * 0.03), (mx, my + eye * 0.03)
        p[iris, :2] = (mx + gaze * 0.03, my)
    return p


def test_head_pose_recovers_known_angles():
    # INVARIANT: the pose model + solvePnP returns the angles the landmarks were rendered at (within 2 degrees).
    for yaw, pitch, roll in ((0, 0, 0), (20, 0, 0), (0, 15, 0), (0, 0, 25), (-15, 10, -10)):
        y, p, r = PM.head_pose(_face(yaw, pitch, roll), W, H)
        assert abs(y - yaw) < 2 and abs(p - pitch) < 2 and abs(r - roll) < 2, (yaw, pitch, roll, y, p, r)


def test_mouth_eye_gaze_signals():
    # INVARIANT: openness ratios and gaze come out as constructed; a missing face is NaN, never a number.
    s = PM.face_signals([_face(mouth=0.5, eye=0.2, gaze=0.5), None], W, H)
    assert s["mouth_open"][0] == pytest.approx(0.5, abs=1e-3)
    assert s["eye_open_l"][0] == pytest.approx(0.2, abs=1e-3) and s["eye_open_r"][0] == pytest.approx(0.2, abs=1e-3)
    assert s["gaze_x"][0] == pytest.approx(0.5, abs=1e-3)
    assert all(np.isnan(s[k][1]) for k in PM.SIGNALS)


def _talk(n=40, shift=0, scale=1.0):
    mouth = [0.1 + scale * 0.5 * max(0.0, math.sin((t + shift) / 3.0)) for t in range(n)]
    return [_face(yaw=10 * math.sin((t + shift) / 7.0), mouth=m) for t, m in enumerate(mouth)]


def test_identical_performance_scores_perfectly():
    # INVARIANT: output == source -> r = 1, MAE = 0, mouth event F1 = 1, not frozen.
    src = PM.face_signals(_talk(), W, H)
    r = PM.performance_scores(src, PM.face_signals(_talk(), W, H))
    assert r["signals"]["mouth_open"]["pearson_r"] == pytest.approx(1.0)
    assert r["signals"]["mouth_open"]["mae"] == pytest.approx(0.0, abs=1e-9)
    assert r["mouth_events"]["f1_tol1"] == pytest.approx(1.0)
    assert r["frozen"] is False


def test_shifted_timing_and_frozen_mouth_are_caught():
    # INVARIANT: lips 4 frames late -> lower r and F1; a closed-mouth output -> frozen.
    src = PM.face_signals(_talk(), W, H)
    late = PM.performance_scores(src, PM.face_signals(_talk(shift=-4), W, H))
    assert late["signals"]["mouth_open"]["pearson_r"] < 0.8 and late["mouth_events"]["f1_tol1"] < 1.0
    frozen = PM.performance_scores(src, PM.face_signals(_talk(scale=0.05), W, H))
    assert frozen["frozen"] is True and frozen["frozen_expression_ratio"] < 0.5


def test_frames_without_a_face_are_excluded_not_imputed():
    # INVARIANT: missing faces are counted and left out of every statistic.
    out = _talk()
    out[5] = out[6] = None
    r = PM.performance_scores(PM.face_signals(_talk(), W, H), PM.face_signals(out, W, H))
    assert r["frames_missing_face"] == 2 and r["signals"]["mouth_open"]["measured_on"] == 38
    assert r["signals"]["mouth_open"]["mae"] == pytest.approx(0.0, abs=1e-9)


def _clip(t=6, h=64, w=64, seed=0):
    rng = np.random.default_rng(seed)
    base = rng.random((h, w, 3)).astype(np.float32) * 0.6 + 0.2
    plate = np.stack([np.roll(base, i, axis=1) for i in range(t)])
    mask = np.zeros((t, h, w), np.float32)
    mask[:, 20:44, 20:44] = 1.0
    return plate, mask


def test_exact_composite_is_perfect_outside_the_mask():
    # INVARIANT: output == plate outside the mask (new content inside) -> infinite PSNR, zero edge leakage.
    plate, mask = _clip()
    out = plate.copy()
    out[mask > 0.5] = 0.5
    r = PM.pixel_scores(plate, out, mask)
    assert math.isinf(r["background"]["psnr_db"]) and r["background"]["max_abs"] == 0.0
    assert r["edge_leakage"]["band_2_12px"] == 0.0


def test_leak_past_the_mask_edge_is_measured():
    # INVARIANT: a change in the band just outside the mask shows in band_2_12px, not in the far band.
    plate, mask = _clip()
    out = plate.copy()
    out[:, 16:48, 16:48] += 0.2                                   # 4 px past the mask on every side
    r = PM.pixel_scores(plate, np.clip(out, 0, 1), mask)
    assert r["edge_leakage"]["band_2_12px"] > 0.02 and r["edge_leakage"]["band_12_24px"] < 0.01


def test_flicker_inside_the_mask_is_excess_over_the_plate():
    # INVARIANT: alternating brightness inside the mask raises excess flicker; a steady fill does not.
    plate, mask = _clip()
    steady = plate.copy()
    steady[mask > 0.5] = 0.5
    flick = steady.copy()
    flick[1::2][mask[1::2] > 0.5] = 0.8
    assert PM.pixel_scores(plate, flick, mask)["flicker"]["excess"] > PM.pixel_scores(plate, steady, mask)["flicker"]["excess"] + 0.1


def test_colour_drift_sees_a_slow_tint_inside_the_mask():
    # INVARIANT: a tint that grows over time inside the mask raises the trajectory excess above a constant tint's.
    plate, mask = _clip(t=8)
    const = plate.copy()
    drift = plate.copy()
    for i in range(8):
        const[i][mask[i] > 0.5] = [0.6, 0.4, 0.4]
        drift[i][mask[i] > 0.5] = [0.4 + 0.05 * i, 0.4, 0.4]
    a = PM.pixel_scores(plate, const, mask)["colour"]["inside_trajectory_excess"]
    b = PM.pixel_scores(plate, drift, mask)["colour"]["inside_trajectory_excess"]
    assert b > a + 0.01
