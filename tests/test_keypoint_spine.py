import numpy as np
import pytest

from mmx_utils.keypoint_spine import (
    coverage_fraction,
    fill_keypoint_gaps,
    interp_gaps,
    plate_point_to_crop,
    smooth_keypoints_one_euro,
    warp_keypoints_to_crop,
)


def test_plate_point_to_crop_matches_affine_spine():
    box = (100.0, 50.0, 200.0, 100.0)
    cw, ch = 384, 384
    cx, cy = plate_point_to_crop(200.0, 100.0, box, cw, ch)
    assert cx == pytest.approx(cw * 0.5, abs=1e-4)
    assert cy == pytest.approx(ch * 0.5, abs=1e-4)


def test_interp_gaps_fills_dropped_frame():
    vals = np.array([1.0, 0.0, 3.0])
    valid = np.array([True, False, True])
    out = interp_gaps(vals, valid)
    assert out[1] == pytest.approx(2.0)


def test_gap_fill_does_not_zero_interpolated_frame():
    kps = np.zeros((3, 20, 3), dtype=np.float32)
    kps[0, 0, :] = [10, 10, 0.9]
    kps[1, 0, :] = [0, 0, 0.0]
    kps[2, 0, :] = [30, 30, 0.9]
    filled = fill_keypoint_gaps(kps, confidence_gate=0.3, max_gap=3)
    assert filled[1, 0, 0] > 0.0
    assert filled[1, 0, 1] > 0.0


def test_one_euro_reduces_jitter():
    t = np.arange(20, dtype=np.float32)
    noisy = np.stack([t + np.random.RandomState(0).normal(scale=2.0, size=20), t], axis=1)
    kps = np.zeros((20, 1, 3), dtype=np.float32)
    kps[:, 0, 0] = noisy[:, 0]
    kps[:, 0, 1] = noisy[:, 1]
    kps[:, 0, 2] = 1.0
    sm = smooth_keypoints_one_euro(kps, enabled=True)
    assert float(np.std(np.diff(sm[:, 0, 0]))) < float(np.std(np.diff(kps[:, 0, 0])))


def test_coverage_fraction_arithmetic():
    kps = np.zeros((4, 20, 3), dtype=np.float32)
    kps[:, :17, 2] = 1.0
    kps[2, :10, 2] = 0.0
    cov = coverage_fraction(kps, confidence_gate=0.5, min_keypoints=17)
    assert cov == pytest.approx(0.75)


def test_warp_keypoints_to_crop():
    kps = np.zeros((1, 1, 3), dtype=np.float32)
    kps[0, 0] = [150.0, 75.0, 1.0]
    box = (100.0, 50.0, 200.0, 100.0)
    out = warp_keypoints_to_crop(kps, [box], (200, 100))
    assert out[0, 0, 0] == pytest.approx(50.0)
    assert out[0, 0, 1] == pytest.approx(25.0)
