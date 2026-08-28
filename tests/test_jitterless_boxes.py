import numpy as np

from mmx_utils.jitterless_boxes import build_jitterless_boxes, lock_anchor_size, smooth_centers


def test_lock_anchor_size_median():
    sizes = np.array([40.0, 44.0, 42.0])
    side = lock_anchor_size(sizes, safety_margin=1.0, max_side=512.0)
    assert side == 42.0


def test_build_jitterless_boxes_returns_per_frame():
    n = 5
    centers = np.stack([np.linspace(100, 200, n), np.full(n, 120.0)], axis=1)
    boxes, _centers = build_jitterless_boxes(
        target_centers=centers,
        anchor_size=80.0,
        crop_w=80.0,
        crop_h=80.0,
        W=640,
        H=480,
    )
    assert len(boxes) == n
    assert all(len(b) == 4 for b in boxes)


def test_smooth_centers_preserves_length():
    c = np.array([[0.0, 0.0], [10.0, 5.0], [20.0, 10.0]], dtype=np.float32)
    out = smooth_centers(c, "one_euro", zero_phase=False)
    assert out.shape == c.shape
