import numpy as np
import pytest

from mmx_utils.crop_planner_tv import plan_tracked_crop_tv
from mmx_utils.h3_constants import align_frame_count, video_latent_t
from mmx_utils.h3_grid import assert_groups_match_latent_t, h3_pixel_frame_groups
from mmx_utils.jitterless_boxes import build_jitterless_boxes


@pytest.mark.parametrize("fc", [5, 22, 39, 56, 73])
def test_h3_grid_matches_core(fc):
    assert_groups_match_latent_t(fc)
    assert video_latent_t(fc) == len(h3_pixel_frame_groups(fc))


def test_align_frame_count():
    assert align_frame_count(6) == 22
    assert align_frame_count(22) == 22


def test_jitterless_lock_size():
    centers = np.array([[50, 50], [52, 51], [48, 49]], dtype=np.float32)
    boxes, _ = build_jitterless_boxes(
        target_centers=centers,
        anchor_size=128,
        crop_w=128,
        crop_h=128,
        W=512,
        H=512,
    )
    widths = [b[2] for b in boxes]
    heights = [b[3] for b in boxes]
    assert len(set(round(w) for w in widths)) == 1
    assert len(set(round(h) for h in heights)) == 1


def test_tv_lp_pan_hold():
    m = 60
    bx0 = np.zeros(m)
    bx1 = np.full(m, 40.0)
    by0 = np.zeros(m)
    by1 = np.full(m, 40.0)
    detected = np.ones(m, dtype=bool)
    drift = 2.0
    for i in range(30):
        bx0[i] += i * drift
        bx1[i] += i * drift
    for i in range(30, m):
        bx0[i] = bx0[29]
        bx1[i] = bx1[29]
    res = plan_tracked_crop_tv(
        bx0=bx0,
        bx1=bx1,
        by0=by0,
        by1=by1,
        detected=detected,
        crop_w=80,
        crop_h=80,
        img_w=400,
        img_h=400,
        movement_cost=2.0,
    )
    assert res.success
    hold_slice = slice(35, 55)
    assert float(np.std(res.x[hold_slice])) < 5.0
    assert float(np.std(res.y[hold_slice])) < 5.0
