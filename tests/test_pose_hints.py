import numpy as np
import torch

from mmx_nodes.pose_hints import MiniMaxH3_StrongestPose
from mmx_utils.transform_types import H3Transform


def _transform(n=2):
    boxes = tuple((0.0, 0.0, 64.0, 64.0) for _ in range(n))
    return H3Transform(
        boxes=boxes,
        canvas=(32, 32),
        src_size=(128, 128),
        frames=n,
        weights=tuple(1.0 for _ in range(n)),
        detected=tuple(True for _ in range(n)),
        subject_rect=None,
        crop_factor=3.0,
        planner_mode="tv_lp",
    )


def test_pose_hints_passthrough_warp():
    plate = torch.rand(2, 64, 64, 3)
    hints_plate = torch.ones(2, 64, 64, 3) * 0.25
    out = MiniMaxH3_StrongestPose.execute(
        plate,
        _transform(2),
        "passthrough",
        hints_plate,
        0.3,
        True,
        3,
    )
    hints, _kjson, cov, report = out[0], out[1], out[2], out[3]
    assert hints.shape == (2, 32, 32, 3)
    assert cov == 1.0
    assert "passthrough" in report
