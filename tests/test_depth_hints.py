import torch

from mmx_nodes.depth_hints import MiniMaxH3_TemporalDepth
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


def test_depth_hints_passthrough_deterministic():
    images = torch.rand(2, 64, 64, 3)
    out1 = MiniMaxH3_TemporalDepth.execute(images, _transform(), "passthrough", None, 17, False)
    out2 = MiniMaxH3_TemporalDepth.execute(images, _transform(), "passthrough", None, 17, False)
    assert torch.equal(out1[0], out2[0])


def test_depth_hints_dvd_raises():
    import pytest

    images = torch.rand(1, 32, 32, 3)
    with pytest.raises(ValueError, match="third_party/DVD"):
        MiniMaxH3_TemporalDepth.execute(images, _transform(1), "dvd", None, 17, False)
