import torch

from mmx_nodes.control_hints import MiniMaxH3_ControlHints
from mmx_utils.transform_types import H3Transform


def _transform():
    return H3Transform(
        boxes=((10.0, 10.0, 40.0, 40.0),),
        canvas=(64, 64),
        src_size=(128, 128),
        frames=1,
        weights=(1.0,),
        detected=(True,),
        subject_rect=((16.0, 16.0, 32.0, 32.0),),
        crop_factor=2.5,
        planner_mode="tv_lp",
    )


def test_control_hints_schema():
    schema = MiniMaxH3_ControlHints.define_schema()
    assert schema.node_id == "MiniMaxH3_ControlHints"


def test_control_hints_canny_execute():
    plate = torch.rand(1, 128, 128, 3)
    out = MiniMaxH3_ControlHints.execute(plate, _transform(), hint_type="canny")
    hints, report = out[0], out[1]
    assert hints.shape == (1, 64, 64, 3)
    assert "no consumer" in MiniMaxH3_ControlHints.define_schema().description.lower()
