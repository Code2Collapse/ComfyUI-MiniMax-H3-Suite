import json

import torch

from mmx_nodes.track_crop import MiniMaxH3_TrackCrop


def test_track_crop_schema():
    assert MiniMaxH3_TrackCrop.define_schema().node_id == "MiniMaxH3_TrackCrop"


def test_track_crop_execute_with_bboxes():
    images = torch.rand(22, 128, 128, 3)
    bboxes = json.dumps([[40, 40, 80, 80]] * 22)
    out = MiniMaxH3_TrackCrop.execute(
        images,
        bboxes_json=bboxes,
        canvas_width=512,
        canvas_height=512,
        planner_mode="tv_lp",
        lock_size=True,
        movement_cost=2.0,
    )
    crops, transform, _preview, report, boxes_json = out[0], out[1], out[2], out[3], out[4]
    assert crops.shape[0] == 22
    assert transform.frames == 22
    assert "tv_lp" in report
    data = json.loads(boxes_json)
    assert data["frames"] == 22
    assert len(data["boxes"]) == 22
    assert data["src_size"] == [128, 128]
