import numpy as np
import pytest
import torch

from mmx_utils.detection import extents_from_mask, merge_detection_sources, parse_bboxes_json


def test_parse_bboxes_json_null_slots():
    bx0, bx1, by0, by1, det = parse_bboxes_json("[[10,20,30,40], null]", 2)
    assert det[0]
    assert not det[1]


def test_extents_from_mask():
    m = torch.zeros(32, 32)
    m[10:20, 12:22] = 1.0
    ext = extents_from_mask(m)
    assert ext == (12.0, 10.0, 21.0, 19.0)


def test_merge_requires_detection():
    with pytest.raises(ValueError, match="planner"):
        merge_detection_sources(frame_count=2, bboxes_json="[]", subject_mask=None)


def test_merge_from_bboxes():
    bx0, bx1, by0, by1, det = merge_detection_sources(
        frame_count=1,
        bboxes_json="[[0,0,10,10]]",
        subject_mask=None,
    )
    assert det[0]
    assert bx1[0] == 10.0
