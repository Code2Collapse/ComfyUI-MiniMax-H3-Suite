import torch

from mmx_utils.puppeteer import render_motion_hints


def test_puppeteer_deterministic():
    pose = torch.rand(2, 32, 32, 3)
    ref = torch.rand(2, 32, 32, 3)
    h1, t1 = render_motion_hints(pose, ref, rig_type="human_to_creature")
    h2, t2 = render_motion_hints(pose, ref, rig_type="human_to_creature")
    assert torch.equal(h1, h2)
    assert t1 == t2
    assert "Union: blocked" in t1


def test_motion_text_catalog_61():
    pose = torch.zeros(1, 16, 16, 3)
    ref = torch.ones(1, 16, 16, 3) * 0.5
    _hints, text = render_motion_hints(pose, ref, rig_type="stick_figure", keypoints_json='{"frames": []}')
    assert "catalog #61" in text
    assert "stick_figure" in text


def test_keypoints_json_renders_skeleton():
    pose = torch.zeros(1, 64, 64, 3)
    ref = torch.ones(1, 64, 64, 3) * 0.2
    kps = {
        "frames": [
            {
                "keypoints": [
                    {"x": 32, "y": 10, "c": 1.0},
                    {"x": 32, "y": 30, "c": 1.0},
                    {"x": 20, "y": 30, "c": 1.0},
                    {"x": 44, "y": 30, "c": 1.0},
                    {"x": 32, "y": 50, "c": 1.0},
                    {"x": 20, "y": 50, "c": 1.0},
                    {"x": 44, "y": 50, "c": 1.0},
                    {"x": 20, "y": 40, "c": 1.0},
                    {"x": 44, "y": 40, "c": 1.0},
                    {"x": 20, "y": 55, "c": 1.0},
                    {"x": 44, "y": 55, "c": 1.0},
                    {"x": 32, "y": 45, "c": 1.0},
                    {"x": 20, "y": 58, "c": 1.0},
                    {"x": 44, "y": 58, "c": 1.0},
                    {"x": 20, "y": 60, "c": 1.0},
                    {"x": 44, "y": 60, "c": 1.0},
                    {"x": 20, "y": 62, "c": 1.0},
                    {"x": 44, "y": 62, "c": 1.0},
                    {"x": 20, "y": 63, "c": 1.0},
                    {"x": 44, "y": 63, "c": 1.0},
                    {"x": 20, "y": 63, "c": 1.0},
                ]
            }
        ]
    }
    import json

    hints_plain, _ = render_motion_hints(pose, ref, rig_type="human_to_creature")
    hints_json, _ = render_motion_hints(pose, ref, rig_type="human_to_creature", keypoints_json=json.dumps(kps))
    assert not torch.equal(hints_plain, hints_json)
    assert float(hints_json.max()) > float(hints_plain.max())
