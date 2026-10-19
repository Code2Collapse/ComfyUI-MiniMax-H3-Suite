"""MiniMaxH3_GeometryReport - finds the geometry faults of a masked H3 setup from shapes alone (A16 Part II)."""

from __future__ import annotations

import json

import pytest
import torch

pytest.importorskip("comfy.ldm.minimax.model", reason="needs ComfyUI with MiniMax H3")
nested = pytest.importorskip("comfy.nested_tensor", reason="needs ComfyUI")

from mmx_nodes.geometry_report import MiniMaxH3_GeometryReport, build_report  # noqa: E402


def _av(frames=124, w=1440, h=1440, ticks=None, mask=None):
    lt = 2 if frames <= 5 else ((frames - 5) // 17) * 5 + 2
    ticks = round(frames / 24 * 40) if ticks is None else ticks
    lat = {"samples": nested.NestedTensor((torch.zeros(1, 24, lt, h // 16, w // 16), torch.zeros(1, 32, 2, ticks)))}
    if mask is not None:
        lat["noise_mask"] = nested.NestedTensor((mask, torch.zeros(1, 1, 2, ticks)))
    return lat


def _cond(refs=None, keyframes=None):
    d = {}
    if refs:
        d["minimax_refs"] = refs
    if keyframes:
        d["minimax_keyframes"] = keyframes
    return [[torch.zeros(1, 4, 8), d]]


def _ref_video(px_w, px_h, declared=None):
    lh, lw = px_h // 16, px_w // 16
    dh, dw = declared or (lh, lw)
    return {"kind": "video", "latent_t": 2, "latent_h": dh, "latent_w": dw, "ref_audio_t": 0,
            "latent": torch.zeros(1, 24, 2, lh, lw), "audio_latent": None}


def test_core_canvas_reference_video_is_reported_with_its_offset():
    rep = build_report(_cond(refs=[_ref_video(768, 768)]), _av())
    warn = " ".join(rep["warnings"])
    assert "-14.0 px horizontally" in warn and "-14.0 px vertically" in warn
    assert "target grid (zero offset)" in warn


def test_target_grid_reference_video_is_clean():
    rep = build_report(_cond(refs=[_ref_video(1440, 1440)]), _av())
    assert rep["warnings"] == []
    assert rep["streams"][0]["offset_if_same_shot"]["same_grid"]


def test_declared_and_actual_reference_sizes_must_agree():
    # the bug class of the old Image + Reference node: latent_h/w computed with the wrong divisor
    rep = build_report(_cond(refs=[_ref_video(768, 768, declared=(24, 24))]), _av())
    assert any("block says 24x24" in w for w in rep["warnings"])


def test_keyframe_on_another_grid_is_flagged():
    kf = {"resolved_frame_index": 0, "latent": torch.zeros(1, 24, 1, 48, 84)}
    rep = build_report(_cond(keyframes=[kf]), _av())
    assert any("keyframe 1" in w and "TARGET's grid" in w for w in rep["warnings"])
    ok = {"resolved_frame_index": 0, "latent": torch.zeros(1, 24, 1, 90, 90)}
    assert build_report(_cond(keyframes=[ok]), _av())["warnings"] == []


def test_audio_length_mismatch_is_flagged():
    rep = build_report(_cond(), _av(ticks=200))
    assert any("200 ticks" in w and "207" in w for w in rep["warnings"])


def test_mask_coverage_on_the_token_grid():
    m = torch.zeros(1, 1, 37, 90, 90)
    m[..., :, 40:50, 40:50] = 1.0
    rep = build_report(_cond(), _av(mask=m))
    cov = rep["mask"]["video"]
    assert cov["regenerated"] > 0 and cov["kept"] > 0
    assert rep["mask"]["audio_kept_pct"] == 100.0


def test_plate_pixels_from_the_crop_box():
    box = [[{"x": 100, "y": 50, "width": 720, "height": 720}]]
    rep = build_report(_cond(refs=[_ref_video(768, 768)]), _av(), bboxes=box)
    assert "in the original plate" in " ".join(rep["warnings"])
    assert rep["crop_box"]["target_to_plate_scale"] == [0.5, 0.5]


def test_node_outputs_text_and_json():
    out = MiniMaxH3_GeometryReport.execute(_cond(refs=[_ref_video(768, 768)]), _av())
    text, js = out.args[0], out.args[1]
    assert "Problems:" in text and json.loads(js)["target"]["pixels"] == [1440, 1440]


def test_not_a_latent_says_so():
    with pytest.raises(ValueError, match="not a latent"):
        build_report(_cond(), {"x": 1})
