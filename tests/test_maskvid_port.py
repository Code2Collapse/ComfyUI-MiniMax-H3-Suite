"""CPU-only tests for MaskVidExperiments port (MiniMaxH3 mask/crop nodes).

Every test carries one INVARIANT comment naming what it pins.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

PACK = Path(__file__).resolve().parents[1]
if str(PACK) not in sys.path:
    sys.path.insert(0, str(PACK))

from mmx_nodes.mask_cleanup import MiniMaxH3_MaskCleanup  # noqa: E402
from mmx_nodes.mask_to_latent import MiniMaxH3_LatentMaskToMask, MiniMaxH3_MaskToLatentSpace  # noqa: E402
from mmx_nodes.subject_crop import MiniMaxH3_SubjectCrop  # noqa: E402
from mmx_nodes.audio_mask import MiniMaxH3_AudioMaskToLatent  # noqa: E402
from mmx_utils.subject_crop_planner import plan  # noqa: E402


def test_mask_cleanup_preserves_subject_blob():
    # INVARIANT: a large persistent blob survives cleanup; specks do not zero the subject.
    masks = torch.zeros(4, 64, 64)
    masks[:, 20:44, 20:44] = 1.0
    masks[1, 5:8, 5:8] = 1.0  # tiny speck
    out, report = MiniMaxH3_MaskCleanup.execute(
        masks,
        0.5,
        {"method": "components", "min_pixels": 32, "min_frames": 2},
        edge_grow=0,
    )
    assert out[0, 30, 30].item() > 0.5
    assert "blobs_removed" in report


def test_combined_crop_emits_one_bbox_per_frame():
    # INVARIANT: combined mode returns constant-size crops and one bbox list entry per frame.
    n, h, w = 5, 128, 128
    images = torch.rand(n, h, w, 3)
    masks = torch.zeros(n, h, w)
    masks[:, 40:88, 40:88] = 1.0
    crops, cm, bboxes, report = MiniMaxH3_SubjectCrop.execute(
        images,
        masks,
        {"mode": "combined", "crop_scale": 1.5, "aspect_ratio": 0.0},
        divisible_by=16,
    )
    assert crops.shape[0] == n
    assert len(bboxes) == n
    assert "mode: combined" in report


def test_planner_derives_boxes_for_static_subject():
    # INVARIANT: combined planner returns identical boxes when the subject does not move.
    binary = np.zeros((3, 64, 64), dtype=bool)
    binary[:, 20:44, 20:44] = True
    boxes, info = plan(binary, "combined", {"crop_scale": 1.5, "aspect_ratio": 0.0}, 16)
    assert len(boxes) == 3
    assert boxes[0] == boxes[1] == boxes[2]
    assert boxes[0]["width"] % 16 == 0


def test_mask_to_latent_reduces_spatial_dims():
    # INVARIANT: manual spatial=8 halves a 64px side to 8 latent pixels (64/8).
    masks = torch.ones(4, 64, 64)
    compression = {
        "compression": "manual",
        "spatial": 8,
        "token_spatial": 1,
        "head_frames": 1,
        "head_latents": 1,
        "chunk_frames": 1,
        "chunk_latents": 1,
        "frames_per_latent": "",
    }
    latent, report = MiniMaxH3_MaskToLatentSpace.execute(
        masks, compression, "max", "max", 0, 0,
    )
    assert latent.shape == (4, 8, 8)
    assert "spatial_factor: 8" in report


def test_latent_mask_roundtrip_matches_frame_count():
    # INVARIANT: expand after reduce with chunk_frames=1 preserves pixel frame count.
    masks = torch.ones(6, 32, 32)
    compression = {
        "compression": "manual",
        "spatial": 8,
        "token_spatial": 1,
        "head_frames": 1,
        "head_latents": 1,
        "chunk_frames": 1,
        "chunk_latents": 1,
        "frames_per_latent": "",
    }
    latent, _ = MiniMaxH3_MaskToLatentSpace.execute(
        masks, compression, "max", "max", 0, 0,
    )
    expanded = MiniMaxH3_LatentMaskToMask.execute(latent, compression, 0)[0]
    assert expanded.shape[0] == 6


def test_audio_mask_fills_time_range_on_audio_latent():
    # INVARIANT: manual timing marks the requested second range on the audio axis.
    audio = torch.zeros(1, 8, 40)  # [B, channels, time]
    latent = {"samples": audio}
    timing = {"timing": "manual", "latents_per_second": 40.0, "layout": "time last"}
    out, report = MiniMaxH3_AudioMaskToLatent.execute(
        latent, timing, 0.0, 0.5, "replace",
        mask=None, time_ranges="0,0.5",
    )
    amask = out["noise_mask"]
    assert amask.shape[-1] == 40
    assert amask[..., :20].amax() == 1.0
    assert amask[..., 20:].amax() == 0.0
    assert "source: time_ranges" in report


# -- UI payload contract (w5_subject_crop.js depends on this) -----------------

def test_subject_crop_emits_a_crop_plan_for_the_widget():
    # INVARIANT: socket values never reach the browser - only NodeOutput.ui does -
    # so w5_subject_crop.js can only draw the plan if this payload exists and
    # carries frame size, per-frame boxes, and the jump list. Silently dropping
    # it would leave the widget permanently on "Run the node".
    import json as _json

    n, h, w = 6, 96, 128
    images = torch.rand(n, h, w, 3)
    masks = torch.zeros(n, h, w)
    masks[:, 30:70, 30:70] = 1.0
    out = MiniMaxH3_SubjectCrop.execute(
        images, masks,
        {"mode": "combined", "crop_scale": 1.5, "aspect_ratio": 0.0},
        divisible_by=16,
    )
    assert getattr(out, "ui", None), "NodeOutput carries no ui payload"
    raw_json = out.ui["mmx_crop_plan"][0]
    plan_d = _json.loads(raw_json)

    # frame dims must be the REAL frame, not the crop - the widget scales the
    # box against them, so swapping w/h would draw every box in the wrong place.
    assert plan_d["frame_w"] == w
    assert plan_d["frame_h"] == h
    assert len(plan_d["boxes"]) == n
    for b in plan_d["boxes"]:
        assert {"x", "y", "w", "h"} <= set(b)
        assert b["x"] >= 0 and b["y"] >= 0
        assert b["x"] + b["w"] <= w and b["y"] + b["h"] <= h
    assert isinstance(plan_d["jumps"], list)
    assert all(0 < j < n for j in plan_d["jumps"])


def test_crop_plan_marks_a_jump_only_when_the_box_actually_moves():
    # INVARIANT: the jump ticks are the whole point of the timeline. A static
    # subject must produce ZERO jumps, or every plan looks equally bad.
    import json as _json

    n, h, w = 8, 96, 96
    images = torch.rand(n, h, w, 3)
    masks = torch.zeros(n, h, w)
    masks[:, 30:60, 30:60] = 1.0          # never moves
    out = MiniMaxH3_SubjectCrop.execute(
        images, masks,
        {"mode": "combined", "crop_scale": 1.5, "aspect_ratio": 0.0},
        divisible_by=16,
    )
    plan_d = _json.loads(out.ui["mmx_crop_plan"][0])
    assert plan_d["jumps"] == []
