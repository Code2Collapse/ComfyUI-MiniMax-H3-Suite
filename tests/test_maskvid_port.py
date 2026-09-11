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
