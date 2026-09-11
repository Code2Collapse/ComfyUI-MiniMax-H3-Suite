"""CPU-only tests for ComfyUI-H3-FaceRefine port (MiniMaxH3 face nodes).

Every test carries one INVARIANT comment naming what it pins.
Weight-dependent execute paths (YOLO, InsightFace, SAM, VAE) are not tested here.
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

from mmx_utils.face_refine import (  # noqa: E402
    best_match,
    continuity_cost,
    feather_mask,
    format_transform_info,
    interp_gaps,
    iou,
    smooth,
)
from mmx_utils.transform_types import H3Transform  # noqa: E402


def test_iou_identical_boxes_is_one():
    # INVARIANT: identical boxes have IoU 1.0.
    box = (10.0, 20.0, 50.0, 80.0)
    assert iou(box, box) == pytest.approx(1.0)


def test_interp_gaps_holds_at_ends():
    # INVARIANT: gaps between valid samples are linearly filled; ends hold the nearest value.
    vals = np.array([1.0, 0.0, 0.0, 4.0], dtype=np.float64)
    valid = np.array([True, False, False, True])
    out = interp_gaps(vals, valid)
    assert out[0] == pytest.approx(1.0)
    assert out[3] == pytest.approx(4.0)
    assert out[1] == pytest.approx(2.0)
    assert out[2] == pytest.approx(3.0)


def test_smooth_window_one_is_identity():
    # INVARIANT: smooth with window<=1 returns the input unchanged.
    vals = np.array([3.0, 1.0, 5.0, 2.0])
    assert np.allclose(smooth(vals, 1), vals)


def test_continuity_cost_zero_when_box_matches_last():
    # INVARIANT: continuity cost is zero when the box centre and height match last.
    box = (100.0, 50.0, 140.0, 90.0)
    last = (120.0, 70.0, 40.0)
    assert continuity_cost(box, last) == pytest.approx(0.0)


def test_best_match_picks_highest_cosine():
    # INVARIANT: best_match returns the candidate with highest dot product to ref_emb.
    ref = np.array([1.0, 0.0], dtype=np.float32)
    cands = [
        ((0, 0, 1, 1), np.array([0.0, 1.0], dtype=np.float32)),
        ((0, 0, 2, 2), np.array([0.9, 0.1], dtype=np.float32)),
    ]
    idx, score = best_match(cands, ref)
    assert idx == 1
    assert score == pytest.approx(0.9)


def test_feather_mask_core_stays_one():
    # INVARIANT: feather_mask leaves the interior at 1.0 when feather>0.
    m = feather_mask(32, 32, feather=4, device=torch.device("cpu"), dtype=torch.float32)
    assert m[16, 16].item() == pytest.approx(1.0)
    assert m[0, 0].item() < 1.0


def test_format_transform_info_lists_canvas():
    # INVARIANT: format_transform_info header includes canvas and frame count from H3Transform.
    transform = H3Transform(
        boxes=((0.0, 0.0, 64.0, 64.0), (1.0, 1.0, 64.0, 64.0)),
        canvas=(512, 512),
        src_size=(1920, 1080),
        frames=2,
        weights=(1.0, 1.0),
        detected=(True, True),
        subject_rect=((200.0, 200.0, 112.0, 112.0), (200.0, 200.0, 112.0, 112.0)),
        crop_factor=2.5,
        planner_mode="face_track",
    )
    txt = format_transform_info(transform, max_rows=4)
    assert "canvas=512x512" in txt
    assert "frames=2" in txt
    assert "frame" in txt.splitlines()[1]


def test_load_detector_raises_file_not_found(monkeypatch):
    # INVARIANT: missing detector models raise FileNotFoundError with a clear message (R7).
    import mmx_utils.face_refine as fr

    class FakeFolderPaths:
        models_dir = "models"

        @staticmethod
        def get_full_path(_key, _name):
            return None

    monkeypatch.setitem(sys.modules, "folder_paths", FakeFolderPaths())
    monkeypatch.setattr("os.path.exists", lambda _p: False)
    fr._DETECTOR_CACHE.clear()

    with pytest.raises(FileNotFoundError, match="Face detector"):
        fr.load_detector("__nonexistent_face_detector_xyz__.pt")
