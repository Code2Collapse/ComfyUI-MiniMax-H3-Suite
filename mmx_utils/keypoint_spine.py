# MIT License — ComfyUI-H3-FaceRefine
# PORTED FROM: ComfyUI-H3-FaceRefine :: nodes.py _interp_gaps @ HEAD

from __future__ import annotations

import json
import math
from typing import Iterable

import numpy as np
import torch

from .jitterless_boxes import OneEuroFilter

MIN_COVERAGE_KEYPOINTS = 17

# COCO-WholeBody foot / ankle indices (feet for ground-contact shots).
FOOT_KEYPOINT_INDICES: tuple[int, ...] = (15, 16, 17, 18, 19, 20)


def interp_gaps(vals: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Fill gaps by linear interpolation; hold at ends."""
    n = len(vals)
    idx = np.arange(n)
    if not valid.any():
        return np.zeros(n, dtype=np.float64)
    return np.interp(idx, idx[valid], vals[valid])


def smooth_keypoints_one_euro(
    keypoints: np.ndarray,
    *,
    confidence: np.ndarray | None = None,
    confidence_gate: float = 0.3,
    freq: float = 24.0,
    min_cutoff: float = 1.0,
    beta: float = 0.05,
    enabled: bool = True,
) -> np.ndarray:
    """
    Per-keypoint One Euro smoothing on [T, K, 3] (x, y, conf).
    Low-confidence frames are gap-filled before smoothing.
    """
    kps = np.asarray(keypoints, dtype=np.float64).copy()
    if kps.ndim != 3 or kps.shape[0] < 2 or not enabled:
        return kps.astype(np.float32)

    t, k, _ = kps.shape
    conf = kps[..., 2] if confidence is None else np.asarray(confidence, dtype=np.float64)
    out = kps.copy()

    for j in range(k):
        valid = conf[:, j] >= float(confidence_gate)
        for axis in (0, 1):
            series = kps[:, j, axis].copy()
            filled = interp_gaps(series, valid)
            filt = OneEuroFilter(freq=freq, min_cutoff=min_cutoff, beta=beta)
            smoothed = np.empty(t, dtype=np.float64)
            for i in range(t):
                smoothed[i] = filt(float(filled[i]))
            out[:, j, axis] = smoothed
        out[:, j, 2] = np.maximum(out[:, j, 2], conf[:, j])

    return out.astype(np.float32)


def fill_keypoint_gaps(
    keypoints: np.ndarray,
    *,
    confidence_gate: float = 0.3,
    max_gap: int = 0,
) -> np.ndarray:
    """Interpolate x/y for frames below confidence; optional max_gap limits span."""
    kps = np.asarray(keypoints, dtype=np.float32).copy()
    t, k, _ = kps.shape
    for j in range(k):
        valid = kps[:, j, 2] >= float(confidence_gate)
        if max_gap > 0:
            # Invalidate short runs only when surrounded by valid frames.
            i = 0
            while i < t:
                if valid[i]:
                    i += 1
                    continue
                j0 = i
                while i < t and not valid[i]:
                    i += 1
                if (j0 > 0 and i < t and (i - j0) > max_gap):
                    valid[j0:i] = False
        for axis in (0, 1):
            kps[:, j, axis] = interp_gaps(kps[:, j, axis].astype(np.float64), valid).astype(np.float32)
    return kps


def coverage_fraction(
    keypoints: np.ndarray,
    *,
    confidence_gate: float = 0.3,
    min_keypoints: int = MIN_COVERAGE_KEYPOINTS,
) -> float:
    """Fraction of frames with >= min_keypoints above confidence gate."""
    kps = np.asarray(keypoints, dtype=np.float32)
    if kps.ndim != 3 or kps.shape[0] == 0:
        return 0.0
    counts = (kps[..., 2] >= float(confidence_gate)).sum(axis=1)
    return float((counts >= int(min_keypoints)).mean())


def plate_point_to_crop(
    px: float,
    py: float,
    box: tuple[float, float, float, float],
    cw: int,
    ch: int,
) -> tuple[float, float]:
    """Map plate pixel (px,py) to crop canvas using N1 box (INVARIANT 10)."""
    x, y, bw, bh = box
    if bw <= 0 or bh <= 0:
        return 0.0, 0.0
    return (px - x) / bw * cw, (py - y) / bh * ch


def warp_keypoints_to_crop(
    keypoints: np.ndarray,
    boxes: Iterable[tuple[float, float, float, float]],
    canvas: tuple[int, int],
) -> np.ndarray:
    """[T,K,3] plate keypoints → crop space."""
    kps = np.asarray(keypoints, dtype=np.float32).copy()
    cw, ch = canvas
    boxes_list = list(boxes)
    for i in range(kps.shape[0]):
        box = boxes_list[i] if i < len(boxes_list) else boxes_list[-1]
        for j in range(kps.shape[1]):
            cx, cy = plate_point_to_crop(float(kps[i, j, 0]), float(kps[i, j, 1]), box, cw, ch)
            kps[i, j, 0] = cx
            kps[i, j, 1] = cy
    return kps


def render_pose_hints(
    keypoints_crop: np.ndarray,
    canvas: tuple[int, int],
    *,
    confidence_gate: float = 0.3,
) -> torch.Tensor:
    """Rasterise skeleton lines into [T,H,W,3] hint images (0..1)."""
    cw, ch = canvas
    t = int(keypoints_crop.shape[0])
    hints = np.zeros((t, ch, cw, 3), dtype=np.float32)
    # Minimal whole-body chain (torso + legs incl. ankles).
    edges = (
        (5, 7),
        (7, 9),
        (6, 8),
        (8, 10),
        (5, 6),
        (5, 11),
        (6, 12),
        (11, 12),
        (11, 13),
        (13, 15),
        (12, 14),
        (14, 16),
        (15, 17),
        (16, 20),
    )
    for fi in range(t):
        layer = hints[fi]
        for a, b in edges:
            if a >= keypoints_crop.shape[1] or b >= keypoints_crop.shape[1]:
                continue
            pa = keypoints_crop[fi, a]
            pb = keypoints_crop[fi, b]
            if pa[2] < confidence_gate or pb[2] < confidence_gate:
                continue
            x0, y0 = int(round(pa[0])), int(round(pa[1]))
            x1, y1 = int(round(pb[0])), int(round(pb[1]))
            _draw_line(layer, x0, y0, x1, y1, value=1.0)
    return torch.from_numpy(hints)


def _draw_line(img: np.ndarray, x0: int, y0: int, x1: int, y1: int, value: float = 1.0) -> None:
    h, w = img.shape[:2]
    steps = int(max(abs(x1 - x0), abs(y1 - y0), 1))
    for s in range(steps + 1):
        t = s / max(steps, 1)
        x = int(round(x0 + (x1 - x0) * t))
        y = int(round(y0 + (y1 - y0) * t))
        if 0 <= x < w and 0 <= y < h:
            img[y, x, :] = value


def keypoints_to_json(keypoints_crop: np.ndarray) -> str:
    payload = {
        "format": "minimax_h3_keypoints_v1",
        "frames": [
            {
                "keypoints": [
                    {"x": float(kp[0]), "y": float(kp[1]), "c": float(kp[2])}
                    for kp in frame
                ]
            }
            for frame in keypoints_crop
        ],
    }
    return json.dumps(payload, separators=(",", ":"))
