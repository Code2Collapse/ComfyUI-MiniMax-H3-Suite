"""Detection extent extraction — N1 owns planning, not detection."""

from __future__ import annotations

import json
from typing import Optional

import numpy as np
import torch


def parse_bboxes_json(text: str, frame_count: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Parse [[x1,y1,x2,y2], null, ...] into extent arrays."""
    raw = json.loads(text or "[]")
    if not isinstance(raw, list):
        raise ValueError("bboxes_json must be a JSON array")
    m = max(frame_count, len(raw))
    bx0 = np.zeros(m, dtype=np.float64)
    bx1 = np.zeros(m, dtype=np.float64)
    by0 = np.zeros(m, dtype=np.float64)
    by1 = np.zeros(m, dtype=np.float64)
    detected = np.zeros(m, dtype=bool)
    for i in range(m):
        entry = raw[i] if i < len(raw) else None
        if entry is None:
            continue
        if not isinstance(entry, (list, tuple)) or len(entry) != 4:
            raise ValueError(f"bboxes_json[{i}] must be [x1,y1,x2,y2] or null")
        x1, y1, x2, y2 = (float(v) for v in entry)
        bx0[i] = min(x1, x2)
        bx1[i] = max(x1, x2)
        by0[i] = min(y1, y2)
        by1[i] = max(y1, y2)
        detected[i] = True
    return bx0, bx1, by0, by1, detected


def extents_from_mask(mask: torch.Tensor, threshold: float = 0.5) -> Optional[tuple[float, float, float, float]]:
    """Return (x0,y0,x1,y1) inclusive pixel extents or None."""
    m = mask.squeeze()
    if m.ndim != 2:
        return None
    ys, xs = (m > threshold).nonzero(as_tuple=True)
    if xs.numel() == 0:
        return None
    return (
        float(xs.min()),
        float(ys.min()),
        float(xs.max()),
        float(ys.max()),
    )


def extents_from_masks(
    masks: torch.Tensor,
    frame_count: int,
    threshold: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    m = frame_count
    bx0 = np.zeros(m, dtype=np.float64)
    bx1 = np.zeros(m, dtype=np.float64)
    by0 = np.zeros(m, dtype=np.float64)
    by1 = np.zeros(m, dtype=np.float64)
    detected = np.zeros(m, dtype=bool)
    if masks is None:
        return bx0, bx1, by0, by1, detected
    if masks.ndim == 2:
        masks = masks.unsqueeze(0)
    for i in range(m):
        fi = min(i, masks.shape[0] - 1)
        ext = extents_from_mask(masks[fi], threshold)
        if ext is None:
            continue
        bx0[i], by0[i], bx1[i], by1[i] = ext
        detected[i] = True
    return bx0, bx1, by0, by1, detected


def interpolate_gaps(
    bx0: np.ndarray,
    bx1: np.ndarray,
    by0: np.ndarray,
    by1: np.ndarray,
    detected: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Linear gap-fill on extent tracks for undetected frames."""
    idx = np.arange(len(bx0))
    known = detected.astype(bool)
    if not known.any():
        return bx0, bx1, by0, by1
    for arr in (bx0, bx1, by0, by1):
        arr[~known] = np.interp(idx[~known], idx[known], arr[known])
    return bx0, bx1, by0, by1


def merge_detection_sources(
    *,
    frame_count: int,
    bboxes_json: str,
    subject_mask: Optional[torch.Tensor],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if bboxes_json and bboxes_json.strip() not in ("", "[]"):
        bx0, bx1, by0, by1, det = parse_bboxes_json(bboxes_json, frame_count)
    elif subject_mask is not None:
        bx0, bx1, by0, by1, det = extents_from_masks(subject_mask, frame_count)
    else:
        raise ValueError(
            "H3 Track + Crop needs detections: connect subject_mask and/or bboxes_json. "
            "This node is a planner, not a detector."
        )
    bx0, bx1, by0, by1 = interpolate_gaps(bx0, bx1, by0, by1, det)
    return bx0, bx1, by0, by1, det
