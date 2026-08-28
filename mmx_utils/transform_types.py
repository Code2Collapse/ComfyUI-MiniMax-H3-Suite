"""H3 crop/stitch transform spine — one object per clip, computed in N1."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Optional

@dataclass(frozen=True)
class H3Transform:
    """Per-frame float crop boxes and metadata for stitch-back."""

    boxes: tuple[tuple[float, float, float, float], ...]
    canvas: tuple[int, int]
    src_size: tuple[int, int]
    frames: int
    weights: tuple[float, ...]
    detected: tuple[bool, ...]
    subject_rect: Optional[tuple[tuple[float, float, float, float], ...]]
    crop_factor: float
    planner_mode: str

    def fingerprint(self) -> str:
        payload = {
            "boxes": self.boxes,
            "canvas": self.canvas,
            "src_size": self.src_size,
            "frames": self.frames,
            "weights": self.weights,
            "detected": self.detected,
            "subject_rect": self.subject_rect,
            "crop_factor": self.crop_factor,
            "planner_mode": self.planner_mode,
        }
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.md5(raw.encode("utf-8")).hexdigest()


try:
    from comfy_api.latest._io import ComfyTypeIO, comfytype

    @comfytype(io_type="H3_TRANSFORM")
    class H3TransformType(ComfyTypeIO):
        Type = H3Transform
except ImportError:
    H3TransformType = None  # type: ignore[misc, assignment]
