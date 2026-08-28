"""N3 — H3 Control Hints (crop-space extractors, no Union loader)."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import torch

from comfy_api.latest import io

_PKG = Path(__file__).resolve().parents[1]
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from mmx_utils.affine_transform import affine_crop_batch
from mmx_utils.canny_torch import canny_torch
from mmx_utils.device import run_with_cpu_fallback
from mmx_utils.transform_types import H3Transform, H3TransformType

try:
    import comfy.model_management as mm
except ImportError:
    mm = None


class MiniMaxH3_ControlHints(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MiniMaxH3_ControlHints",
            display_name="H3 Control Hints",
            category="MiniMax H3/Spine",
            description=(
                "Produce control hints in CROP space using the exact H3_TRANSFORM from "
                "Track + Crop (same affine_crop_batch as N1 — never re-detect).\n\n"
                "canny: native torch Canny on the plate, warped into crop space.\n"
                "passthrough: warp externally computed plate_hints (depth/pose/HED/MLSD).\n\n"
                "These hints currently have NO CONSUMER in ComfyUI — ControlNet-Union has no "
                "apply path yet. They exist so crop-space geometry is correct the day it does."
            ),
            inputs=[
                io.Image.Input("plate"),
                H3TransformType.Input("transform"),
                io.Combo.Input("hint_type", options=["canny", "passthrough"], default="canny"),
                io.Image.Input(
                    "plate_hints",
                    optional=True,
                    tooltip="Plate-space hints for passthrough mode (depth/pose/HED/MLSD).",
                ),
                io.Float.Input("low_threshold", default=0.1, min=0.0, max=1.0, step=0.01),
                io.Float.Input("high_threshold", default=0.2, min=0.0, max=1.0, step=0.01),
                io.Float.Input("blur_sigma", default=1.4, min=0.1, max=8.0, step=0.1),
            ],
            outputs=[
                io.Image.Output("hints"),
                io.String.Output("report"),
            ],
        )

    @classmethod
    def fingerprint_inputs(
        cls,
        plate,
        transform,
        hint_type="canny",
        plate_hints=None,
        low_threshold=0.1,
        high_threshold=0.2,
        blur_sigma=1.4,
    ):
        base = hashlib.md5(plate.cpu().numpy().tobytes()).hexdigest()
        base += ":" + transform.fingerprint()
        base += f":{hint_type}|{low_threshold}|{high_threshold}|{blur_sigma}"
        if plate_hints is not None:
            base += ":" + hashlib.md5(plate_hints.cpu().numpy().tobytes()).hexdigest()
        return base

    @classmethod
    def execute(
        cls,
        plate,
        transform: H3Transform,
        hint_type="canny",
        plate_hints=None,
        low_threshold=0.1,
        high_threshold=0.2,
        blur_sigma=1.4,
    ):
        if plate.ndim == 3:
            plate = plate.unsqueeze(0)
        boxes = list(transform.boxes)
        cw, ch = transform.canvas
        b = min(plate.shape[0], len(boxes))

        dev = plate.device
        if mm is not None:
            try:
                dev = mm.get_torch_device()
            except Exception:
                dev = plate.device

        chunk = 8

        def _work(dev):
            hints_out = []
            for c0 in range(0, b, chunk):
                c1 = min(c0 + chunk, b)
                chunk_plate = plate[c0:c1].to(dev)
                chunk_boxes = boxes[c0:c1]

                if hint_type == "passthrough":
                    if plate_hints is None:
                        raise ValueError(
                            "hint_type=passthrough requires plate_hints "
                            "(depth/pose/HED/MLSD in plate space)."
                        )
                    src = plate_hints
                    if src.ndim == 3:
                        src = src.unsqueeze(0)
                    src = src[c0:c1].to(dev)
                else:
                    src = canny_torch(
                        chunk_plate,
                        low_threshold=float(low_threshold),
                        high_threshold=float(high_threshold),
                        blur_sigma=float(blur_sigma),
                    )

                warped = affine_crop_batch(src, chunk_boxes, cw, ch)
                hints_out.append(warped.to(plate.dtype))
            return torch.cat(hints_out, dim=0).to(plate.device)

        hints = run_with_cpu_fallback(_work, device=dev, label="MiniMaxH3_ControlHints")
        report = (
            f"hint_type={hint_type} crop_space={cw}x{ch} frames={b}\n"
            f"transform planner={transform.planner_mode} (shared affine spine)\n"
            f"No downstream Union apply node exists yet — hints are geometry-only."
        )
        return io.NodeOutput(hints, report)
