"""N8 — Strongest pose hints (plate detect → crop warp)."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import torch

from comfy_api.latest import io

_PKG = Path(__file__).resolve().parents[1]
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from mmx_utils.affine_transform import affine_crop_batch
from mmx_utils.device import run_with_cpu_fallback
from mmx_utils.keypoint_spine import (
    coverage_fraction,
    fill_keypoint_gaps,
    keypoints_to_json,
    render_pose_hints,
    smooth_keypoints_one_euro,
    warp_keypoints_to_crop,
)
from mmx_utils.transform_types import H3Transform, H3TransformType
from mmx_utils.vitpose_plate import VitPoseUnavailableError, detect_keypoints_plate

try:
    import comfy.model_management as mm
except ImportError:
    mm = None


class MiniMaxH3_StrongestPose(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MiniMaxH3_StrongestPose",
            display_name="H3 Strongest Pose",
            category="MiniMax H3/Control",
            description=(
                "Detect pose in PLATE space (ViTPose-H wholebody when available), then warp "
                "hints to crop space via the N1 H3_TRANSFORM — never detect in crop space "
                "(INVARIANT 10). Includes feet/ankles for ground-contact shots."
            ),
            inputs=[
                io.Image.Input("images"),
                H3TransformType.Input("transform"),
                io.Combo.Input(
                    "mode",
                    options=["inhouse_vitpose", "passthrough"],
                    default="inhouse_vitpose",
                ),
                io.Image.Input(
                    "passthrough_hints",
                    optional=True,
                    tooltip="Plate-space pose/depth hints when mode=passthrough.",
                ),
                io.Float.Input("confidence", default=0.3, min=0.0, max=1.0, step=0.01),
                io.Boolean.Input("temporal_smooth", default=True),
                io.Int.Input("gap_fill", default=3, min=0, max=30),
            ],
            outputs=[
                io.Image.Output("hints"),
                io.String.Output("keypoints_json"),
                io.Float.Output("coverage"),
                io.String.Output("report"),
            ],
        )

    @classmethod
    def fingerprint_inputs(
        cls,
        images,
        transform,
        mode="inhouse_vitpose",
        passthrough_hints=None,
        confidence=0.3,
        temporal_smooth=True,
        gap_fill=3,
    ):
        base = hashlib.md5(images.cpu().numpy().tobytes()).hexdigest()
        base += ":" + transform.fingerprint()
        if passthrough_hints is not None:
            base += ":" + hashlib.md5(passthrough_hints.cpu().numpy().tobytes()).hexdigest()
        return f"{base}|{mode}|{confidence}|{temporal_smooth}|{gap_fill}"

    @classmethod
    def execute(
        cls,
        images,
        transform: H3Transform,
        mode="inhouse_vitpose",
        passthrough_hints=None,
        confidence=0.3,
        temporal_smooth=True,
        gap_fill=3,
    ) -> io.NodeOutput:
        if images.ndim == 3:
            images = images.unsqueeze(0)
        boxes = list(transform.boxes)
        cw, ch = transform.canvas
        b = min(images.shape[0], len(boxes))

        dev = images.device
        if mm is not None:
            try:
                dev = mm.get_torch_device()
            except Exception:
                dev = images.device

        chunk = 4

        def _work(device: torch.device) -> tuple[torch.Tensor, str, float, str]:
            if mode == "passthrough":
                if passthrough_hints is None:
                    raise ValueError(
                        "mode=passthrough requires passthrough_hints in plate space."
                    )
                src = passthrough_hints
                if src.ndim == 3:
                    src = src.unsqueeze(0)
                hints_chunks = []
                for c0 in range(0, b, chunk):
                    c1 = min(c0 + chunk, b)
                    warped = affine_crop_batch(
                        src[c0:c1].to(device),
                        boxes[c0:c1],
                        cw,
                        ch,
                    )
                    hints_chunks.append(warped)
                hints = torch.cat(hints_chunks, dim=0)
                cov = 1.0
                kjson = keypoints_to_json(
                    np.zeros((b, 133, 3), dtype=np.float32)
                )
                return hints, kjson, cov, "passthrough plate hints warped to crop space"

            try:
                kps_plate, det_note = detect_keypoints_plate(
                    images[:b],
                    confidence=float(confidence),
                )
            except VitPoseUnavailableError as exc:
                raise ValueError(str(exc)) from exc

            kps_plate = fill_keypoint_gaps(
                kps_plate,
                confidence_gate=float(confidence),
                max_gap=int(gap_fill),
            )
            kps_plate = smooth_keypoints_one_euro(
                kps_plate,
                confidence_gate=float(confidence),
                enabled=bool(temporal_smooth),
            )
            kps_crop = warp_keypoints_to_crop(kps_plate, boxes[:b], (cw, ch))
            hints = render_pose_hints(kps_crop, (cw, ch), confidence_gate=float(confidence))
            cov = coverage_fraction(kps_plate, confidence_gate=float(confidence))
            kjson = keypoints_to_json(kps_crop)
            return hints.to(images.dtype), kjson, cov, det_note

        hints, kjson, cov, note = run_with_cpu_fallback(
            _work, device=dev, label="MiniMaxH3_StrongestPose"
        )
        report = (
            f"{note}\n"
            f"mode={mode} coverage={cov:.3f} confidence_gate={confidence} "
            f"smooth={temporal_smooth} gap_fill={gap_fill}\n"
            f"crop={transform.canvas[0]}x{transform.canvas[1]} frames={b} "
            f"planner={transform.planner_mode} (shared transform spine)"
        )
        return io.NodeOutput(hints.to(images.device), kjson, float(cov), report)
