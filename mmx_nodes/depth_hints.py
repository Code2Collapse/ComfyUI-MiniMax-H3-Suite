"""N9 — Temporal depth hints (passthrough default, crop warp)."""

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
from mmx_utils.depth_backends import DepthBackendUnavailableError, run_depth_backend
from mmx_utils.device import run_with_cpu_fallback
from mmx_utils.transform_types import H3Transform, H3TransformType

try:
    import comfy.model_management as mm
except ImportError:
    mm = None


class MiniMaxH3_TemporalDepth(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MiniMaxH3_TemporalDepth",
            display_name="H3 Temporal Depth",
            category="MiniMax H3/Control",
            description=(
                "Depth hints in crop space via plate-side backend + N1 transform warp. "
                "Default backend=passthrough (offline-safe). depthcrafter and dvd are "
                "selectable but raise a human error when weights/repos are missing."
            ),
            inputs=[
                io.Image.Input("images"),
                H3TransformType.Input("transform"),
                io.Combo.Input(
                    "backend",
                    options=["passthrough", "depthcrafter", "dvd"],
                    default="passthrough",
                ),
                io.Image.Input(
                    "passthrough_hints",
                    optional=True,
                    tooltip="Plate-space depth hints for passthrough backend.",
                ),
                io.Int.Input("window", default=17, min=1, max=128),
                io.Boolean.Input("emit_normals", default=False),
            ],
            outputs=[
                io.Image.Output("hints"),
                # NOTE: io.*.Output takes (id, display_name, tooltip, is_output_list) only —
# there is NO `optional` on Output (comfy_api/latest/_io.py:217). Optionality is an
# INPUT concept. This socket is always present; when emit_normals is False it carries
# a zeros tensor of the hint shape, which is documented in the node description.
                io.Image.Output("normals"),
                io.String.Output("report"),
            ],
        )

    @classmethod
    def fingerprint_inputs(
        cls,
        images,
        transform,
        backend="passthrough",
        passthrough_hints=None,
        window=17,
        emit_normals=False,
    ):
        base = hashlib.md5(images.cpu().numpy().tobytes()).hexdigest()
        base += ":" + transform.fingerprint()
        if passthrough_hints is not None:
            base += ":" + hashlib.md5(passthrough_hints.cpu().numpy().tobytes()).hexdigest()
        return f"{base}|{backend}|{window}|{emit_normals}"

    @classmethod
    def execute(
        cls,
        images,
        transform: H3Transform,
        backend="passthrough",
        passthrough_hints=None,
        window=17,
        emit_normals=False,
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
        _window = int(window)

        def _work(device: torch.device) -> tuple[torch.Tensor, torch.Tensor | None, str]:
            try:
                depth_plate, note = run_depth_backend(
                    backend,
                    images[:b].to(device),
                    passthrough_hints=passthrough_hints.to(device) if passthrough_hints is not None else None,
                )
            except DepthBackendUnavailableError as exc:
                raise ValueError(str(exc)) from exc

            hints_out = []
            for c0 in range(0, b, chunk):
                c1 = min(c0 + chunk, b)
                warped = affine_crop_batch(depth_plate[c0:c1], boxes[c0:c1], cw, ch)
                hints_out.append(warped)
            hints = torch.cat(hints_out, dim=0)

            normals = None
            if emit_normals:
                from mmx_utils.depth_backends import normals_from_depth

                n_plate = normals_from_depth(depth_plate)
                norm_out = []
                for c0 in range(0, b, chunk):
                    c1 = min(c0 + chunk, b)
                    norm_out.append(affine_crop_batch(n_plate[c0:c1], boxes[c0:c1], cw, ch))
                normals = torch.cat(norm_out, dim=0)

            return hints, normals, f"{note}; window={_window}"

        hints, normals, note = run_with_cpu_fallback(_work, device=dev, label="MiniMaxH3_TemporalDepth")
        report = (
            f"backend={backend} {note}\n"
            f"crop={cw}x{ch} frames={b} planner={transform.planner_mode} "
            "(plate depth → affine_crop_batch)"
        )
        if emit_normals and normals is not None:
            return io.NodeOutput(hints.to(images.device), normals.to(images.device), report)
        return io.NodeOutput(hints.to(images.device), None, report)
