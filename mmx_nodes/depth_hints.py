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
from mmx_utils.depth_backends import BACKENDS, DepthBackendUnavailableError, run_depth_backend, temporal_align
from mmx_utils.device import run_with_cpu_fallback
from mmx_utils.transform_types import H3Transform, H3TransformType

try:
    import comfy.model_management as mm
except Exception:  # not just ImportError: the comfy_kitchen skew raises AttributeError
    mm = None


class MiniMaxH3_TemporalDepth(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MiniMaxH3_TemporalDepth",
            display_name="H3 Temporal Depth",
            category="MiniMax H3/Control",
            description=(
                "Depth hints in crop space: plate depth from a backend, warped through the N1 transform. "
                "passthrough = the depth map you wire into passthrough_hints (e.g. core 'Run Depth Anything 3'); "
                "with nothing wired it is a LUMINANCE proxy, not depth, and the report says so. The model backends "
                "(Depth Anything V2 / 3, Depth Pro, DepthCrafter, DVD) run through ComfyUI-CustomNodePacks' depth "
                "runners when that pack is installed; otherwise they stop with what is missing. temporal = align "
                "(experimental, off by default): fits every frame's depth scale / offset to the median of the "
                "previous `window` frames and normalises the clip once - it removes per-frame scale / offset "
                "flicker (measured on synthetic jitter) but did not help a clean pan, so it is opt-in. normals "
                "carries zeros when emit_normals is off."
            ),
            inputs=[
                io.Image.Input("images"),
                H3TransformType.Input("transform"),
                io.Combo.Input(
                    "backend",
                    options=list(BACKENDS),
                    default="passthrough",
                ),
                io.Image.Input(
                    "passthrough_hints",
                    optional=True,
                    tooltip="Plate-space depth hints for passthrough backend.",
                ),
                io.Int.Input("window", default=17, min=1, max=128,
                             tooltip="Frames the temporal alignment looks back over (1 = no alignment)."),
                io.Boolean.Input("emit_normals", default=False),
                # L9.39: appended last so saved workflows keep their widget order
                io.Combo.Input("temporal", options=["off", "align (experimental)"], default="off", optional=True,
                               tooltip="align: fit each frame's depth scale / offset to the previous `window` frames "
                                       "(removes per-frame normalisation flicker). Experimental: measured to help "
                                       "per-frame scale jitter, not a clean pan."),
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
    def execute(
        cls,
        images,
        transform: H3Transform,
        backend="passthrough",
        passthrough_hints=None,
        window=17,
        emit_normals=False,
        temporal="off",
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
            align = str(temporal).startswith("align")
            stats = None
            if align and depth_plate.shape[0] > 1:
                depth_plate, stats = temporal_align(depth_plate, _window)
                note += (f"; temporal align over {_window} frames: flicker {stats['flicker_before']:.4f} -> "
                         f"{stats['flicker_after']:.4f}")

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

            return hints, normals, note

        hints, normals, note = run_with_cpu_fallback(_work, device=dev, label="MiniMaxH3_TemporalDepth")
        report = (
            f"backend={backend} {note}\n"
            f"crop={cw}x{ch} frames={b} planner={transform.planner_mode} "
            "(plate depth → affine_crop_batch)"
        )
        if emit_normals and normals is not None:
            return io.NodeOutput(hints.to(images.device), normals.to(images.device), report)
        return io.NodeOutput(hints.to(images.device), torch.zeros_like(hints, device=images.device), report)
