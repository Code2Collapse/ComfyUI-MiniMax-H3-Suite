"""N2 — H3 Plate Restore (offset, colour, literal-plate composite)."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import torch

from comfy_api.latest import io

_PKG = Path(__file__).resolve().parents[1]
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from mmx_utils.device import run_with_cpu_fallback
from mmx_utils.plate_restore import restore_frames

try:
    import comfy.model_management as mm
except Exception:  # not just ImportError: the comfy_kitchen skew raises AttributeError
    mm = None


def _normalize_boxes(bboxes, n: int) -> list[dict]:
    if isinstance(bboxes, dict):
        return [bboxes] * n
    if isinstance(bboxes, list) and bboxes and all(isinstance(f, list) and f for f in bboxes):
        frames = bboxes
        if len(frames) == 1:
            return [frames[0][0]] * n
        return [f[0] for f in frames[:n]]
    raise ValueError("bboxes must be a single box or a per-frame list of box lists")


def _align_frame_counts(cropped_images, original_images, bboxes, edit_mask):
    if isinstance(bboxes, list) and bboxes and all(isinstance(f, list) and f for f in bboxes):
        frames = bboxes
    else:
        frames = None
    counts = {
        "cropped_images": cropped_images.shape[0],
        "original_images": original_images.shape[0],
    }
    if frames is not None and len(frames) > 1:
        counts["bboxes"] = len(frames)
    if edit_mask is not None and edit_mask.shape[0] > 1:
        counts["edit_mask"] = edit_mask.shape[0]
    n = min(counts.values())
    if n < max(counts.values()):
        got = ", ".join(f"{k}={v}" for k, v in counts.items())
        logging.warning(
            f"MiniMaxH3 PlateRestore: frame counts differ ({got}), using the first {n} frames"
        )
    return n


class MiniMaxH3_PlateRestore(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MiniMaxH3_PlateRestore",
            display_name="H3 Plate Restore (offset · colour · clean plate)",
            category="MiniMax H3/Finish",
            description=(
                "Paste decoded crops back onto the plate with sub-pixel registration, "
                "linear-light colour matching on the preserved ring, and a composite that "
                "keeps literal plate pixels everywhere the edit mask is off.\n\n"
                "Superset of Subject Uncrop: wire cropped_masks / edit_mask so background "
                "inside the crop is not re-diffused."
            ),
            inputs=[
                io.Image.Input(
                    "cropped_images",
                    tooltip=(
                        "Generated crops from the decoder, one per frame. "
                        "Must match the Subject Crop output size. If these drift in colour "
                        "or position from the plate, registration and colour run on the "
                        "preserved ring before compositing."
                    ),
                ),
                io.Image.Input(
                    "original_images",
                    tooltip=(
                        "The untouched plate frames at full resolution. "
                        "Pixels outside the composite alpha are copied bit-exact from here — "
                        "this is what restores grain and sharpness the VAE destroyed."
                    ),
                ),
                io.BoundingBox.Input(
                    "bboxes",
                    force_input=True,
                    tooltip=(
                        "Crop boxes from Subject Crop (one per frame). "
                        "Wrong boxes misalign the paste and make registration fail."
                    ),
                ),
                io.Mask.Input(
                    "edit_mask",
                    optional=True,
                    tooltip=(
                        "Crop-space mask: white = regenerated subject, black = preserved plate. "
                        "If omitted, the entire crop is treated as regenerated (same as "
                        "Subject Uncrop without cropped_masks) and the report warns loudly."
                    ),
                ),
                io.Int.Input(
                    "grow_px",
                    default=8,
                    min=0,
                    max=128,
                    tooltip=(
                        "Grow the edit mask before building the preserved ring used for "
                        "shift/colour measurement. Too small: ring sits on the seam. "
                        "Too large: not enough pixels for a stable fit."
                    ),
                ),
                io.Int.Input(
                    "feather_px",
                    default=24,
                    min=0,
                    max=256,
                    tooltip=(
                        "Gaussian feather width for the edit mask at composite time. "
                        "Controls how softly the generated region blends into the plate."
                    ),
                ),
                io.Combo.Input(
                    "register",
                    options=["off", "measure", "correct"],
                    default="correct",
                    tooltip=(
                        "Spatial alignment on the preserved ring. "
                        "off: skip. measure: report shift only. correct: warp the generated "
                        "crop when shift is within max_shift_px; larger shifts are refused."
                    ),
                ),
                io.Float.Input(
                    "max_shift_px",
                    default=4.0,
                    min=0.0,
                    max=32.0,
                    step=0.05,
                    tooltip=(
                        "Maximum sub-pixel shift applied in correct mode. "
                        "Shifts above this are measured but not applied — check the report "
                        "if the plate still looks offset."
                    ),
                ),
                io.Combo.Input(
                    "colour",
                    options=["off", "mean", "affine"],
                    default="affine",
                    tooltip=(
                        "Colour match on the preserved ring in linear light. "
                        "affine: full 3×4 fit (falls back to mean if ill-conditioned). "
                        "mean: per-channel offset only. off: skip."
                    ),
                ),
                io.Boolean.Input(
                    "temporal_lag_check",
                    default=True,
                    tooltip=(
                        "Report frame-index lag between plate and generated crops (−3…+3). "
                        "Non-zero lag usually means a frame-count bug upstream — this node "
                        "never retimes, it only warns."
                    ),
                ),
            ],
            outputs=[
                io.Image.Output(
                    "images",
                    tooltip="Full frames with restored composite.",
                ),
                io.Mask.Output(
                    "restore_mask",
                    tooltip="The alpha actually used at composite time (full frame).",
                ),
                io.String.Output(
                    "report",
                    tooltip="JSON measurements plus a one-paragraph summary.",
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        cropped_images,
        original_images,
        bboxes,
        edit_mask=None,
        grow_px=8,
        feather_px=24,
        register="correct",
        max_shift_px=4.0,
        colour="affine",
        temporal_lag_check=True,
    ) -> io.NodeOutput:
        n = _align_frame_counts(cropped_images, original_images, bboxes, edit_mask)
        cropped_images = cropped_images[:n]
        original_images = original_images[:n]
        boxes = _normalize_boxes(bboxes, n)
        if edit_mask is not None:
            edit_mask = edit_mask.expand(n, -1, -1) if edit_mask.shape[0] == 1 else edit_mask[:n]

        dev = original_images.device
        if mm is not None:
            try:
                dev = mm.get_torch_device()
            except Exception:
                dev = original_images.device

        images, restore_mask, report_dict = run_with_cpu_fallback(
            lambda d: restore_frames(
                cropped_images,
                original_images,
                boxes,
                edit_mask,
                int(grow_px),
                int(feather_px),
                str(register),
                float(max_shift_px),
                str(colour),
                bool(temporal_lag_check),
                device=d,
            ),
            device=dev,
            label="MiniMaxH3_PlateRestore",
        )
        report_text = json.dumps(report_dict, separators=(",", ":")) + "\n\n" + report_dict["summary"]
        return io.NodeOutput(images, restore_mask, report_text)
