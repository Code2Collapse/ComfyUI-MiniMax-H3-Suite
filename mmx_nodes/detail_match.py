"""N3 — H3 Detail Match (measured sharpness + plate grain)."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

from comfy_api.latest import io

_PKG = Path(__file__).resolve().parents[1]
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from mmx_utils.detail_match import detail_match_frames  # noqa: E402


def _normalize_boxes(bboxes, n: int) -> list[dict]:
    if isinstance(bboxes, dict):
        return [bboxes] * n
    if isinstance(bboxes, list) and bboxes and all(isinstance(f, list) and f for f in bboxes):
        frames = bboxes
        if len(frames) == 1:
            return [frames[0][0]] * n
        return [f[0] for f in frames[:n]]
    raise ValueError("bboxes must be a single box or a per-frame list of box lists")


def _align_counts(cropped_images, original_images, bboxes, edit_mask):
    counts = {
        "cropped_images": cropped_images.shape[0],
        "original_images": original_images.shape[0],
    }
    if isinstance(bboxes, list) and bboxes and all(isinstance(f, list) and f for f in bboxes):
        counts["bboxes"] = len(bboxes)
    if edit_mask is not None and edit_mask.shape[0] > 1:
        counts["edit_mask"] = edit_mask.shape[0]
    n = min(counts.values())
    if n < max(counts.values()):
        logging.warning(
            "MiniMaxH3 DetailMatch: frame counts differ (%s), using first %d",
            ", ".join(f"{k}={v}" for k, v in counts.items()),
            n,
        )
    return n


class MiniMaxH3_DetailMatch(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MiniMaxH3_DetailMatch",
            display_name="H3 Detail Match (measured sharpness + plate grain)",
            category="MiniMax H3/Finish",
            description=(
                "Restore the plate's measured sharpness and grain on decoded crops BEFORE "
                "Plate Restore composites them back.\n\n"
                "MTF gains are measured on the preserved ring (same evidence as Plate "
                "Restore) and normalised so contrast differences are not mistaken for blur. "
                "Grain is matched from the plate's noise level function. Wire this between "
                "decode and Plate Restore."
            ),
            inputs=[
                io.Image.Input(
                    "cropped_images",
                    tooltip=(
                        "Raw decoded crops from the VAE — one per frame, before compositing. "
                        "Output geometry matches input so you can feed straight into Plate Restore."
                    ),
                ),
                io.Image.Input(
                    "original_images",
                    tooltip=(
                        "Full-resolution plate frames. The bbox region is resized to crop size "
                        "for measurement — same path Plate Restore uses."
                    ),
                ),
                io.BoundingBox.Input(
                    "bboxes",
                    force_input=True,
                    tooltip="Subject Crop boxes — one per frame.",
                ),
                io.Mask.Input(
                    "edit_mask",
                    optional=True,
                    tooltip=(
                        "Crop-space mask: white = regenerated subject. Defines where MTF and "
                        "grain are applied. Without it the whole crop is treated as generated."
                    ),
                ),
                io.Int.Input(
                    "grow_px",
                    default=8,
                    min=0,
                    max=128,
                    tooltip=(
                        "Grow the edit mask before building the preserved ring for MTF "
                        "measurement. Same semantics as Plate Restore — too large and the ring "
                        "vanishes."
                    ),
                ),
                io.Float.Input(
                    "sharpen",
                    default=0.7,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip=(
                        "How much of the measured MTF correction to apply. 0 = off. "
                        "Turn down if edges halate or grain doubles up with Detail Reinject."
                    ),
                ),
                io.Float.Input(
                    "max_gain",
                    default=2.5,
                    min=1.0,
                    max=8.0,
                    step=0.05,
                    tooltip=(
                        "Cap per-frequency gain — prevents boosting sensor noise as texture. "
                        "Lower on noisy plates."
                    ),
                ),
                io.Float.Input(
                    "grain",
                    default=1.0,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip=(
                        "How much plate grain to add inside the edit region. 0 = MTF only. "
                        "Turn down if the composite already looks gritty."
                    ),
                ),
                io.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=0x7FFFFFFF,
                    tooltip="Seed for per-frame grain synthesis (deterministic per frame index).",
                ),
            ],
            outputs=[
                io.Image.Output(
                    "cropped_images",
                    tooltip="Detail-matched crops — same size as input, ready for Plate Restore.",
                ),
                io.String.Output("report", tooltip="Per-level gains, NLF measurements, grain added."),
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
        sharpen=0.7,
        max_gain=2.5,
        grain=1.0,
        seed=0,
    ) -> io.NodeOutput:
        n = _align_counts(cropped_images, original_images, bboxes, edit_mask)
        cropped_images = cropped_images[:n]
        original_images = original_images[:n]
        boxes = _normalize_boxes(bboxes, n)
        if edit_mask is not None:
            edit_mask = edit_mask.expand(n, -1, -1) if edit_mask.shape[0] == 1 else edit_mask[:n]

        crops, report_dict = detail_match_frames(
            cropped_images,
            original_images,
            boxes,
            edit_mask,
            int(grow_px),
            float(sharpen),
            float(max_gain),
            float(grain),
            int(seed),
        )
        report_text = json.dumps(report_dict, separators=(",", ":")) + "\n\n" + report_dict["summary"]
        return io.NodeOutput(crops, report_text)
