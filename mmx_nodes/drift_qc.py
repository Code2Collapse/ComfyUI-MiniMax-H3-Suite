"""N11 — Exterior drift QC (RAFT-small on CPU).

UI payload note: the heatmap goes out through `ui.PreviewImage`, which writes to the
temp folder (`FolderType.temp`, prefix `ComfyUI_temp_` — comfy_api/latest/_ui.py:395-400)
and is cleared by `cleanup_temp()` on startup and shutdown (main.py:531, main.py:614).
It is therefore a NATIVE node preview, not a saved artefact, and it accumulates nothing.

The W4 drift-curve widget deliberately IGNORES it: the curve is plotted from the
`mmx_drift` key of the same UI payload. Wiring W4 to the heatmap would duplicate a
preview ComfyUI already renders for free. Only the first frame is sent, because
PreviewImage writes one PNG per batch frame; the IMAGE socket keeps the full batch.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import torch

from comfy_api.latest import io, ui

_PKG = Path(__file__).resolve().parents[1]
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from mmx_utils.device import run_with_cpu_fallback
from mmx_utils.drift_qc import (
    drift_per_frame_json,
    measure_exterior_drift,
    placement_report,
    subject_placement,
)

try:
    import comfy.model_management as mm
except Exception:  # not just ImportError: the comfy_kitchen skew raises AttributeError
    mm = None


class MiniMaxH3_DriftQC(io.ComfyNode):
    """Measure spatial drift of the UNEDITED region between plate and edited output.

    CALIBRATION (measured on this box, RAFT-small, supersample=2, synthetic shifts):

        frame size   injected   reported   error
             16 px      1.0        1.0114   0.011
             32 px      1.0        1.0019   0.002
             48 px      1.0        0.8670   0.133
            192 px      0.0        0.0231   0.023
            192 px      0.5        0.3770   0.123
            192 px      1.0        0.8209   0.179
            192 px      2.0        2.0207   0.021
            256 px      1.0        0.8213   0.179

    KNOWN BIAS, not a bug: RAFT-small UNDER-reports around 1 px by roughly 0.18 px at
    native crop sizes. The noise floor at zero shift is ~0.02-0.04 px. Keep
    `threshold_px` at or above 0.25 so the gate sits above noise floor + bias; a
    tighter threshold measures the estimator, not the pipeline.

    Without supersampling the bias is far worse (192 px: 1.0 -> 0.7285, err 0.272;
    2.0 -> 1.6848, err 0.315) — i.e. outside the gate. See utils/drift_qc.py for why
    the supersample is explicit rather than an accident of frame size.
    """
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MiniMaxH3_DriftQC",
            display_name="H3 Drift QC",
            category="MiniMax H3/QC",
            description=(
                "Measure pixel displacement OUTSIDE the edit mask only (RAFT-small, CPU). "
                "Calibration gate: synthetic shifts 0/0.5/1/2 px should read within ±0.25 px."
            ),
            inputs=[
                io.Image.Input("original"),
                io.Image.Input("edited"),
                io.Mask.Input("edit_mask"),
                io.Float.Input("threshold_px", default=2.0, min=0.0, max=64.0, step=0.1),
                # A16, appended: where the SUBJECT ended up, not only whether the background moved
                io.Mask.Input("source_subject_mask", optional=True,
                              tooltip="Mask of the subject in the SOURCE frames (optional). With the output's "
                                      "subject mask it measures where the new subject sits against the old one."),
                io.Mask.Input("output_subject_mask", optional=True,
                              tooltip="Mask of the subject in the EDITED frames, from any segmenter run on them "
                                      "(e.g. the same points / prompt as the source mask)."),
                io.BoundingBox.Input("bboxes", optional=True, force_input=True,
                                     tooltip="Optional crop boxes (Subject Crop) when the masks are crop-sized: the "
                                             "placement is then also given in original-plate pixels."),
            ],
            outputs=[
                io.String.Output("drift_per_frame"),
                io.Image.Output("drift_heatmap"),
                io.Boolean.Output("passed"),
                io.String.Output("report"),
                io.String.Output("placement_json",
                                 tooltip="Subject placement per frame and summary (empty when the subject masks "
                                         "are not wired)."),
            ],
        )

    @classmethod
    def execute(cls, original, edited, edit_mask, threshold_px=2.0, source_subject_mask=None,
                output_subject_mask=None, bboxes=None) -> io.NodeOutput:
        dev = original.device
        if mm is not None:
            try:
                dev = mm.get_torch_device()
            except Exception:
                dev = original.device

        def _work(device: torch.device) -> tuple:
            result = measure_exterior_drift(
                original.to(device),
                edited.to(device),
                edit_mask.to(device),
                threshold_px=float(threshold_px),
                raft_size="small",
            )
            return result

        result = run_with_cpu_fallback(_work, device=dev, label="MiniMaxH3_DriftQC")
        report = result.report
        if not result.passed:
            report = "FAILED — " + report
        heatmap = result.drift_heatmap.to(original.device, original.dtype)
        peak_px = max(result.drift_per_frame) if result.drift_per_frame else 0.0
        peak_frame = (
            result.drift_per_frame.index(peak_px) if result.drift_per_frame else -1
        )
        drift_ui_json = json.dumps(
            {
                "drift_px": result.drift_per_frame,
                "passed": bool(result.passed),
                "threshold_px": float(threshold_px),
                "peak_px": float(peak_px),
                "peak_frame": int(peak_frame),
            },
            separators=(",", ":"),
        )
        # First frame only — PreviewImage writes one PNG per batch frame. The curve the
        # widget plots comes from mmx_drift, not the heatmap. The IMAGE socket is unchanged.
        heatmap_ui = ui.PreviewImage(heatmap[:1], cls=cls).as_dict()
        placement_json = ""
        if source_subject_mask is not None and output_subject_mask is not None:
            scale = _plate_scale(bboxes, source_subject_mask)
            placement = subject_placement(source_subject_mask.cpu(), output_subject_mask.cpu(), plate_scale=scale)
            report = report + "\n" + placement_report(placement)
            placement_json = json.dumps(placement, separators=(",", ":"))
        return io.NodeOutput(
            drift_per_frame_json(result.drift_per_frame),
            heatmap,
            bool(result.passed),
            report,
            placement_json,
            ui={**heatmap_ui, "mmx_drift": [drift_ui_json]},
        )


def _plate_scale(bboxes, mask):
    """(sx, sy) crop-mask px -> plate px from the first Subject Crop box; None without one."""
    box = None
    if isinstance(bboxes, dict):
        box = bboxes
    elif isinstance(bboxes, list) and bboxes:
        first = bboxes[0]
        box = first[0] if isinstance(first, list) and first else (first if isinstance(first, dict) else None)
    if not box:
        return None
    try:
        return float(box["width"]) / float(mask.shape[-1]), float(box["height"]) / float(mask.shape[-2])
    except (KeyError, TypeError, ZeroDivisionError):
        return None
