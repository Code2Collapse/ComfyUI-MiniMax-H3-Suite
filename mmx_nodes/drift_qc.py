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
from mmx_utils.drift_qc import drift_per_frame_json, measure_exterior_drift

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
            ],
            outputs=[
                io.String.Output("drift_per_frame"),
                io.Image.Output("drift_heatmap"),
                io.Boolean.Output("passed"),
                io.String.Output("report"),
            ],
        )

    @classmethod
    def fingerprint_inputs(cls, original, edited, edit_mask, threshold_px=2.0):
        parts = [
            hashlib.md5(original.cpu().numpy().tobytes()).hexdigest(),
            hashlib.md5(edited.cpu().numpy().tobytes()).hexdigest(),
            hashlib.md5(edit_mask.cpu().numpy().tobytes()).hexdigest(),
            str(threshold_px),
        ]
        return ":".join(parts)

    @classmethod
    def execute(cls, original, edited, edit_mask, threshold_px=2.0) -> io.NodeOutput:
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
        drift_ui_json = json.dumps(
            {"drift_px": result.drift_per_frame, "passed": bool(result.passed)},
            separators=(",", ":"),
        )
        # First frame only — PreviewImage writes one PNG per batch frame. The curve the
        # widget plots comes from mmx_drift, not the heatmap. The IMAGE socket is unchanged.
        heatmap_ui = ui.PreviewImage(heatmap[:1], cls=cls).as_dict()
        return io.NodeOutput(
            drift_per_frame_json(result.drift_per_frame),
            heatmap,
            bool(result.passed),
            report,
            ui={**heatmap_ui, "mmx_drift": [drift_ui_json]},
        )
