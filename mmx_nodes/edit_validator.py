"""N13 — Edit validator gate (catalog #8)."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

from comfy_api.latest import io

_PKG = Path(__file__).resolve().parents[1]
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from mmx_utils.edit_validator import validate_edit_package


class MiniMaxH3_EditValidator(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MiniMaxH3_EditValidator",
            display_name="H3 Edit Validator",
            category="MiniMax H3/Authoring",
            description=(
                "Pre-flight gate (#8): validates six-section prompt, pipeline_notes, "
                "optional region_mask and drift_report. gate=False lists all issues — does not raise."
            ),
            inputs=[
                io.String.Input("prompt", multiline=True),
                io.String.Input("pipeline_notes", multiline=True),
                io.Mask.Input("region_mask", optional=True),
                io.String.Input("drift_report", optional=True, multiline=True),
            ],
            outputs=[
                io.Boolean.Output("gate"),
                io.String.Output("report"),
            ],
        )

    @classmethod
    def fingerprint_inputs(cls, prompt, pipeline_notes, region_mask=None, drift_report=None):
        h = hashlib.md5((prompt or "").encode()).hexdigest()
        h += ":" + hashlib.md5((pipeline_notes or "").encode()).hexdigest()
        if region_mask is not None:
            h += ":" + hashlib.md5(region_mask.cpu().numpy().tobytes()).hexdigest()
        if drift_report:
            h += ":" + hashlib.md5(drift_report.encode()).hexdigest()
        return h

    @classmethod
    def execute(cls, prompt, pipeline_notes, region_mask=None, drift_report=None) -> io.NodeOutput:
        result = validate_edit_package(
            prompt=prompt or "",
            pipeline_notes=pipeline_notes or "",
            region_mask=region_mask,
            drift_report=drift_report,
        )
        return io.NodeOutput(result.gate, result.report)
