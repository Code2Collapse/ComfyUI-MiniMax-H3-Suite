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
                "optional region_mask and drift_report. gate=False lists all issues; with on_fail = stop the "
                "run it stops the queue instead."
            ),
            inputs=[
                io.String.Input("prompt", multiline=True,
                                tooltip="The six-section H3 prompt to check: bare labels subject_definitions:, "
                                        "summary: ([task type] prefix), retention_analysis: (one 'label: marker - "
                                        "text' line per subject), detailed_description:, overall_soundscape:, "
                                        "non_diegetic_music:, in that order; no ## headers."),
                io.String.Input("pipeline_notes", multiline=True,
                                tooltip="JSON object (empty = none). Checked keys: mask_required (true needs a "
                                        "region_mask that is not the whole frame), frame_count (must be 17k + 5), "
                                        "guidance_scale (must be 1.0). Invalid JSON fails the gate."),
                io.Mask.Input("region_mask", optional=True,
                              tooltip="The edit region (1 = edited). Only checked when pipeline_notes has "
                                      "mask_required: true - then it must exist and its first frame must not cover "
                                      "the whole frame."),
                io.String.Input("drift_report", optional=True, multiline=True,
                                tooltip="Tracking drift text or JSON: 'peak' and 'threshold' numbers are read (or "
                                        "peak = max of drift_px / drift_per_frame); peak above threshold (pixels) "
                                        "fails the gate."),
                # C2C L9.39: appended. A gate whose bool feeds nothing never ran (not an output node) and could
                # not stop anything; it is an output node now, and this decides what a failure does.
                io.Combo.Input("on_fail", options=["report only", "stop the run"], default="report only",
                               optional=True,
                               tooltip="report only: the verdict shows on the node and in the outputs. stop the "
                                       "run: a failure stops the queue - put this gate BEFORE your Save (wire the "
                                       "passthrough output) and nothing failing is saved."),
            ],
            outputs=[
                io.Boolean.Output("gate", tooltip="True when every pre-flight check passed; False lists issues in "
                                                  "report."),
                io.String.Output("report", tooltip="Text: one bullet per issue found, or 'All pre-flight checks "
                                                   "passed.' Also shown on the node."),
                io.String.Output("prompt_out", tooltip="The validated prompt, unchanged: wire it into the text "
                                                        "encoder so a failed check (stop the run) stops first."),
            ],
            is_output_node=True,
        )

    @classmethod
    def fingerprint_inputs(cls, prompt, pipeline_notes, region_mask=None, drift_report=None, on_fail="report only"):
        h = hashlib.md5((prompt or "").encode()).hexdigest()
        h += ":" + hashlib.md5((pipeline_notes or "").encode()).hexdigest()
        if region_mask is not None:
            h += ":" + hashlib.md5(region_mask.cpu().numpy().tobytes()).hexdigest()
        if drift_report:
            h += ":" + hashlib.md5(drift_report.encode()).hexdigest()
        return h + ":" + str(on_fail)

    @classmethod
    def execute(cls, prompt, pipeline_notes, region_mask=None, drift_report=None,
                on_fail="report only") -> io.NodeOutput:
        result = validate_edit_package(
            prompt=prompt or "",
            pipeline_notes=pipeline_notes or "",
            region_mask=region_mask,
            drift_report=drift_report,
        )
        if not result.gate and on_fail == "stop the run":
            raise ValueError(f"H3 Edit Validator stopped the run: {result.report}")
        return io.NodeOutput(result.gate, result.report, prompt, ui={"text": [result.report]})
