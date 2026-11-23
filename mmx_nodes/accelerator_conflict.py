"""P11 — Accelerator conflict guard (inline MODEL passthrough)."""

from __future__ import annotations

import sys
from pathlib import Path

from comfy_api.latest import io

_PKG = Path(__file__).resolve().parents[1]
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from mmx_utils.sampling_guard import check_accelerator_conflicts, infer_total_blocks


class MiniMaxH3_AcceleratorConflict(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MiniMaxH3_AcceleratorConflict",
            display_name="H3 Accelerator Conflict Guard",
            category="MiniMax H3/Sampling",
            description=(
                "Fail loud before sampling if incompatible H3 accelerators are stacked "
                "(EasyCache + Block Cache, Spectrum + Block Cache, duplicate dit patches). "
                "Passes MODEL through on success so it can sit inline: "
                "N4 → ProtectedLayer → this → SigmaShift → …"
            ),
            inputs=[io.Model.Input("model", tooltip="H3 MODEL after all acceleration nodes. Its model_options are "
                                                    "checked for conflicting stacks (Block Cache with Spectrum or "
                                                    "EasyCache, EasyCache with LazyCache, First-Block Cache with "
                                                    "another cache, duplicate dit patches); a conflict raises an error.")],
            outputs=[
                io.Model.Output("model", tooltip="The input MODEL, unchanged (same object), when no conflict was "
                                                 "found. Continue the chain from here."),
                io.String.Output("report", tooltip="One line confirming that no conflicting cache or dit-patch stack "
                                                   "was detected."),
            ],
        )

    @classmethod
    def execute(cls, model) -> io.NodeOutput:
        total = infer_total_blocks(model)
        opts = model.model_options if hasattr(model, "model_options") else {}
        check_accelerator_conflicts(opts, total_blocks=total)
        report = "H3 accelerator graph OK — no conflicting cache or dit-patch stacks detected."
        return io.NodeOutput(model, report)
