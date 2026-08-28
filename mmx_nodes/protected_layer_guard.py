"""P10 — Protected DiT layer guard (inline MODEL passthrough)."""

from __future__ import annotations

import sys
from pathlib import Path

from comfy_api.latest import io

_PKG = Path(__file__).resolve().parents[1]
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from mmx_utils.sampling_guard import check_protected_layers, infer_total_blocks


class MiniMaxH3_ProtectedLayerGuard(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MiniMaxH3_ProtectedLayerGuard",
            display_name="H3 Protected Layer Guard",
            category="MiniMax H3/Sampling",
            description=(
                "Refuse sampling if patches_replace hijacks protected DiT blocks "
                "(block 0, last block, Union injection blocks 10/20/30/40) or FinalLayer / "
                "norm_out / proj_out / audio_proj_out. Official Block Cache T8 hooks on "
                "block 0 and the last block are allowed. Passes MODEL through on success."
            ),
            inputs=[io.Model.Input("model")],
            outputs=[
                io.Model.Output("model"),
                io.String.Output("report"),
            ],
        )

    @classmethod
    def execute(cls, model) -> io.NodeOutput:
        total = infer_total_blocks(model)
        opts = model.model_options if hasattr(model, "model_options") else {}
        check_protected_layers(opts, total_blocks=total)
        report = (
            f"H3 protected layers OK — {total} DiT blocks; no illegal patches on "
            "block 0, last block, Union blocks, or FinalLayer paths."
        )
        return io.NodeOutput(model, report)
