# PORTED FROM: https://github.com/supElement/ComfyUI_MinimaxH3_AutoContext@bfa7ac3 h3_tst_patch.py (H3TSTPatch),
# Apache-2.0. Changes: see mmx_utils/tst_attention.py.
"""H3 TST Attention (experimental): training-free correction of the video queries (arXiv 2609.08505)."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from comfy_api.latest import io

_PKG = Path(__file__).resolve().parents[1]
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from mmx_utils.tst_attention import KILL_SWITCH, patch_model  # noqa: E402

LOG = logging.getLogger("MiniMaxSuite.tst_attention")
CATEGORY = "MiniMax H3/Sampling"


class MiniMaxH3_TSTAttention(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MiniMaxH3_TSTAttention",
            display_name="H3 TST Attention (experimental)",
            category=CATEGORY,
            description=(
                "Experimental, unmeasured on H3 here. Training-free correction against temporal flicker and small "
                "faces that break up (Spectral Transport Homeostasis, arXiv 2609.08505). For every video attention "
                "call it measures, per head, whether the frames over-mix or fragment, and scales the video queries "
                "to pull them back. Put it after other attention patches (closer to the sampler). It changes the "
                "result, so compare against tau = 0. The console prints one summary per sampling run."),
            search_aliases=["tst", "temporal state transport", "flicker", "small face", "spectral tension"],
            inputs=[
                io.Model.Input("model", tooltip="The MiniMax H3 model."),
                io.Float.Input("tau", default=0.2, min=0.0, max=1.0, step=0.01,
                               tooltip="Strength. 0 = no patch (the A/B baseline). Usual 0.1-0.3; up to 0.3 when "
                                       "small faces break up. Applied strongest in deep layers and early steps."),
            ],
            outputs=[io.Model.Output(display_name="model"), io.String.Output(display_name="report")],
        )

    @classmethod
    def execute(cls, model, tau=0.2) -> io.NodeOutput:
        if float(tau) == 0.0:
            report = "tau = 0: the model is passed through unchanged (A/B baseline)"
            return io.NodeOutput(model, report, ui={"text": [report]})
        patched, _state = patch_model(model, float(tau), log=lambda m: LOG.info("[H3 TST] %s", m))
        report = (f"TST mounted, tau {float(tau):.2f}; one summary line per sampling run in the console. "
                  f"Kill switch: {KILL_SWITCH}=off.")
        return io.NodeOutput(patched, report, ui={"text": [report]})


__all__ = ["MiniMaxH3_TSTAttention"]
