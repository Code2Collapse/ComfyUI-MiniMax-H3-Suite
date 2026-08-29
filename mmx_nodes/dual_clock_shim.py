"""P4 — Dual-clock sampler shim (native ModelSamplingAV detection)."""

from __future__ import annotations

import sys
from pathlib import Path

from comfy_api.latest import io

_PKG = Path(__file__).resolve().parents[1]
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from mmx_utils.turbo_sampler import native_av_schedule


class MiniMaxH3_DualClockShim(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MiniMaxH3_DualClockShim",
            display_name="H3 Dual-Clock Shim",
            category="MiniMax H3/Sampling",
            description=(
                "Detect whether this MODEL uses ComfyUI's native ModelSamplingAV "
                "(single coupled video/audio σ schedule). When native AV is present, "
                "passes MODEL through unchanged. When absent, documents the legacy "
                "dual-clock path in MiniMaxH3_TurboSampler without fabricating a "
                "stepping implementation here."
            ),
            inputs=[io.Model.Input("model")],
            outputs=[
                io.Model.Output("model"),
                io.String.Output("report"),
            ],
        )

    @classmethod
    def execute(cls, model) -> io.NodeOutput:
        if native_av_schedule(model):
            report = (
                "H3 Dual-Clock Shim: native ModelSamplingAV detected on this MODEL. "
                "ComfyUI steps video and audio on their paired shift schedules — "
                "no shim patch is required. MODEL passed through unchanged."
            )
        else:
            report = (
                "H3 Dual-Clock Shim: ModelSamplingAV is not present on this MODEL. "
                "On older ComfyUI builds, the legacy dual-clock Euler path is "
                "implemented in MiniMaxH3_TurboSampler "
                "(mmx_utils/turbo_sampler.py) — video steps on the video σ grid while "
                "audio follows σ_a(σ_v) via a separate slope. This node does not "
                "install that path automatically; wire MiniMaxH3_TurboSampler into "
                "SamplerCustomAdvanced if you need legacy dual-clock stepping. "
                "MODEL passed through unchanged."
            )
        return io.NodeOutput(model, report)
