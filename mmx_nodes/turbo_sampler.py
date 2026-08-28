# Apache License 2.0 — MiniMax-H3 Turbo LoRA / sampler reference
# PORTED FROM: comfyui-minimax-h3-turbo :: __init__.py @ 4274783a23afcfdbea3b4876cb79effd6c510785

"""H3 Turbo sampler node — returns a KSAMPLER for SamplerCustomAdvanced."""

from __future__ import annotations

import sys
from pathlib import Path

from comfy_api.latest import io

_PKG = Path(__file__).resolve().parents[1]
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from mmx_utils.turbo_sampler import build_ksampler


class MiniMaxH3_TurboSampler(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MiniMaxH3_TurboSampler",
            display_name="H3 Turbo Sampler (4-step)",
            category="MiniMax H3/Sampling",
            description=(
                "4-step sampler for the MiniMax-H3 Turbo LoRA. Feed into "
                "SamplerCustomAdvanced with BasicScheduler set to 4 steps. "
                "Auto-adapts to ComfyUI version: recent builds with ModelSamplingAV "
                "use a single-schedule Euler step; older builds step video and audio "
                "on separate clocks."
            ),
            inputs=[],
            outputs=[io.Sampler.Output("sampler")],
        )

    @classmethod
    def execute(cls) -> io.NodeOutput:
        return io.NodeOutput(build_ksampler())
