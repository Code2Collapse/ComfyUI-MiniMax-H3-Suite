"""P1 — Sigma shift with video/audio ratio lock (distinct from core MiniMaxH3SigmaShift)."""

from __future__ import annotations

import sys
from pathlib import Path

import comfy.model_sampling
from comfy_api.latest import io

_PKG = Path(__file__).resolve().parents[1]
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from mmx_utils.sigma_schedule import TRAINED_SHIFT_AUDIO, TRAINED_SHIFT_RATIO, TRAINED_SHIFT_VIDEO, apply_ratio_lock


class MiniMaxH3_SigmaShiftLocked(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MiniMaxH3_SigmaShiftLocked",
            display_name="H3 Sigma Shift (Ratio Locked)",
            category="MiniMax H3/Sampling",
            description=(
                "Patch ModelSamplingAV with a coherent video/audio shift pair. Unlike core "
                "MiniMaxH3SigmaShift, ratio_lock keeps shift_audio = shift_video / ratio "
                "(default ratio 4 → trained 12/3). Moving shift_video moves shift_audio "
                "unless you disable ratio_lock to dial audio independently."
            ),
            inputs=[
                io.Model.Input("model"),
                io.Float.Input(
                    "shift_video",
                    default=TRAINED_SHIFT_VIDEO,
                    min=0.01,
                    max=100.0,
                    step=0.01,
                ),
                io.Float.Input(
                    "shift_audio",
                    default=TRAINED_SHIFT_AUDIO,
                    min=0.01,
                    max=100.0,
                    step=0.01,
                ),
                io.Boolean.Input("ratio_lock", default=True),
                io.Float.Input("ratio", default=TRAINED_SHIFT_RATIO, min=0.1, max=100.0, step=0.01),
            ],
            outputs=[
                io.Model.Output("model"),
                io.String.Output("report"),
            ],
        )

    @classmethod
    def execute(
        cls,
        model,
        shift_video,
        shift_audio,
        ratio_lock=True,
        ratio=TRAINED_SHIFT_RATIO,
    ) -> io.NodeOutput:
        sv, sa, note = apply_ratio_lock(
            shift_video,
            shift_audio,
            ratio_lock=bool(ratio_lock),
            ratio=float(ratio),
        )
        m = model.clone()

        class ModelSamplingAdvanced(comfy.model_sampling.ModelSamplingAV, comfy.model_sampling.CONST):
            pass

        original = m.get_model_object("model_sampling")
        model_sampling = ModelSamplingAdvanced(model.model.model_config)
        model_sampling.set_parameters(shift=sv, audio_shift=sa)
        if hasattr(original, "noise_scale"):
            model_sampling.set_noise_scale(original.noise_scale)
        m.add_object_patch("model_sampling", model_sampling)

        to = m.model_options["transformer_options"] = m.model_options.get("transformer_options", {}).copy()
        to["minimax_h3_sigma_shift_video"] = sv
        to["minimax_h3_sigma_shift_audio"] = sa

        report = f"H3 sigma shift: video={sv:g}, audio={sa:g}"
        if note:
            report += f"\n{note}"
        if ratio_lock:
            report += f"\nratio_lock active (ratio={float(ratio):g})"
        return io.NodeOutput(m, report)
