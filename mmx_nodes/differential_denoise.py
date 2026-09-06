"""MiniMaxH3_DifferentialDenoise — soft-edged per-step denoise masks.

Original implementation, GPL-3.0. Written from docs/cleanroom_maskvid.md §A1.
"""

from __future__ import annotations

import sys
from pathlib import Path

from comfy_api.latest import io

_PKG = Path(__file__).resolve().parents[1]
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from mmx_utils.differential_denoise import schedule_threshold, soft_step_mask


class MiniMaxH3_DifferentialDenoise(io.ComfyNode):
    @classmethod
    def _inputs(cls):
        ramp = io.Float.Input(
            "softness",
            default=0.2, min=0.0, max=1.0, step=0.01,
            tooltip="Ramp width in mask-value units.\n\n0 = hard threshold, identical to stock differential diffusion.\n0.15-0.3 = a soft edge that tracks your mask's own feather; the useful range for video.\n1 = no schedule at all, blends by the raw mask.",
        )
        blend = io.Float.Input(
            "strength",
            default=1.0, min=0.0, max=1.0, step=0.01, optional=True,
            tooltip="Blend the scheduled mask back toward the raw one. 1 = full schedule, 0 = ignore the schedule entirely. Dial down when the schedule is fighting the composite.",
        )
        return [io.Model.Input("model"), ramp, blend]

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MiniMaxH3_DifferentialDenoise",
            display_name="H3 Differential Denoise (soft edge)",
            category="MiniMax H3/Sampling",
            description=(
                "Differential diffusion whose PER-STEP masks keep a soft edge instead "
                "of the usual hard binary threshold.\n\n"
                "Stock differential diffusion thresholds your greyscale mask at every "
                "step, so however carefully you feathered it, each intermediate mask "
                "is razor sharp. On video that edge lands on a different pixel each "
                "frame and crawls. This ramps instead, and because a ramp of width w "
                "in mask-VALUE space spans w/|gradient| in PIXEL space, the softness "
                "follows the blur already in your mask: feathered stays feathered, "
                "sharp stays sharp. No spatial blur, so it stays temporally stable."
            ),
            inputs=cls._inputs(),
            outputs=[
                io.Model.Output("model"),
                io.String.Output("report"),
            ],
        )

    @classmethod
    def fingerprint_inputs(cls, model, softness, strength=1.0):
        return f"{id(model)}|{softness}|{strength}"

    @classmethod
    def execute(cls, model, softness, strength=1.0) -> io.NodeOutput:
        patched = model.clone()

        def _denoise_mask(sigma, denoise_mask, extra_options):
            inner = extra_options["model"].inner_model
            sampling = inner.model_sampling
            sigmas = extra_options["sigmas"]
            threshold = schedule_threshold(
                sigma[0], sigmas[0], sigmas[-1],
                sampling.timestep,
                sigma_floor=getattr(sampling, "sigma_min", None),
            )
            return soft_step_mask(denoise_mask, threshold, softness, strength)

        patched.set_model_denoise_mask_function(_denoise_mask)

        if softness <= 0:
            note = "softness=0 - hard threshold, identical to stock differential diffusion."
        elif softness >= 1:
            note = "softness=1 - schedule disabled; blending by the raw mask."
        else:
            note = (f"softness={softness:g} - per-step edge spans "
                    f"{softness:g}/|mask gradient| pixels, so it follows your mask's feather.")
        report = f"Differential denoise installed. {note} strength={strength:g}."
        return io.NodeOutput(patched, report)
