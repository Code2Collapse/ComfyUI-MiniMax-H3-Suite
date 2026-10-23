"""H3 Context Window Sampling - patch an H3 model so every denoising step runs as overlapping time windows and fuses
their predictions (L8.36). Original code, GPL-3.0; design: docs/h3_context_window_executor_design.md."""

from __future__ import annotations

import sys
from pathlib import Path

from comfy_api.latest import io

_PKG = Path(__file__).resolve().parents[1]
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from mmx_utils.context_executor import POSITIONS, H3ContextWindows, describe_settings  # noqa: E402

CATEGORY = "MiniMax H3/Long"


class MiniMaxH3_ContextWindowSampling(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MiniMaxH3_ContextWindowSampling",
            display_name="H3 Context Window Sampling (experimental)",
            category=CATEGORY,
            description=(
                "Experimental. Patches an H3 model so every denoising step splits the clip into overlapping time "
                "windows, runs the model once per window and cross-fades the predictions. The whole clip stays in one "
                "sampling pass (unlike H3 Continuation, which renders segments one after the other); attention runs "
                "per window, so its cost grows with the window, not with the clip. Windows start on 17-frame cycles "
                "and are legal H3 lengths; the audio is cut at the same moments. Masks and core's H3 Fun ControlNet "
                "are cut per window. Feed the patched model to any H3 sampler. Image quality over long clips is not "
                "measured yet."),
            search_aliases=["context windows", "context window sampling", "long video", "temporal tiling",
                            "sliding window", "h3 long"],
            inputs=[
                io.Model.Input("model", tooltip="The MiniMax H3 diffusion model to patch."),
                io.Int.Input("window_frames", default=141, min=5, max=100_000,
                             tooltip="Frames per window at 24 fps, snapped up to a legal H3 length (17k + 5). A clip "
                                     "this long or shorter runs untouched. 141 is about 6 seconds."),
                io.Int.Input("overlap_frames", default=39, min=5, max=100_000,
                             tooltip="Smallest overlap between neighbouring windows, snapped up to 17m + 5 frames so "
                                     "every window starts on a 17-frame cycle. More overlap gives a softer transition "
                                     "and costs time. With 141 / 39 the stride is 102 frames and the audio edges are "
                                     "exact."),
                io.Combo.Input("positions", options=list(POSITIONS), default=POSITIONS[0],
                               tooltip="global: every window keeps the clip's own time positions, so it predicts "
                                       "what the whole clip would there (recommended). local (experimental): every "
                                       "window is laid out as a clip of its own, keyframes inside it re-indexed and "
                                       "the others left out - for A/B tests only."),
            ],
            outputs=[
                io.Model.Output(display_name="model", tooltip="The patched model: feed it to an H3 sampler."),
                io.String.Output(display_name="report"),
            ],
        )

    @classmethod
    def execute(cls, model, window_frames=141, overlap_frames=39, positions=POSITIONS[0]) -> io.NodeOutput:
        report = describe_settings(window_frames, overlap_frames, positions)
        patched = model.clone()
        inner = patched.model_options.get("model_function_wrapper")
        patched.set_model_unet_function_wrapper(H3ContextWindows(window_frames, overlap_frames, positions,
                                                                 inner=inner))
        if inner is not None:
            report += f" Runs inside the model function wrapper that was already set ({type(inner).__name__})."
        return io.NodeOutput(patched, report, ui={"text": [report]})


__all__ = ["MiniMaxH3_ContextWindowSampling"]
