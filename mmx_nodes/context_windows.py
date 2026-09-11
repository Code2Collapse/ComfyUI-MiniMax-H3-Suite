"""MiniMaxH3_ContextWindows — plan unbounded-length H3 generation.

Original implementation, GPL-3.0. The H3 arithmetic it encodes was
cross-checked against mmx_utils/h3_constants.py (align_frame_count,
video_latent_t) and against
third_party/ComfyUI-MiniMax-H3-LongMedia/temporal_positioning.py
(H3_OUTPUT_FPS 24, H3_TEMPORAL_ROPE_HZ 40).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from comfy_api.latest import io

_PKG = Path(__file__).resolve().parents[1]
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from mmx_utils.context_windows import (  # noqa: E402
    ContextWindowError,
    plan_context_windows,
    plan_report,
)


class MiniMaxH3_ContextWindows(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MiniMaxH3_ContextWindows",
            display_name="H3 Context Windows (long / infinite length)",
            category="MiniMax H3/Long",
            description=(
                "Plan a long clip as overlapping H3 passes.\n\n"
                "H3 will not render an arbitrarily long shot in one go, so length "
                "comes from tiling it with windows that overlap and are then "
                "stitched. The tiling is not free choice: a legal window is "
                "n %% 17 == 5 frames, every window after the first must start on a "
                "17-frame cycle boundary, and each one must be offset in H3's "
                "40 Hz temporal RoPE by start_frame * 40/24 or the model thinks "
                "every pass begins at t=0 and the seams drift apart.\n\n"
                "This node does that arithmetic, tells you what it had to change "
                "and why, and hands the schedule to the nodes that sample and "
                "stitch. Nothing here touches weights, so it is instant."
            ),
            inputs=[
                io.Int.Input(
                    "total_frames", default=409, min=1, max=1_000_000,
                    tooltip="How long the finished clip should be, in frames at "
                            "24 fps. 409 is about 17 seconds. This is the only "
                            "number you normally set."),
                io.Int.Input(
                    "window_frames", default=81, min=5, max=100_000,
                    tooltip="Frames per pass - how much H3 renders at once, which "
                            "is bounded by VRAM. Snapped up to the nearest legal "
                            "length (n %% 17 == 5) because a window off the grid "
                            "does not tile onto the latent rows at all. 81 fits "
                            "8 GB comfortably; 149 or 217 give longer coherent "
                            "motion if you have the memory."),
                io.Int.Input(
                    "overlap_frames", default=17, min=0, max=100_000,
                    tooltip="Shared frames between neighbouring passes - the "
                            "material the stitch cross-fades over. More overlap "
                            "buys a softer seam and costs render time, since "
                            "overlapped frames are generated twice. It is rounded "
                            "so the stride stays a whole number of 17-frame "
                            "cycles; otherwise later windows stop sharing the "
                            "first window's latent structure and cannot be "
                            "stitched at all."),
                io.Combo.Input(
                    "snap", options=["up", "down"], default="up",
                    tooltip="Which way to move window_frames onto the grid. 'up' "
                            "never drops frames you asked for; 'down' keeps peak "
                            "VRAM below the length you chose."),
            ],
            outputs=[
                io.Int.Output(display_name="window_count",
                              tooltip="How many passes this plan needs."),
                io.Int.Output(display_name="window_frames",
                              tooltip="Legal window length after snapping - feed "
                                      "this to the sampler, not your raw request."),
                io.Int.Output(display_name="stride_frames",
                              tooltip="Frames advanced per pass (window minus overlap)."),
                io.String.Output(display_name="schedule_json",
                                 tooltip="The full plan: per-window start, end, "
                                         "latent rows and RoPE offset. Feed to the "
                                         "sampling/stitching nodes."),
                io.String.Output(display_name="report",
                                 tooltip="Plain-English summary, including anything "
                                         "that had to be adjusted and why."),
            ],
        )

    @classmethod
    def fingerprint_inputs(cls, total_frames, window_frames, overlap_frames, snap):
        return f"{total_frames}|{window_frames}|{overlap_frames}|{snap}"

    @classmethod
    def execute(cls, total_frames, window_frames, overlap_frames, snap) -> io.NodeOutput:
        try:
            plan = plan_context_windows(
                total_frames=total_frames,
                window_frames=window_frames,
                overlap_frames=overlap_frames,
                snap=snap,
            )
        except ContextWindowError as exc:
            # An authoring mistake in a widget, not an internal fault: surface it
            # as a sentence rather than a traceback.
            raise ValueError(f"H3 Context Windows: {exc}") from exc

        report = plan_report(plan)
        schedule = json.dumps(plan)

        # Only NodeOutput.ui reaches the browser; socket values never do. The
        # timeline widget cannot draw the tiling without this.
        return io.NodeOutput(
            plan["window_count"],
            plan["window_frames"],
            plan["stride_frames"],
            schedule,
            report,
            ui={"mmx_context_plan": [schedule]},
        )
