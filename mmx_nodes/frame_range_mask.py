"""MiniMaxH3_FrameRangeMask — author a video mask directly from a frame list.

Original implementation, GPL-3.0. Written from docs/cleanroom_maskvid.md §A2.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

from comfy_api.latest import io

_PKG = Path(__file__).resolve().parents[1]
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from mmx_utils.frame_ranges import FrameRangeError, parse_frame_ranges_detailed


class MiniMaxH3_FrameRangeMask(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MiniMaxH3_FrameRangeMask",
            display_name="H3 Frame Range Mask",
            category="MiniMax H3/Spine",
            description=(
                "Build a video MASK straight from a frame list - selected frames are "
                "fully masked, the rest empty.\n\n"
                "Saves building the batch elsewhere and hoping the frame count lines "
                "up. Python slice semantics, so it behaves the way list indexing "
                "already does: the stop frame is EXCLUDED, negatives count back from "
                "the end, and a range that overruns the clip clamps instead of failing."
            ),
            inputs=[
                io.String.Input(
                    "frames", default="0:end", multiline=True,
                    tooltip="Comma- or newline-separated frames and ranges:\n"
                            "  12         a single frame\n"
                            "  0:24       frames 0-23 (stop EXCLUDED, as in Python)\n"
                            "  0:24:2     every second frame\n"
                            "  100:end    from 100 to the last frame\n"
                            "  -5         the fifth frame from the end\n"
                            "Overlaps collapse. A RANGE that overruns the clip clamps; "
                            "a single FRAME outside it is an error, because that is a "
                            "typo rather than an intent.",
                ),
                io.Int.Input("frame_count", default=81, min=1, max=100_000,
                             tooltip="Frames in the clip this mask must match."),
                io.Int.Input("width", default=832, min=16, max=8192, step=8),
                io.Int.Input("height", default=480, min=16, max=8192, step=8),
                io.Boolean.Input(
                    "invert", default=False, optional=True,
                    tooltip="Mask the frames you did NOT list instead."),
            ],
            outputs=[
                io.Mask.Output("mask"),
                io.String.Output("report"),
            ],
        )

    @classmethod
    def fingerprint_inputs(cls, frames, frame_count, width, height, invert=False):
        return f"{frames}|{frame_count}|{width}|{height}|{invert}"

    @classmethod
    def execute(cls, frames, frame_count, width, height, invert=False) -> io.NodeOutput:
        try:
            parsed = parse_frame_ranges_detailed(frames, frame_count)
        except FrameRangeError as exc:
            # Surfaced as a plain sentence: this is an authoring mistake in a text
            # box, not an internal fault, and a traceback would bury it.
            raise ValueError(f"H3 Frame Range Mask: {exc}") from exc

        selected = parsed["frames"]
        n = int(frame_count)
        mask = torch.zeros(n, int(height), int(width), dtype=torch.float32)
        if selected:
            mask[torch.tensor(selected, dtype=torch.long)] = 1.0
        if invert:
            mask = 1.0 - mask

        count = int(mask[:, 0, 0].sum().item())
        segs = ", ".join(f"{d['segment']}({d['count']})" for d in parsed["segments"]) or "(none)"
        report = (
            f"{count}/{n} frames masked at {width}x{height}"
            f"{' (inverted)' if invert else ''}.\n"
            f"segments: {segs}"
        )
        return io.NodeOutput(mask, report)
