"""H3 Timeline Mapper - builds the frame ledger (owner A20 Phase 1 slice 2, ledger L9.74; Revision 2 Parts 6.1 / 8.4).

One authoritative map between the source clip's frames and the output timeline: trim, reverse, frame-rate
conversion and the global frame numbering, plus a planner's segments when given. It describes; it moves no pixels.
Every consumer (images, masks, tracks, audio) reads the same ledger, so a reverse or a retime applies to every
frame-indexed signal at once (Part 5).

Engine: the vendored C2C core (mmx_vendor/c2c_core/timeline/ledger.py, Apache-2.0); this node is GPL-3.0.
Design: docs/evidence/A20/phase1/DESIGN_slice2_frame_ledger.md in the workspace.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from comfy_api.latest import io

_PKG = Path(__file__).resolve().parents[1]
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from mmx_vendor.c2c_core.timeline import ledger as L  # noqa: E402


def _load(text: str, what: str) -> dict:
    try:
        obj = json.loads(text)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"H3 Timeline Mapper: {what} is not JSON ({exc}).") from None
    if not isinstance(obj, dict):
        raise ValueError(f"H3 Timeline Mapper: {what} must be a JSON object.")
    return obj


def ledger_from_info(info: dict) -> dict:
    """A ledger for every frame an H3 Media Probe info_json describes (its selection, trim included)."""
    first = int(info.get("first_source_frame") or 0)
    n = int(info["frames"])
    audio = None
    if info.get("audio"):
        audio = {"sample_rate": int(info["audio"]["sample_rate"]), "offset_samples": 0, "reverse": False}
    return L.new(range(first, first + n), fps=info.get("fps_exact") or info["fps"], path=info.get("source"),
                 time_base=info.get("time_base"), total_source=int(info.get("source_frames") or first + n), audio=audio)


class MiniMaxH3_TimelineMapper(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MiniMaxH3_TimelineMapper",
            display_name="H3 Timeline Mapper",
            category="MiniMax H3/Media",
            description=(
                "Builds the frame ledger: which source frame every output frame shows, after trim, reverse and "
                "frame-rate conversion, numbered from 'offset'. Reverse and retime are described here once and every "
                "later stage (images, masks, audio) follows the same map. No pixels are touched."
            ),
            inputs=[
                io.String.Input("info_json", optional=True, force_input=True,
                                tooltip="From H3 Media Probe: the ledger covers the whole (trimmed) media."),
                io.String.Input("window_json", optional=True, force_input=True,
                                tooltip="From H3 Frame Window Reader: the ledger covers the window's rows, handles included."),
                io.Int.Input("trim_start", default=0, min=0, max=10_000_000, tooltip="First output frame to keep."),
                io.Int.Input("trim_end", default=0, min=0, max=10_000_000, tooltip="Frame to stop before. 0 = to the end."),
                io.Boolean.Input("reverse", default=False, tooltip="Play backwards. The audio flag flips with it."),
                io.Float.Input("output_fps", default=0.0, min=0.0, max=240.0, step=0.001,
                               tooltip="Convert to this constant frame rate by nearest frame. 0 = keep the source rate."),
                io.Int.Input("offset", default=0, min=0, max=10_000_000, tooltip="Global number of the first output frame."),
                io.String.Input("segments_json", optional=True, force_input=True,
                                tooltip="A planner's segments (a JSON list), attached and checked."),
            ],
            outputs=[
                io.String.Output("ledger_json", tooltip="Contract h3.frame_ledger v1."),
                io.String.Output("report", tooltip="The first and last rows of the map, the operations, and any problem."),
                io.Int.Output("frame_count"),
            ],
        )

    @classmethod
    def execute(cls, info_json=None, window_json=None, trim_start=0, trim_end=0, reverse=False, output_fps=0.0,
                offset=0, segments_json=None) -> io.NodeOutput:
        if bool(info_json) == bool(window_json):
            raise ValueError("H3 Timeline Mapper: connect exactly one of 'info_json' (H3 Media Probe) or "
                             "'window_json' (H3 Frame Window Reader).")
        if window_json:
            led = L.from_window(_load(window_json, "window_json"))
        else:
            led = ledger_from_info(_load(info_json, "info_json"))
        if int(trim_start) or int(trim_end):
            led = L.trim(led, int(trim_start), int(trim_end) if int(trim_end) else None)
        if float(output_fps) > 0:
            led = L.retime(led, float(output_fps))
        if reverse:
            led = L.reverse(led)
        if int(offset):
            led = L.offset(led, int(offset))
        if segments_json:
            segs = json.loads(segments_json)
            if not isinstance(segs, list):
                raise ValueError("H3 Timeline Mapper: segments_json must be a JSON list of segments.")
            led = L.add_segments(led, segs)
        problems = L.validate(led)
        report = "\n".join([L.table(led), "", "ops: " + (", ".join(led["ops"]) or "none"),
                            f"frames: {led['output']['frames']} at {led['output']['fps']} fps, "
                            f"global {led['output']['offset']}..{led['output']['offset'] + led['output']['frames'] - 1}",
                            "problems: " + ("; ".join(problems) if problems else "none")])
        return io.NodeOutput(json.dumps(led), report, int(led["output"]["frames"]))


NODE_CLASS_MAPPINGS = {"MiniMaxH3_TimelineMapper": MiniMaxH3_TimelineMapper}
