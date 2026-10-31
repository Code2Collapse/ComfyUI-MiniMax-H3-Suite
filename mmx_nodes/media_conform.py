"""H3 Media Conform (owner A20 Phase 1 slice 6, ledger L9.74; Revision 2 Part 8.2).

Conforms a lazy C2C video to a source range, a frame rate (variable -> constant, judged on the file's REAL
timestamps) and a size (fit / fill / exact, optionally a multiple of N) - still lazy: nothing is decoded until a
window is read. The frame ledger records which source frame every output frame shows. Colour tags pass through
unchanged (colour / HDR conversion is a later stage, said in the report).

Engine: the vendored C2C core (handle operations + c2c_core.timeline). Original node, GPL-3.0.
Design: docs/evidence/A20/phase1/DESIGN_slice6_media_conform.md in the workspace.
"""

from __future__ import annotations

import json
import sys
from fractions import Fraction
from pathlib import Path

from comfy_api.latest import io

_PKG = Path(__file__).resolve().parents[1]
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from mmx_vendor.c2c_core.timeline import ledger as L  # noqa: E402
from mmx_vendor.c2c_core.video import probe as _probe  # noqa: E402

from .media_window import C2C_VIDEO, is_handle  # noqa: E402

FPS_POLICIES = ("keep", "constant")
SIZE_POLICIES = ("keep", "fit", "fill", "exact")


def source_times(media) -> tuple[list[float], str]:
    """Seconds of every source frame: the file's own timestamps, or index / fps for a sequence."""
    src = media.source
    if getattr(src, "kind", "") == "file":
        table = _probe.get_index_table(src.stat_key, stream_index=media.stream_index)
        tb = Fraction(table.time_base)
        first = table.pts_of(0)
        return [float((table.pts_of(i) - first) * tb) for i in range(int(media.frame_count))], "file timestamps"
    fps = Fraction(media.fps)
    return [i / float(fps) for i in range(int(media.frame_count))], "frame index / fps (no timestamps in a sequence)"


def _round(v: float, multiple: int, up: bool) -> int:
    if multiple <= 1:
        return max(1, int(round(v)))
    q = (v / multiple)
    n = int(-(-q // 1)) if up else int(q // 1)
    return max(multiple, n * multiple)


def conform_size(w: int, h: int, policy: str, tw: int, th: int, multiple: int):
    """(crop fractions or None, out width, out height)."""
    if policy == "keep":
        if multiple > 1:
            ow, oh = _round(w, multiple, False), _round(h, multiple, False)
            return (((w - ow) / 2 / w, (h - oh) / 2 / h, ow / w, oh / h) if (ow, oh) != (w, h) else None), ow, oh
        return None, w, h
    if tw <= 0 or th <= 0:
        raise ValueError(f"H3 Media Conform: size_policy '{policy}' needs target_width and target_height.")
    if policy == "exact":
        return None, _round(tw, multiple, False), _round(th, multiple, False)
    scale = min(tw / w, th / h) if policy == "fit" else max(tw / w, th / h)
    if policy == "fit":
        return None, _round(w * scale, multiple, False), _round(h * scale, multiple, False)
    ow, oh = _round(tw, multiple, False), _round(th, multiple, False)           # fill: cover, then centre crop
    cw, ch = ow / scale, oh / scale                                              # the source area that maps to the box
    return ((w - cw) / 2 / w, (h - ch) / 2 / h, cw / w, ch / h), ow, oh


class MiniMaxH3_MediaConform(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MiniMaxH3_MediaConform",
            display_name="H3 Media Conform",
            category="MiniMax H3/Media",
            description=(
                "Conforms a lazy video to a source range, a constant frame rate (variable-rate clips are judged on their "
                "real timestamps) and a size (fit / fill / exact, optionally a multiple of N) - still lazy, nothing "
                "decoded. The frame ledger says which source frame every output frame shows."
            ),
            inputs=[
                C2C_VIDEO.Input("media", tooltip="From H3 Media Probe."),
                io.Int.Input("start", default=0, min=0, max=10_000_000, tooltip="First source frame to keep."),
                io.Int.Input("count", default=0, min=0, max=10_000_000, tooltip="Frames to keep. 0 = to the end."),
                io.Combo.Input("fps_policy", options=list(FPS_POLICIES), default="keep",
                               tooltip="constant: resample to target_fps (repeats or skips frames, by real timestamps)."),
                io.Float.Input("target_fps", default=24.0, min=0.0, max=240.0, step=0.001),
                io.Combo.Input("size_policy", options=list(SIZE_POLICIES), default="keep",
                               tooltip="fit: inside the box, aspect kept. fill: cover the box, centre crop. exact: stretch."),
                io.Int.Input("target_width", default=0, min=0, max=16384),
                io.Int.Input("target_height", default=0, min=0, max=16384),
                io.Int.Input("multiple_of", default=0, min=0, max=256,
                             tooltip="Round the size to a multiple (e.g. 16 or 32). 0 = off."),
            ],
            outputs=[
                C2C_VIDEO.Output("media", tooltip="The conformed lazy handle. Feed H3 Frame Window Reader."),
                io.String.Output("ledger_json", tooltip="Contract h3.frame_ledger v1: output frame -> source frame."),
                io.String.Output("report"),
                io.Int.Output("frame_count"),
            ],
        )

    @classmethod
    def execute(cls, media, start=0, count=0, fps_policy="keep", target_fps=24.0, size_policy="keep", target_width=0,
                target_height=0, multiple_of=0) -> io.NodeOutput:
        if not is_handle(media):
            raise ValueError("H3 Media Conform: 'media' is not a C2C video handle - connect H3 Media Probe.")
        before = (len(media.selected_indices()), str(media.fps), int(media.width), int(media.height))
        h = media
        n = len(h.selected_indices())
        if int(start) >= n:
            raise ValueError(f"H3 Media Conform: start {start} is past the end - the media has {n} frames.")
        if int(start) or int(count):
            h = h.trimmed(int(start), int(start) + int(count) if int(count) else None)
        notes = []
        if fps_policy == "constant":
            if float(target_fps) <= 0:
                raise ValueError("H3 Media Conform: fps_policy 'constant' needs target_fps above 0.")
            times, how = source_times(h)
            h = h.retimed(Fraction(target_fps).limit_denominator(1001000), times)
            notes.append(f"retimed by {how}")
        crop, ow, oh = conform_size(int(h.width), int(h.height), size_policy, int(target_width), int(target_height),
                                    int(multiple_of))
        if crop is not None:
            h = h.cropped(*crop)
        if (ow, oh) != (int(h.width), int(h.height)):
            h = h.scaled(ow, oh)
        led = L.new(list(h.selected_indices()), fps=Fraction(h.fps), path=getattr(h.source, "path", None),
                    total_source=int(media.frame_count))
        problems = L.validate(led)
        colour = getattr(h, "colour", None)
        report = "\n".join([
            f"before: {before[0]} frames @ {before[1]} fps, {before[2]}x{before[3]}",
            f"after:  {len(h.selected_indices())} frames @ {Fraction(h.fps)} fps, {h.width}x{h.height}"
            + (f" (crop {', '.join(f'{v:.4f}' for v in crop)} of the source)" if crop else ""),
            "colour tags carried unchanged: " + (str({k: getattr(colour, k) for k in ("primaries", "transfer", "matrix",
                                                                                     "range")}) if colour else "none")
            + " - colour / HDR conversion is a later stage",
            "audio: " + ("re-timed when the video is written" if getattr(h, "audio", None) else "none"),
            "notes: " + ("; ".join(notes) or "none"),
            "ledger problems: " + ("; ".join(problems) or "none"),
        ])
        return io.NodeOutput(h, json.dumps(led), report, len(h.selected_indices()))


NODE_CLASS_MAPPINGS = {"MiniMaxH3_MediaConform": MiniMaxH3_MediaConform}
