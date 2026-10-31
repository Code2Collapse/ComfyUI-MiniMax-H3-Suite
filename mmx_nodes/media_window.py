"""H3 Media Probe + H3 Frame Window Reader (owner A20 Phase 1 slice 1, ledger L9.74; Revision 2 Parts 8.1 / 8.3).

Media comes in through ComfyUI core's own Load Video (upload, file list, trim, crop) or as CustomNodePacks' lazy
C2C_VIDEO handle. Nothing is decoded whole: the probe reads the container's index, the window reader decodes only
the frames it is asked for (plus tracking handles), checked against the memory budget first. Every frame keeps its
source index and its exact presentation time (VFR-safe).

The engine is the Suite's vendored copy of the C2C shared core (mmx_vendor/c2c_core, L9.73): this pack never needs
CustomNodePacks. Design: docs/evidence/A20/phase1/DESIGN_slice1.md in the workspace. Original code, GPL-3.0; the
engine is Apache-2.0 (Code2Collapse, THIRD_PARTY_LICENSES/c2c_core-LICENSE.txt).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

from comfy_api.latest import io

_PKG = Path(__file__).resolve().parents[1]
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from mmx_vendor.c2c_core.video import budget as _budget  # noqa: E402
from mmx_vendor.c2c_core.video import probe as _probe  # noqa: E402
from mmx_vendor.c2c_core.video import reader as _reader  # noqa: E402

C2C_VIDEO = io.Custom("C2C_VIDEO")
WINDOW_CONTRACT = "h3.frame_window"
WINDOW_VERSION = "1.0"
_HANDLE_FIELDS = ("source", "selection", "frame_count", "fps", "width", "height", "selected_indices")


def is_handle(obj) -> bool:
    """A lazy C2C video handle, checked by its fields: CustomNodePacks' own copy of the core makes a different class."""
    return obj is not None and all(hasattr(obj, a) for a in _HANDLE_FIELDS)


def _path_of(video) -> str:
    get = getattr(video, "get_stream_source", None)
    if get is None:
        raise ValueError("H3 Media Probe: 'video' is not a ComfyUI VIDEO - connect core Load Video.")
    src = get()
    if not isinstance(src, str):
        raise ValueError("H3 Media Probe: this video has no file on disk (it is held in memory). Save it to a file "
                         "and load that: lazy decoding reads from the file.")
    if not os.path.isfile(src):
        raise FileNotFoundError(f"H3 Media Probe: the video file is missing: {src}")
    return src


def _core_crop(video, width: int, height: int):
    """Core Load Video's crop as (x, y, w, h) in pixels, normalised by core's own rule, or None."""
    crop = getattr(video, "_VideoFromFile__crop", None)
    if not crop:
        return None
    try:
        from comfy_api.latest._util import normalize_crop_rect
        x, y, w, h = normalize_crop_rect(*crop, width, height)
    except Exception:  # noqa: BLE001 - older core: no crop support, nothing to apply
        return None
    return (int(x), int(y), int(w), int(h)) if w > 0 and h > 0 else None


def _trim_to_frames(video, handle):
    """Core's start / duration trim (seconds) as positions in the handle's selection."""
    window = getattr(video, "get_active_trim_window", None)
    if window is None:
        return handle, None
    start_s, dur_s = window()
    if not start_s and not dur_s:
        return handle, None
    fps = float(handle.fps)
    first = max(0, int(round(start_s * fps)))
    stop = None if not dur_s else first + max(1, int(round(dur_s * fps)))
    return handle.trimmed(first, stop), [start_s, dur_s]


def media_info(handle) -> dict:
    sel = list(handle.selected_indices())
    colour = getattr(handle, "colour", None)
    audio = getattr(handle, "audio", None)
    src = handle.source
    exact = getattr(handle, "index_mode", "auto")
    table = None
    if getattr(src, "kind", "") == "file":
        try:
            table = _probe.get_index_table(src.stat_key, stream_index=handle.stream_index)
        except Exception:  # noqa: BLE001
            table = None
    vfr = bool(table is not None and getattr(table, "pts_step", 1) is None)
    return {
        "source": getattr(src, "path", None) or getattr(src, "pattern", None),
        "kind": getattr(src, "kind", "?"),
        "frames": len(sel),
        "source_frames": int(handle.frame_count),
        "first_source_frame": sel[0] if sel else None,
        "fps": float(handle.fps),
        "fps_exact": str(handle.fps),
        "vfr": vfr,
        "time_base": str(handle.time_base),
        "duration_s": round(len(sel) / float(handle.fps), 6) if handle.fps else None,
        "width": int(handle.width),
        "height": int(handle.height),
        "rotation": int(getattr(handle, "rotation", 0) or 0),
        "pix_fmt": handle.pix_fmt,
        "bit_depth": int(handle.bit_depth),
        "has_alpha": bool(handle.has_alpha),
        "colour": None if colour is None else {k: getattr(colour, k) for k in ("primaries", "transfer", "matrix", "range")},
        "audio": None if audio is None else {"stream": audio.stream_index, "sample_rate": audio.sample_rate,
                                             "channels": audio.channels, "duration_s": audio.duration},
        "index": exact,
    }


class MiniMaxH3_MediaProbe(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MiniMaxH3_MediaProbe",
            display_name="H3 Media Probe",
            category="MiniMax H3/Media",
            description=(
                "Reads a video's streams, frame count, frame rate (constant or variable), size, rotation, colour tags "
                "and audio WITHOUT decoding the pictures, and hands on a lazy handle for H3 Frame Window Reader.\n"
                "Connect core Load Video (its trim is honoured) or a C2C video handle."
            ),
            inputs=[
                io.Video.Input("video", optional=True, tooltip="From core Load Video. Its trim is honoured."),
                C2C_VIDEO.Input("c2c_video", optional=True,
                                tooltip="A lazy C2C video handle (CustomNodePacks' Load Video (C2C)), instead of video."),
            ],
            outputs=[
                C2C_VIDEO.Output("media", tooltip="Lazy handle: no pixels yet. Feed H3 Frame Window Reader."),
                io.String.Output("info_json", tooltip="Streams, frames, fps, VFR, size, colour tags, audio."),
                io.Int.Output("frame_count"),
                io.Float.Output("fps"),
            ],
        )

    @classmethod
    def fingerprint_inputs(cls, video=None, c2c_video=None):
        if video is not None and isinstance(getattr(video, "get_stream_source", lambda: None)(), str):
            p = video.get_stream_source()
            try:
                st = os.stat(p)
                return f"{p}|{st.st_size}|{st.st_mtime_ns}|{getattr(video, 'get_active_trim_window', lambda: '')()}"
            except OSError:
                return p
        return None

    @classmethod
    def execute(cls, video=None, c2c_video=None) -> io.NodeOutput:
        if (video is None) == (c2c_video is None):
            raise ValueError("H3 Media Probe: connect exactly one of 'video' (core Load Video) or 'c2c_video'.")
        trim = None
        if c2c_video is not None:
            if not is_handle(c2c_video):
                raise ValueError("H3 Media Probe: 'c2c_video' is not a C2C video handle.")
            handle = c2c_video
        else:
            handle = _probe.probe_file(_path_of(video))
            handle, trim = _trim_to_frames(video, handle)
        crop = _core_crop(video, int(handle.width), int(handle.height)) if video is not None else None
        if crop:                                         # the handle carries it; the reader crops every frame it decodes
            x, y, w, h = crop
            W, H = float(handle.width), float(handle.height)
            handle = handle.cropped(x / W, y / H, w / W, h / H)
        info = media_info(handle)
        info["core_trim_s"] = trim
        info["core_crop"] = list(crop) if crop else None
        return io.NodeOutput(handle, json.dumps(info, indent=1), int(info["frames"]), float(info["fps"]))


class MiniMaxH3_FrameWindow(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MiniMaxH3_FrameWindow",
            display_name="H3 Frame Window Reader",
            category="MiniMax H3/Media",
            description=(
                "Decodes only a window of a long video - start, count, plus tracking handles before and after - "
                "checked against free memory first. Every frame keeps its source index and exact time (window_json, "
                "contract h3.frame_window v1). Variable frame rate safe."
            ),
            inputs=[
                C2C_VIDEO.Input("media", tooltip="From H3 Media Probe (or CustomNodePacks' Load Video (C2C))."),
                io.Int.Input("start", default=0, min=0, max=10_000_000, tooltip="First frame of the window (0 = first frame of the media)."),
                io.Int.Input("count", default=81, min=0, max=100_000, tooltip="Frames in the window. 0 = to the end (still checked against memory)."),
                io.Int.Input("lookbehind", default=0, min=0, max=1000, tooltip="Extra frames before the window, e.g. for a tracker. Reported separately, never counted as window frames."),
                io.Int.Input("lookahead", default=0, min=0, max=1000, tooltip="Extra frames after the window."),
                io.Int.Input("step", default=1, min=1, max=100, tooltip="Take every Nth frame (inside the window and the handles)."),
            ],
            outputs=[
                io.Image.Output("images", tooltip="Lookbehind + window + lookahead, in order."),
                io.String.Output("window_json", tooltip="Contract h3.frame_window v1: source frames, exact times, which rows are the window."),
                io.Int.Output("core_start", tooltip="Row of the first window frame inside images."),
                io.Int.Output("core_count", tooltip="Number of window frames inside images."),
            ],
        )

    @classmethod
    def execute(cls, media, start=0, count=81, lookbehind=0, lookahead=0, step=1) -> io.NodeOutput:
        if not is_handle(media):
            raise ValueError("H3 Frame Window Reader: 'media' is not a C2C video handle - connect H3 Media Probe.")
        n = len(media.selected_indices())
        start, step = int(start), max(1, int(step))
        if start >= n:
            raise ValueError(f"H3 Frame Window Reader: start {start} is past the end - the media has {n} frames "
                             f"(0..{n - 1}).")
        stop = n if int(count) <= 0 else min(n, start + int(count) * step)
        before = start - min(int(lookbehind), start // step) * step      # whole steps back: stay on the window's grid
        after = min(n, stop + int(lookahead) * step)
        positions = list(range(before, after, step))
        core_first = len(range(before, start, step))
        core_n = len(range(start, stop, step))
        ch = 4 if media.has_alpha else 3
        fits = _budget.frames_that_fit(int(media.width), int(media.height), ch, "float32")
        if len(positions) > fits:
            raise _budget.BudgetError(
                "H3 Frame Window Reader: " + _budget.budget_error_message(
                    asked=len(positions), fits=fits, width=int(media.width), height=int(media.height),
                    channels=ch, dtype="float32")
                + f" Here: lower 'count' (or the handles) to at most {max(0, fits - (len(positions) - core_n))}.")
        arr = _reader.read(media, positions, fmt="float32")
        if arr.shape[-1] == 4:
            arr = arr[..., :3]
        images = torch.from_numpy(np.ascontiguousarray(arr))
        sel = list(media.selected_indices())
        src_frames = [int(sel[p]) for p in positions]
        pts, time_base = _times(media, src_frames)
        window = {
            "c2c_contract": WINDOW_CONTRACT, "c2c_contract_version": WINDOW_VERSION,
            "source": getattr(media.source, "path", None) or getattr(media.source, "pattern", None),
            "positions": positions, "source_frames": src_frames, "pts": pts, "time_base": time_base,
            "fps": float(media.fps), "step": step,
            "lookbehind": core_first, "lookahead": len(positions) - core_first - core_n,
            "core_slice": [core_first, core_first + core_n],
            "clamped": {"lookbehind": int(lookbehind) - core_first if int(lookbehind) > core_first else 0,
                        "lookahead": int(lookahead) - (len(positions) - core_first - core_n)
                        if int(lookahead) > (len(positions) - core_first - core_n) else 0,
                        "count": (int(count) - core_n) if int(count) > core_n else 0},
            "crop": list(media.crop) if getattr(media, "crop", None) else None,
        }
        return io.NodeOutput(images, json.dumps(window), core_first, core_n)


def _times(media, src_frames):
    src = media.source
    if getattr(src, "kind", "") != "file":
        return None, None
    try:
        table = _probe.get_index_table(src.stat_key, stream_index=media.stream_index)
        return [int(table.pts_of(i)) for i in src_frames], str(table.time_base)
    except Exception:  # noqa: BLE001 - sequences / missing tables: frame numbers still identify the frames
        return None, None


NODE_CLASS_MAPPINGS = {
    "MiniMaxH3_MediaProbe": MiniMaxH3_MediaProbe,
    "MiniMaxH3_FrameWindow": MiniMaxH3_FrameWindow,
}
