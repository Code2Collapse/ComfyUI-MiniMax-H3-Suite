"""H3 Segment Writer + H3 Sequence Assembler (owner A20 Phase 1 slice 5, ledger L9.74; Revision 2 Part 21.4).

A long job never holds the whole video: every run writes its segment to its own chunk file in a job folder, and the
assembler joins the chunks by remuxing their packets (no decode, no re-encode, one packet in memory at a time).
Engine: the vendored C2C core (mmx_vendor/c2c_core/video/writer.py, Apache-2.0). Original nodes, GPL-3.0.
Design: docs/evidence/A20/phase1/DESIGN_slice5_stream_writer.md in the workspace.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

from comfy_api.latest import io

_PKG = Path(__file__).resolve().parents[1]
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from mmx_vendor.c2c_core.video import writer as _w  # noqa: E402

_EXT = {"h264": ".mp4", "prores": ".mov", "ffv1": ".mkv"}
_SEG = re.compile(r"^seg_(\d+)\.(mp4|mov|mkv)$")


def job_folder(job_dir: str, *, create: bool) -> Path:
    """job_dir inside ComfyUI's output folder - never outside it."""
    import folder_paths
    root = Path(folder_paths.get_output_directory()).resolve()
    rel = str(job_dir or "").strip().replace("\\", "/").strip("/")
    if not rel:
        raise ValueError("H3 segment I/O: 'job_dir' is empty - name a folder inside ComfyUI's output folder, "
                         "e.g. h3_jobs/shot010.")
    target = (root / rel).resolve()
    if target != root and root not in target.parents:
        raise ValueError(f"H3 segment I/O: 'job_dir' must stay inside ComfyUI's output folder ({root}); "
                         f"'{job_dir}' points outside it.")
    if create:
        target.mkdir(parents=True, exist_ok=True)
    elif not target.is_dir():
        raise ValueError(f"H3 Sequence Assembler: the job folder does not exist: {target}")
    return target


class MiniMaxH3_SegmentWriter(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MiniMaxH3_SegmentWriter",
            display_name="H3 Segment Writer",
            category="MiniMax H3/Media",
            description=(
                "Writes this run's frames as ONE segment file (seg_00012.mp4 + a .json note) in a job folder inside "
                "ComfyUI's output folder, without keeping them. Run it per segment; H3 Sequence Assembler joins the "
                "segments without re-encoding."
            ),
            is_output_node=True,
            inputs=[
                io.Image.Input("images"),
                io.String.Input("job_dir", default="h3_jobs/job", tooltip="A folder inside ComfyUI's output folder."),
                io.Int.Input("segment", default=0, min=0, max=99999, tooltip="This segment's number (file seg_00012)."),
                io.Float.Input("fps", default=24.0, min=1.0, max=240.0, step=0.001),
                io.Combo.Input("codec", options=list(_EXT), default="h264",
                               tooltip="h264 (.mp4, even size), prores 422 HQ 10-bit (.mov), ffv1 lossless (.mkv)."),
                io.Int.Input("quality", default=90, min=0, max=100, tooltip="h264 only: higher = better, bigger."),
                io.Boolean.Input("overwrite", default=True, tooltip="Replace this segment's file if it exists."),
            ],
            outputs=[
                io.Image.Output("images", tooltip="The same frames, passed on."),
                io.String.Output("chunk_json", tooltip="The file written: path, frames, codec, size."),
            ],
        )

    @classmethod
    def execute(cls, images, job_dir="h3_jobs/job", segment=0, fps=24.0, codec="h264", quality=90,
                overwrite=True) -> io.NodeOutput:
        if images.ndim == 3:
            images = images.unsqueeze(0)
        folder = job_folder(job_dir, create=True)
        path = folder / f"seg_{int(segment):05d}{_EXT[codec]}"
        if path.exists() and not overwrite:
            raise ValueError(f"H3 Segment Writer: {path.name} already exists in {folder} (overwrite is off).")
        for other in _EXT.values():                       # a re-run with another codec must not leave two versions
            stale = folder / f"seg_{int(segment):05d}{other}"
            if stale != path and stale.exists():
                stale.unlink()
        n, h, w, _c = images.shape
        tmp = path.with_name(path.stem + ".partial" + path.suffix)
        try:
            with _w.StreamWriter(str(tmp), float(fps), int(w), int(h), codec=codec, quality=int(quality)) as wr:
                step = 16
                for s in range(0, int(n), step):              # bounded: one small batch converted at a time
                    wr.write(images[s:s + step])
            os.replace(tmp, path)                             # a crash never leaves a half chunk under the real name
        except _w.WriterError as exc:
            raise ValueError(f"H3 Segment Writer: {exc}") from None
        finally:
            if tmp.exists():
                tmp.unlink()
        info = {"path": str(path), "segment": int(segment), "frames": int(n), "codec": codec, "width": int(w),
                "height": int(h), "fps": float(fps)}
        path.with_suffix(".json").write_text(json.dumps(info, indent=1), encoding="utf-8")
        return io.NodeOutput(images, json.dumps(info))


class MiniMaxH3_SequenceAssembler(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MiniMaxH3_SequenceAssembler",
            display_name="H3 Sequence Assembler",
            category="MiniMax H3/Media",
            description=(
                "Joins the segment files of a job folder (seg_00000, seg_00001, ...) into one video by remuxing - no "
                "decoding, no re-encoding, nothing held in memory. A missing segment number stops it unless allowed."
            ),
            is_output_node=True,
            inputs=[
                io.String.Input("job_dir", default="h3_jobs/job", tooltip="The folder H3 Segment Writer wrote into."),
                io.String.Input("output_name", default="assembled.mp4",
                                tooltip="File name in the job folder; its extension must match the segments' container."),
                io.Int.Input("expect_count", default=0, min=0, max=99999, tooltip="0 = any number of segments."),
                io.Boolean.Input("allow_gaps", default=False, tooltip="Join even if a segment number is missing."),
                io.Boolean.Input("overwrite", default=True),
                io.String.Input("wait_for", optional=True, force_input=True,
                                tooltip="Connect H3 Segment Writer's chunk_json to assemble in the same run, after it."),
            ],
            outputs=[
                io.String.Output("path"),
                io.String.Output("report"),
                io.Int.Output("frame_count"),
            ],
        )

    @classmethod
    def fingerprint_inputs(cls, job_dir="h3_jobs/job", **kw):
        try:
            folder = job_folder(job_dir, create=False)
            return tuple(sorted((p.name, p.stat().st_size, p.stat().st_mtime_ns) for p in folder.iterdir()
                                if _SEG.match(p.name))) + (tuple(sorted(kw.items())),)
        except Exception:  # noqa: BLE001
            return job_dir

    @classmethod
    def execute(cls, job_dir="h3_jobs/job", output_name="assembled.mp4", expect_count=0, allow_gaps=False,
                overwrite=True, wait_for=None) -> io.NodeOutput:
        folder = job_folder(job_dir, create=False)
        segs = sorted(((int(m.group(1)), folder / name) for name in os.listdir(folder) if (m := _SEG.match(name))))
        if not segs:
            raise ValueError(f"H3 Sequence Assembler: no segment files (seg_00000.mp4 ...) in {folder}.")
        nums = [n for n, _ in segs]
        if len(set(nums)) != len(nums):
            raise ValueError("H3 Sequence Assembler: a segment number appears twice (two containers?) - re-run that "
                             "segment's writer.")
        gaps = [n for n in range(nums[0], nums[-1] + 1) if n not in set(nums)]
        if gaps and not allow_gaps:
            raise ValueError(f"H3 Sequence Assembler: segment(s) {gaps[:10]} are missing in {folder} - write them, or "
                             f"turn on allow_gaps.")
        if int(expect_count) and len(segs) != int(expect_count):
            raise ValueError(f"H3 Sequence Assembler: {len(segs)} segments found, {int(expect_count)} expected.")
        name = os.path.basename(str(output_name).strip()) or "assembled.mp4"
        out = folder / name
        if out.exists() and not overwrite:
            raise ValueError(f"H3 Sequence Assembler: {out.name} already exists (overwrite is off).")
        tmp = out.with_name(out.stem + ".partial" + out.suffix)
        try:
            info = _w.concat([str(p) for _, p in segs], str(tmp))
            os.replace(tmp, out)
        except _w.WriterError as exc:
            raise ValueError(f"H3 Sequence Assembler: {exc}") from None
        finally:
            if tmp.exists():
                tmp.unlink()
        report = (f"{info['chunks']} segments ({nums[0]}..{nums[-1]}) -> {out.name}: {info['frames']} frames, "
                  f"{info['codec']} {info['width']}x{info['height']} @ {info['rate']}, remuxed (no re-encode)"
                  + (f"; missing segments allowed: {gaps}" if gaps else ""))
        return io.NodeOutput(str(out), report, int(info["frames"]))


NODE_CLASS_MAPPINGS = {
    "MiniMaxH3_SegmentWriter": MiniMaxH3_SegmentWriter,
    "MiniMaxH3_SequenceAssembler": MiniMaxH3_SequenceAssembler,
}
