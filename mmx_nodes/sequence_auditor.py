"""H3 Sequence Auditor (owner A20 Phase 1 slice 4, ledger L9.74; Revision 2 Part 8.6).

Checks an image sequence before anything renders from it: missing frames, duplicates, unreadable / empty files,
frames whose size differs, against an expected range and size. Reads only file headers (OpenImageIO, else PIL);
pixels are decoded only for the few contact-sheet samples. The sequence is resolved by the vendored C2C core
(mmx_vendor/c2c_core/video/probe.py: a pattern ####, @@@@, %04d, $F4, a directory, or one frame of the sequence).
Original node, GPL-3.0. Design: docs/evidence/A20/phase1/DESIGN_slice4_sequence_auditor.md in the workspace.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from comfy_api.latest import io

_PKG = Path(__file__).resolve().parents[1]
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from mmx_vendor.c2c_core.video import probe as _probe  # noqa: E402

MANIFEST = "h3.sequence_manifest"
THUMB_W = 160


def frame_number(path: str) -> int | None:
    m = _probe._FRAME_FILE.match(os.path.basename(path))
    return int(m.group("num")) if m else None


def _header(path: str):
    """(width, height, channels) from the file header, or an error sentence."""
    if os.path.getsize(path) == 0:
        return None, "empty file (0 bytes)"
    try:
        import OpenImageIO as oiio  # type: ignore[import-not-found]
        inp = oiio.ImageInput.open(path)
        if inp is None:
            return None, f"unreadable: {oiio.geterror() or 'OpenImageIO could not open it'}"
        try:
            s = inp.spec()
            return (int(s.width), int(s.height), int(s.nchannels)), None
        finally:
            inp.close()
    except ImportError:
        pass
    try:
        from PIL import Image
        with Image.open(path) as im:
            return (int(im.width), int(im.height), len(im.getbands())), None
    except Exception as exc:  # noqa: BLE001
        return None, f"unreadable: {type(exc).__name__}: {str(exc)[:80]}"


def _fingerprint(path: str, full: bool) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        if full:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        else:
            h.update(str(os.path.getsize(path)).encode())
            h.update(f.read(65536))
    return h.hexdigest()


def _thumb(path: str) -> torch.Tensor | None:
    try:
        try:
            import OpenImageIO as oiio  # type: ignore[import-not-found]
            buf = oiio.ImageBuf(path)
            arr = buf.get_pixels(oiio.FLOAT)
            if arr is None or arr.size == 0:
                return None
        except ImportError:
            from PIL import Image
            with Image.open(path) as im:
                arr = np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0
        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim == 2:
            arr = arr[..., None]
        arr = arr[..., :3] if arr.shape[-1] >= 3 else np.repeat(arr[..., :1], 3, axis=-1)
        t = torch.from_numpy(np.ascontiguousarray(arr)).clamp(0, 1).permute(2, 0, 1)[None]
        h = max(1, round(THUMB_W * t.shape[2] / t.shape[3]))
        return F.interpolate(t, size=(h, THUMB_W), mode="area").permute(0, 2, 3, 1)[0]
    except Exception:  # noqa: BLE001 - a frame that cannot be shown is reported by the header check
        return None


def _ranges(nums: list[int]) -> list[list[int]]:
    out: list[list[int]] = []
    for n in sorted(nums):
        if out and n == out[-1][1] + 1:
            out[-1][1] = n
        else:
            out.append([n, n])
    return out


class MiniMaxH3_SequenceAuditor(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MiniMaxH3_SequenceAuditor",
            display_name="H3 Sequence Auditor",
            category="MiniMax H3/Media",
            description=(
                "Checks an image sequence before anything renders from it: missing frames, duplicates, unreadable or "
                "empty files and frames of another size, against an expected range and size. Reads only file headers; "
                "a few frames are decoded for the contact sheet."
            ),
            inputs=[
                io.String.Input("path", default="", tooltip="A pattern (shot.####.exr, %04d, @@@@, $F4), a folder, or "
                                                            "one frame of the sequence."),
                io.Int.Input("expected_first", default=-1, min=-1, max=10_000_000, tooltip="-1 = the first frame found."),
                io.Int.Input("expected_last", default=-1, min=-1, max=10_000_000, tooltip="-1 = the last frame found."),
                io.Int.Input("expected_width", default=0, min=0, max=32768, tooltip="0 = the first frame's width."),
                io.Int.Input("expected_height", default=0, min=0, max=32768, tooltip="0 = the first frame's height."),
                io.Boolean.Input("checksums", default=False,
                                 tooltip="Hash every file completely (slow on big EXRs). Off: size + first 64 KB, enough "
                                         "to catch copied frames."),
                io.Int.Input("contact_frames", default=8, min=0, max=64, tooltip="How many frames to show, spread evenly."),
            ],
            outputs=[
                io.String.Output("manifest_json", tooltip="Contract h3.sequence_manifest v1."),
                io.String.Output("report", tooltip="What is wrong, in sentences."),
                io.Int.Output("valid_count", tooltip="Frames that exist, read, and have the expected size."),
                io.Image.Output("contact_sheet", tooltip="A few frames side by side."),
            ],
        )

    @classmethod
    def fingerprint_inputs(cls, path="", **kw):
        try:
            files = _probe.resolve_sequence(path) if path else ()
            return (len(files), sum(os.path.getmtime(f) + os.path.getsize(f) for f in files), tuple(sorted(kw.items())))
        except Exception:  # noqa: BLE001
            return path

    @classmethod
    def execute(cls, path="", expected_first=-1, expected_last=-1, expected_width=0, expected_height=0,
                checksums=False, contact_frames=8) -> io.NodeOutput:
        if not str(path).strip():
            raise ValueError("H3 Sequence Auditor: 'path' is empty - give a pattern, a folder or one frame.")
        try:
            files = list(_probe.resolve_sequence(str(path).strip()))
        except FileNotFoundError as exc:
            raise ValueError(f"H3 Sequence Auditor: {exc}") from None
        by_num: dict[int, list[str]] = {}
        unnumbered = []
        for f in files:
            n = frame_number(f)
            (by_num.setdefault(n, []).append(f) if n is not None else unnumbered.append(f))
        nums = sorted(by_num)
        first = nums[0] if nums else None
        last = nums[-1] if nums else None
        lo = int(expected_first) if int(expected_first) >= 0 else first
        hi = int(expected_last) if int(expected_last) >= 0 else last
        missing = [n for n in range(lo, hi + 1) if n not in by_num] if lo is not None and hi is not None else []
        outside = [n for n in nums if lo is not None and (n < lo or n > hi)]
        unreadable, size_bad, sizes = {}, {}, {}
        for n in nums:
            for f in by_num[n]:
                hdr, err = _header(f)
                if err:
                    unreadable[n] = err
                else:
                    sizes[n] = hdr
        ref = None
        if int(expected_width) and int(expected_height):
            ref = (int(expected_width), int(expected_height))
        elif sizes:
            ref = sizes[min(sizes)][:2]
        for n, s in sizes.items():
            if ref and s[:2] != ref:
                size_bad[n] = f"{s[0]}x{s[1]} (expected {ref[0]}x{ref[1]})"
        groups: dict[str, list[int]] = {}
        for n in nums:
            if n in unreadable:
                continue
            groups.setdefault(_fingerprint(by_num[n][0], bool(checksums)), []).append(n)
        duplicates = [g for g in groups.values() if len(g) > 1]
        clashes = {n: [os.path.basename(f) for f in fs] for n, fs in by_num.items() if len(fs) > 1}
        bad = set(unreadable) | set(size_bad) | set(outside)
        valid = [n for n in nums if n not in bad and lo <= n <= hi] if lo is not None else []
        manifest = {
            "c2c_contract": MANIFEST, "c2c_contract_version": "1.0", "path": str(path), "files": len(files),
            "first": first, "last": last, "expected": [lo, hi], "size": list(ref) if ref else None,
            "missing": _ranges(missing), "duplicates": duplicates, "unreadable": unreadable, "size_mismatch": size_bad,
            "outside_expected": outside, "same_number_twice": clashes, "unnumbered": [os.path.basename(f) for f in unnumbered],
            "valid_ranges": _ranges(valid), "valid_count": len(valid), "checksums": "sha256 full" if checksums else "size + first 64 KB",
        }
        lines = [f"{len(files)} files, frames {first}..{last}" + (f", expected {lo}..{hi}" if (lo, hi) != (first, last) else "")]
        if missing:
            lines.append(f"missing {len(missing)}: " + ", ".join(f"{a}" if a == b else f"{a}-{b}" for a, b in _ranges(missing)))
        if duplicates:
            lines.append("identical content: " + "; ".join(", ".join(map(str, g)) for g in duplicates))
        for n, why in sorted(unreadable.items()):
            lines.append(f"frame {n}: {why}")
        for n, why in sorted(size_bad.items()):
            lines.append(f"frame {n}: size {why}")
        if outside:
            lines.append(f"outside the expected range: {', '.join(map(str, outside[:20]))}")
        if clashes:
            lines.append(f"the same frame number twice: {clashes}")
        if len(lines) == 1:
            lines.append("no problem found")
        picks = []
        if int(contact_frames) and valid:
            k = min(int(contact_frames), len(valid))
            picks = [valid[round(i * (len(valid) - 1) / max(1, k - 1))] for i in range(k)]
        thumbs = [t for t in (_thumb(by_num[n][0]) for n in dict.fromkeys(picks)) if t is not None]
        if thumbs:
            h = max(t.shape[0] for t in thumbs)
            thumbs = [F.pad(t.permute(2, 0, 1), (0, 0, 0, h - t.shape[0])).permute(1, 2, 0) for t in thumbs]
            sheet = torch.cat(thumbs, dim=1)[None]
        else:
            sheet = torch.zeros(1, 64, THUMB_W, 3)
        return io.NodeOutput(json.dumps(manifest), "\n".join(lines), len(valid), sheet)


NODE_CLASS_MAPPINGS = {"MiniMaxH3_SequenceAuditor": MiniMaxH3_SequenceAuditor}
