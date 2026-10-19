"""H3 Continuation session: on-disk state for resumable, unbounded H3 generation (A16 Part IV).

Original implementation, GPL-3.0. Design: docs/h3_continuation_design.md. Pure logic + file I/O, no ComfyUI import, so
the state machine, the global time accounting and the drift metrics are all testable on CPU.

Folder layout (one session):
    session.json                 the state, rewritten atomically after every change
    tails/seg_NNNN.safetensors   the last O frames of each committed segment, as latent (video rows + audio ticks)
    frames/frame_NNNNNN.png      accepted frames, 16-bit PNG, global numbering from 1
    audio/seg_NNNN.wav           accepted audio of each segment, cut on GLOBAL time (float32 WAV)
    pending/seg_NNNN/            a segment awaiting approval (its frames + wav), moved into place on accept
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

FPS = 24
CYCLE = 17
HEAD = 5
ROWS_PER_CYCLE = 5
FRAME_PER_TOKEN = (1, 4, 4, 4, 4)
AUDIO_TICKS_PER_SECOND = 40
OVERLAPS = (39, 22, 56, 5)
STATUSES = ("running", "awaiting approval", "accepted", "failed", "rejected")
VERSION = 1


class SessionError(ValueError):
    """Something about the session the user has to fix, said in plain English."""


# ── grid ────────────────────────────────────────────────────────────────────────────────────────────────────────────

def frames_for_rows(rows: int) -> int:
    return int(sum(FRAME_PER_TOKEN[k % 5] for k in range(int(rows))))


def rows_for_frames(frames: int) -> int | None:
    """Latent rows covering exactly `frames` from a cycle start, or None if not a whole number of rows."""
    k, covered = 0, 0
    while covered < frames:
        covered += FRAME_PER_TOKEN[k % 5]
        k += 1
    return k if covered == frames else None


def ticks_for_frames(frames: int) -> int:
    return int(round(int(frames) / FPS * AUDIO_TICKS_PER_SECOND))


def seam_offset_ms(overlap: int) -> float:
    """How much later the audio context ends than the video context for this overlap (0 for multiples of 3)."""
    return (ticks_for_frames(overlap) / AUDIO_TICKS_PER_SECOND - overlap / FPS) * 1000.0


def samples_at(frame: int, sr: int) -> int:
    """Global audio sample index of a global frame boundary - the ONE rounding, so nothing accumulates."""
    return int(round(int(frame) / FPS * int(sr)))


# ── atomic files ────────────────────────────────────────────────────────────────────────────────────────────────────

def atomic_write_json(path: str, data: dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=1)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_name(name: str) -> str:
    n = re.sub(r"[^A-Za-z0-9._-]+", "_", (name or "").strip()).strip("._")
    if not n:
        raise SessionError("Give the session a name (letters, numbers, - and _).")
    return n[:80]


# ── the session ─────────────────────────────────────────────────────────────────────────────────────────────────────

@dataclass
class Plan:
    segment: dict
    tail: dict | None            # {"video": [1,24,R,h,w], "audio": [1,32,2,A]} of the last accepted segment


class Session:
    def __init__(self, root: str):
        self.root = root
        self.path = os.path.join(root, "session.json")
        self.state: dict[str, Any] = {}

    # -- open / create
    @classmethod
    def open(cls, base_dir: str, name: str, *, restart: bool = False) -> "Session":
        root = os.path.join(base_dir, safe_name(name))
        if restart and os.path.isdir(root):
            os.replace(root, root + time.strftime("_archived_%Y%m%d_%H%M%S"))
        s = cls(root)
        if os.path.isfile(s.path):
            try:
                with open(s.path, encoding="utf-8") as fh:
                    s.state = json.load(fh)
            except (OSError, json.JSONDecodeError) as exc:
                raise SessionError(f"The session file {s.path} is unreadable ({exc}). Restart the session or "
                                   "restore the file.") from exc
            if s.state.get("version") != VERSION:
                raise SessionError(f"{s.path} was written by another version of this node.")
        else:
            for sub in ("tails", "frames", "audio", "pending"):
                os.makedirs(os.path.join(root, sub), exist_ok=True)
            s.state = {"version": VERSION, "name": safe_name(name), "created": time.time(), "fps": FPS,
                       "accepted_frames": 0, "audio_samples": 0, "sample_rate": None, "geometry": None,
                       "segments": [], "baseline": None}
            s.save()
        return s

    def save(self) -> None:
        self.state["updated"] = time.time()
        atomic_write_json(self.path, self.state)

    # -- queries
    @property
    def segments(self) -> list[dict]:
        return self.state["segments"]

    def by_status(self, status: str) -> list[dict]:
        return [g for g in self.segments if g["status"] == status]

    def last_accepted(self) -> dict | None:
        acc = self.by_status("accepted")
        return acc[-1] if acc else None

    def get(self, seg_id: int) -> dict:
        for g in self.segments:
            if g["id"] == seg_id:
                return g
        raise SessionError(f"Segment {seg_id} is not in this session.")

    # -- planning the next segment
    def plan_next(self, segment_frames: int, overlap: int, geometry: dict, fingerprint: str,
                  decision: str = "keep waiting") -> Plan:
        if rows_for_frames(segment_frames) is None or segment_frames % CYCLE != HEAD:
            raise SessionError(f"The segment is {segment_frames} frames; H3 segments must be 17k+5 frames.")
        if overlap not in OVERLAPS:
            raise SessionError(f"overlap {overlap} is not one of {OVERLAPS} (whole H3 latent steps).")
        if overlap >= segment_frames:
            raise SessionError(f"overlap {overlap} must be shorter than the segment ({segment_frames} frames).")
        if self.state["geometry"] is None:
            self.state["geometry"] = geometry
        elif self.state["geometry"] != geometry:
            raise SessionError(f"This session was started at {self.state['geometry']} and this segment is "
                               f"{geometry}: a latent tail cannot be resized. Use the same size, or restart.")
        # a crash left a segment running: it failed, its interval is generated again
        for g in self.by_status("running"):
            g["status"] = "failed"
            g["note"] = "was still running when the next segment started (crash or cancel); generating it again"
        pending = self.by_status("awaiting approval")
        if pending:
            p = pending[-1]
            if decision.startswith("accept"):
                self._accept_pending(p)
            elif decision.startswith("reject"):
                self._reject_pending(p)
            else:
                self.save()
                raise SessionError(f"Segment {p['id']} (frames {p['start']}-{p['end'] - 1}) is awaiting approval: "
                                   f"{p.get('note', '')}. Look at it in {os.path.join(self.root, 'pending')} and set "
                                   "'decision' to accept or reject.")
        last = self.last_accepted()
        start = 0 if last is None else self.state["accepted_frames"] - overlap
        seg_id = (self.segments[-1]["id"] + 1) if self.segments else 0
        retry_of = next((g["id"] for g in reversed(self.segments)
                         if g["status"] in ("failed", "rejected") and g["start"] == start), None)
        seg = {"id": seg_id, "parent": None if last is None else last["id"], "start": start,
               "end": start + segment_frames, "overlap": 0 if last is None else overlap,
               "status": "running", "fingerprint": fingerprint, "created": time.time(),
               "retry_of": retry_of}
        self.segments.append(seg)
        self.save()
        return Plan(segment=seg, tail=None if last is None else self.load_tail(last))

    # -- tails
    def tail_path(self, seg_id: int) -> str:
        return os.path.join(self.root, "tails", f"seg_{seg_id:04d}.safetensors")

    def store_tail(self, seg: dict, video: torch.Tensor, audio: torch.Tensor) -> None:
        from safetensors.torch import save_file
        path = self.tail_path(seg["id"])
        tmp = path + ".tmp"
        save_file({"video": video.contiguous().cpu(), "audio": audio.contiguous().cpu()}, tmp,
                  metadata={"format": "h3_continuation_tail_v1"})
        os.replace(tmp, path)
        seg["tail_file"] = os.path.basename(path)
        seg["tail_sha256"] = sha256_file(path)

    def load_tail(self, seg: dict) -> dict:
        from safetensors.torch import load_file
        path = self.tail_path(seg["id"])
        if not os.path.isfile(path):
            raise SessionError(f"The stored tail of segment {seg['id']} is missing ({path}).")
        if sha256_file(path) != seg.get("tail_sha256"):
            raise SessionError(f"The stored tail of segment {seg['id']} does not match its checksum: the file was "
                               "changed or damaged. Restart the session from scratch or restore the file.")
        data = load_file(path)
        return {"video": data["video"], "audio": data["audio"]}

    # -- approval
    def _pending_dir(self, seg_id: int) -> str:
        return os.path.join(self.root, "pending", f"seg_{seg_id:04d}")

    def _accept_pending(self, seg: dict) -> None:
        d = self._pending_dir(seg["id"])
        if not os.path.isdir(d):
            raise SessionError(f"Segment {seg['id']} awaits approval but {d} is gone.")
        self._place(seg, d)
        shutil.rmtree(d, ignore_errors=True)
        seg["status"] = "accepted"
        seg["note"] = (seg.get("note", "") + " - accepted by the user").strip(" -")
        self.save()

    def _reject_pending(self, seg: dict) -> None:
        shutil.rmtree(self._pending_dir(seg["id"]), ignore_errors=True)
        seg["status"] = "rejected"
        seg["note"] = (seg.get("note", "") + " - rejected by the user; generating it again").strip(" -")
        self.save()

    def _place(self, seg: dict, src_dir: str) -> None:
        """Move a committed segment's frames / audio from src_dir into the output, in global numbering."""
        a, b = seg["out_frames"]
        for i, f in enumerate(sorted(n for n in os.listdir(src_dir) if n.endswith(".png"))):
            os.replace(os.path.join(src_dir, f), os.path.join(self.root, "frames", f"frame_{a + i + 1:06d}.png"))
        wav = os.path.join(src_dir, "audio.wav")
        if os.path.isfile(wav):
            os.replace(wav, os.path.join(self.root, "audio", f"seg_{seg['id']:04d}.wav"))
        self.state["accepted_frames"] = b
        if seg.get("audio_samples_end") is not None:
            self.state["audio_samples"] = seg["audio_samples_end"]


# ── committing a sampled segment ───────────────────────────────────────────────────────────────────────────────────

def new_frame_range(seg: dict, accepted_frames: int) -> tuple[int, int]:
    """Global frames this segment ADDS: everything after what is already accepted."""
    a = accepted_frames if seg["parent"] is not None else 0
    if seg["parent"] is not None and seg["start"] + seg["overlap"] != accepted_frames:
        raise SessionError(f"Segment {seg['id']} starts at frame {seg['start']} with {seg['overlap']} frames of "
                           f"overlap but {accepted_frames} frames are accepted: the session moved on since it was "
                           "planned. Queue it again.")
    return a, seg["end"]


def write_png16(path: str, frame: torch.Tensor) -> None:
    import cv2
    arr = (frame[..., :3].clamp(0, 1).cpu().numpy() * 65535.0 + 0.5).astype(np.uint16)
    if not cv2.imwrite(path, arr[..., ::-1]):
        raise SessionError(f"Could not write {path}.")


def read_png16(path: str) -> torch.Tensor:
    import cv2
    arr = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise SessionError(f"Could not read {path}.")
    arr = arr[..., ::-1].astype(np.float32) / (65535.0 if arr.dtype == np.uint16 else 255.0)
    return torch.from_numpy(arr.copy())


def write_wav(path: str, waveform: torch.Tensor, sr: int) -> None:
    import soundfile as sf
    data = waveform.detach().cpu().float().numpy()          # [C, N]
    sf.write(path, data.T, int(sr), subtype="FLOAT")


# ── drift metrics ───────────────────────────────────────────────────────────────────────────────────────────────────

def _srgb_to_lab(img: torch.Tensor) -> torch.Tensor:
    """[..., 3] sRGB 0..1 -> CIE Lab (D65)."""
    c = img.clamp(0, 1)
    lin = torch.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    m = torch.tensor([[0.4124564, 0.3575761, 0.1804375], [0.2126729, 0.7151522, 0.0721750],
                      [0.0193339, 0.1191920, 0.9503041]], dtype=lin.dtype)
    xyz = lin @ m.T / torch.tensor([0.95047, 1.0, 1.08883], dtype=lin.dtype)
    eps = 216 / 24389
    f = torch.where(xyz > eps, xyz.clamp(min=1e-12) ** (1 / 3), (24389 / 27 * xyz + 16) / 116)
    return torch.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


def _sharpness(frames: torch.Tensor) -> float:
    g = frames[..., :3].mean(-1)[:, None]
    k = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=g.dtype)[None, None]
    lap = torch.nn.functional.conv2d(g, k)
    return float(lap.var(dim=(1, 2, 3)).mean())


def frame_stats(frames: torch.Tensor) -> dict:
    step = max(1, frames.shape[0] // 16)
    sub = frames[::step]
    lab = _srgb_to_lab(sub[..., :3].float())
    return {"lab_mean": [round(float(v), 4) for v in lab.reshape(-1, 3).mean(0)],
            "lab_std": [round(float(v), 4) for v in lab.reshape(-1, 3).std(0)],
            "sharpness": _sharpness(sub.float())}


def drift_metrics(new_frames: torch.Tensor, baseline: dict | None, previous_last: torch.Tensor | None) -> dict:
    st = frame_stats(new_frames)
    out = {"stats": st}
    if baseline:
        d = [a - b for a, b in zip(st["lab_mean"], baseline["lab_mean"])]
        out["colour_delta_e"] = round(math.sqrt(sum(x * x for x in d)), 3)
        out["sharpness_ratio"] = round(st["sharpness"] / max(1e-12, baseline["sharpness"]), 4)
    diffs = (new_frames[1:, ..., :3] - new_frames[:-1, ..., :3]).abs().mean(dim=(1, 2, 3)) \
        if new_frames.shape[0] > 1 else torch.zeros(0)
    out["frozen_frames"] = int((diffs < 1e-4).sum())
    median = float(diffs.median()) if diffs.numel() else 0.0
    out["median_frame_change"] = round(median, 6)
    if previous_last is not None:
        jump = float((new_frames[0, ..., :3] - previous_last[..., :3].to(new_frames)).abs().mean())
        out["seam_change"] = round(jump, 6)
        out["seam_jump_ratio"] = round(jump / max(1e-6, median), 3)
    return out


def drift_problems(m: dict, colour_max: float, sharp_min: float, seam_max: float) -> list[str]:
    bad = []
    if m.get("colour_delta_e") is not None and m["colour_delta_e"] > colour_max:
        bad.append(f"colour moved {m['colour_delta_e']:.2f} dE from segment 0 (limit {colour_max})")
    if m.get("sharpness_ratio") is not None and m["sharpness_ratio"] < sharp_min:
        bad.append(f"sharpness at {m['sharpness_ratio']:.2f}x segment 0 (limit {sharp_min})")
    if m.get("seam_jump_ratio") is not None and m["seam_jump_ratio"] > seam_max:
        bad.append(f"seam jump {m['seam_jump_ratio']:.1f}x the clip's usual frame change (limit {seam_max})")
    if m.get("frozen_frames", 0) > 2:
        bad.append(f"{m['frozen_frames']} frozen / duplicated frames")
    return bad


# ── audio phase: keep the audio timeline within half a tick of the video, whatever the segment count ────────────────

TAIL_ROWS = 17          # stored video rows: the last 56 frames (largest overlap), starts on a cycle for any 17k+5 clip
TAIL_TICKS = 96         # stored audio ticks: covers a 56-frame overlap (93.3 ticks) plus the phase choice


def audio_origin_seconds(seg: dict) -> float:
    return int(seg.get("audio_origin_ticks", 0)) / AUDIO_TICKS_PER_SECOND


def audio_slack_seconds(frames: int, ticks: int) -> float:
    """How much longer a clip's audio (ticks / 40 s) is than its picture (frames / 24 s); negative = shorter."""
    return ticks / AUDIO_TICKS_PER_SECOND - frames / FPS


def phase_bound_ms(slack_s: float) -> float:
    """Largest |audio origin - video origin| choose_tail_tick can give for a clip with this slack."""
    return max(500.0 / AUDIO_TICKS_PER_SECOND, 1000.0 / AUDIO_TICKS_PER_SECOND - slack_s * 1000.0)


def choose_tail_tick(prev: dict, new_start_frame: int, slack_s: float = 0.0) -> int:
    """First tick of the previous segment's audio that the next segment pins, chosen by GLOBAL time.

    The next segment's audio starts where its video starts (frame `new_start_frame`) only up to the tick grid. The tick
    nearest to that time keeps the audio origin within half a tick (12.5 ms) of the video origin, measured on global
    time, so the error never accumulates. One more rule: the segment's sound must still cover its last frame. A
    segment's audio lasts `slack_s` longer than its picture, so an origin earlier than -slack_s would leave the end of
    the segment without sound; then the next tick is taken (bound: phase_bound_ms).
    """
    local_s = new_start_frame / FPS - audio_origin_seconds(prev)
    nearest = int(round(local_s * AUDIO_TICKS_PER_SECOND))
    earliest = int(math.ceil((local_s - slack_s) * AUDIO_TICKS_PER_SECOND - 1e-9))
    return max(nearest, earliest)


def tail_slice(prev: dict, tail: dict, overlap: int, new_start_frame: int, ticks_total_prev: int,
               slack_s: float = 0.0):
    """(video rows, audio ticks, first tick) the next segment pins, from the stored tail of `prev`."""
    rows = rows_for_frames(overlap)
    v = tail["video"]
    if rows is None or rows > v.shape[2]:
        raise SessionError(f"The stored tail of segment {prev['id']} has {v.shape[2]} latent rows; an overlap of "
                           f"{overlap} frames needs {rows}.")
    if (v.shape[2] - rows) % 5:
        raise SessionError("The overlap would start mid-cycle in the stored tail (internal error, report it).")
    video_rows = v[:, :, v.shape[2] - rows:]
    t0 = choose_tail_tick(prev, new_start_frame, slack_s)
    stored_from = ticks_total_prev - int(tail["audio"].shape[-1])
    if t0 < stored_from or t0 >= ticks_total_prev:
        raise SessionError(f"The audio tail needed from tick {t0} is not stored (ticks {stored_from}..."
                           f"{ticks_total_prev - 1}).")
    audio_ticks = tail["audio"][..., t0 - stored_from:]
    return video_rows, audio_ticks, t0
