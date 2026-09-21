"""Pure multishot script / seam / shot-planning logic.

PORTED FROM: third_party/MiniMax-H3-NativeAudio-MusicVideo-Workflow ::
  custom_nodes/ComfyUI-H3-Multishot/h3_multishot_utils.py
"""

from __future__ import annotations

import json
import math
import re
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F

from .h3_constants import FPS
from .h3_grid import snap_frame_count_nearest


def xfade_audio(parts: Sequence[torch.Tensor], sr: int, ms: int = 40) -> torch.Tensor | None:
    """Join independently generated audio with a short equal-power seam fade."""
    if not parts:
        return None
    if len(parts) == 1:
        return parts[0]
    n = max(1, int(sr * ms / 1000.0))
    out = parts[0]
    for nxt in parts[1:]:
        # Never consume more than HALF of either side. Clamping to the full
        # chunk length instead lets a seam fade eat an entire chunk: two
        # 1000-sample parts with a 40ms (1280-sample) fade at 32kHz joined to
        # 1000 samples, destroying half the audio with no error.
        k = min(n, out.shape[-1] // 2, nxt.shape[-1] // 2)
        if k < 8:
            out = torch.cat([out, nxt], dim=-1)
            continue
        t = torch.linspace(0, 1, k, dtype=out.dtype, device=out.device)
        fade_out = torch.cos(t * 3.14159265 / 2)
        fade_in = torch.sin(t * 3.14159265 / 2)
        out = torch.cat(
            [
                out[..., :-k],
                out[..., -k:] * fade_out + nxt[..., :k] * fade_in,
                nxt[..., k:],
            ],
            dim=-1,
        )
    return out


def repair_json(text: str) -> tuple[Any | None, str]:
    """Parse JSON, auto-closing unterminated brackets/quotes."""
    try:
        return json.loads(text), ""
    except json.JSONDecodeError as e:
        first_err = str(e)

    stack, in_str, esc = [], False, False
    for ch in text:
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in "{[":
            stack.append(ch)
        elif ch in "}]":
            if stack and ((ch == "}" and stack[-1] == "{") or (ch == "]" and stack[-1] == "[")):
                stack.pop()

    candidate = text.rstrip()
    fixes: list[str] = []
    if in_str:
        candidate += '"'
        fixes.append("closed an open string")
    if candidate.endswith(","):
        candidate = candidate[:-1]
        fixes.append("dropped a trailing comma")
    cleaned = re.sub(r",(\s*[\]}])", r"\1", candidate)
    if cleaned != candidate:
        candidate = cleaned
        fixes.append("removed comma(s) before a closing bracket")
    for opener in reversed(stack):
        candidate += "}" if opener == "{" else "]"
    if stack:
        fixes.append("added " + "".join("}" if o == "{" else "]" for o in reversed(stack)))
    if not fixes:
        return None, first_err
    try:
        return json.loads(candidate), ", ".join(fixes)
    except json.JSONDecodeError as e:
        return None, str(e)


def parse_script(text: str) -> list[str]:
    """JoyEcho script -> list of shot prompts."""
    text = (text or "").strip()
    shots: list[str] = []
    if text.startswith("{") or (text.startswith("[") and len(text) > 1 and text[1] in "[{\""):
        data, repaired = repair_json(text)
        if data is None:
            raise ValueError(
                f"H3 script looks like JSON but does not parse ({repaired}). "
                f"Auto-repair of unclosed brackets/quotes was attempted and "
                f"failed. Common cause: a doubled {{ on the first lines, or a "
                f"missing comma between prompts. Fix the script or use plain "
                f"prompts separated by --- lines."
            )
        if isinstance(data, dict):
            shots = [str(p) for p in data.get("prompts", [])]
        elif isinstance(data, list):
            shots = [str(p) for p in data]
    if not shots:
        shots = [b.strip().replace('\\"', '"') for b in re.split(r"(?m)^---\s*$", text) if b.strip()]
    if not shots:
        shots = [text]
    return shots


def frames_per_shot_from_seconds(seconds_per_shot: float, fps: float = FPS) -> int:
    """Snap approximate shot duration to H3's 17n+5 grid (nearest, per upstream multishot)."""
    requested = max(5, int(round(float(seconds_per_shot) * fps)))
    return snap_frame_count_nearest(requested)


def shot_count_for_audio(
    audio: Mapping[str, Any],
    frames_per_shot: int,
    fps: float = FPS,
) -> int:
    audio_seconds = audio["waveform"].shape[-1] / float(audio["sample_rate"])
    target_frames = max(1, int(math.floor(audio_seconds * fps + 1e-8)))
    hop = max(1, int(frames_per_shot) - 1)
    return max(1, int(math.ceil((target_frames - frames_per_shot) / float(hop))) + 1)


def normalize_shot_list(
    shots: list[str],
    shot_count: int,
) -> tuple[list[str], list[str]]:
    """Pad/truncate prompts to *shot_count*; return (shots, console_notes)."""
    notes: list[str] = []
    n = int(shot_count)
    if len(shots) > n:
        notes.append(f"dropping {len(shots) - n} extra script prompt(s) (shot_count={n})")
        shots = shots[:n]
    while len(shots) < n:
        notes.append(
            f"shot {len(shots) + 1} continues the last prompt "
            f"(script had fewer prompts than shot_count)"
        )
        shots.append(shots[-1])
    return shots, notes


def split_script_for_workflow(
    script: str,
    shot_count: int = 0,
    workflow_shots: int = 3,
) -> tuple[tuple[str, str, str, str], int, str]:
    """H3ScriptSplit logic: up to 4 outputs + count + report."""
    shots = parse_script(script)
    notes: list[str] = []
    if shot_count and shot_count > 0:
        if len(shots) > shot_count:
            notes.append(f"shot_count={shot_count}: dropping {len(shots) - shot_count} extra script shot(s)")
            shots = shots[:shot_count]
        while len(shots) < shot_count:
            shots.append(shots[-1])
    n = len(shots)
    if n < workflow_shots:
        notes.append(
            f"script has {n} shot(s); a {workflow_shots}-shot graph will render the last "
            f"prompt {workflow_shots - n} extra time(s) as a continuation"
        )
    elif n > workflow_shots:
        notes.append(
            f"script has {n} shots; a {workflow_shots}-shot graph DROPS shot(s) 4+"
        )
    while len(shots) < 4:
        shots.append(shots[-1])
    report = f"parsed {n} shot(s)" + ("; " + "; ".join(notes) if notes else "")
    return (shots[0], shots[1], shots[2], shots[3]), n, report


def exact_chunk_window(
    shot_index: int,
    frames_per_shot: int,
    sample_rate: int,
    fps: float = FPS,
) -> tuple[int, int]:
    """Return (start_sample, length_samples) for one shot's exact-audio chunk."""
    start_frame = 0 if shot_index == 0 else shot_index * (frames_per_shot - 1) - 1
    start_sample = max(0, int(round(start_frame * sample_rate / fps)))
    length_samples = int(round(frames_per_shot * sample_rate / fps))
    return start_sample, length_samples


def slice_exact_audio_chunk(
    source: torch.Tensor,
    source_rate: int,
    shot_index: int,
    frames_per_shot: int,
    target_rate: int | None = None,
    fps: float = FPS,
) -> tuple[torch.Tensor, int, str]:
    """Slice (and optionally resample) one shot's audio chunk; pad tail, never loop."""
    from .native_audio import resample_waveform

    wav = source[:1].clone()
    sr = int(source_rate)
    notes: list[str] = []
    if target_rate is not None and sr != int(target_rate):
        wav, did = resample_waveform(wav, sr, int(target_rate))
        if did:
            notes.append(f"resampled {sr} -> {target_rate} Hz")
            sr = int(target_rate)
    start, length = exact_chunk_window(shot_index, frames_per_shot, sr, fps=fps)
    end = start + length
    chunk = wav[..., start:end].clone()
    if chunk.shape[-1] < length:
        chunk = F.pad(chunk, (0, length - chunk.shape[-1]))
        notes.append("chunk shorter than shot — zero-padded tail (not looped)")
    return chunk, sr, "; ".join(notes)


def seam_trim_samples(sample_rate: int, fps: float = FPS) -> int:
    """One video-frame of audio samples (1/fps second)."""
    return int(round(sample_rate / fps))


def trim_master_to_soundtrack(
    master_frames: int,
    source_waveform: torch.Tensor,
    sample_rate: int,
    fps: float = FPS,
) -> tuple[int, int, str]:
    """Align master length to soundtrack duration on the 24fps grid."""
    target_frames = max(1, int(math.floor(source_waveform.shape[-1] / float(sample_rate) * fps + 1e-8)))
    target_frames = min(target_frames, int(master_frames))
    target_samples = int(round(target_frames * sample_rate / fps))
    report = f"trimmed master to {target_frames} frames / {target_samples} samples"
    return target_frames, target_samples, report
