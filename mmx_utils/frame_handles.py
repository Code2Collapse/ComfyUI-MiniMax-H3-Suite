"""Frame-handle padding / trimming with optional audio resync (N7 core logic)."""

from __future__ import annotations

from typing import Callable, Optional

import torch

from .h3_constants import FPS
from .h3_grid import calculate_h3_frames, calculate_next_h3_frames, is_h3_compatible
from .wan_ltx_grid import (
    calculate_ltx2_frames,
    calculate_next_ltx2_frames,
    calculate_next_wan_frames,
    calculate_wan_frames,
    is_ltx2_compatible,
    is_wan_compatible,
)

PADDING_MODES = (
    "disabled",
    "H3 (17n+5)",
    "WAN (4n+1)",
    "LTX2 (8n+1)",
)


def _mode_funcs(padding_mode: str) -> tuple[
    Callable[[int], int],
    Callable[[int], int],
    Callable[[int], bool],
    str,
]:
    if padding_mode == "H3 (17n+5)":
        return calculate_h3_frames, calculate_next_h3_frames, is_h3_compatible, "H3"
    if padding_mode == "WAN (4n+1)":
        return calculate_wan_frames, calculate_next_wan_frames, is_wan_compatible, "WAN"
    if padding_mode == "LTX2 (8n+1)":
        return calculate_ltx2_frames, calculate_next_ltx2_frames, is_ltx2_compatible, "LTX2"
    return (
        lambda n: n,
        lambda n: n,
        lambda _n: True,
        "none",
    )


def plan_frame_handles(
    current_frames: int,
    *,
    handle_frames: int = 0,
    padding_mode: str = "disabled",
    handle_side: str = "head",
    auto_pad_when_zero: bool = True,
) -> tuple[int, int, int, str]:
    """Return (target_frames, head_pad, tail_pad, policy_note).

    *handle_frames* is the editorial request (repeat first/last frame).
    When *padding_mode* is active and *handle_frames* is 0, auto-pad to the next
  compatible count (AV-Handles pattern).
    """
    cur = max(0, int(current_frames))
    req = int(handle_frames)
    calc_up, calc_next, is_compat, label = _mode_funcs(padding_mode)

    if padding_mode == "disabled":
        if req >= 0:
            target = cur + req
            if handle_side == "tail":
                return target, 0, req, f"added {req} frame(s) at tail (padding disabled)"
            return target, req, 0, f"added {req} frame(s) at head (padding disabled)"
        # trim
        target = max(0, cur + req)
        trim = cur - target
        if handle_side == "tail":
            return target, 0, -trim, f"trimmed {trim} frame(s) from tail"
        return target, -trim, 0, f"trimmed {trim} frame(s) from head"

    if req == 0 and auto_pad_when_zero:
        target = calc_next(cur) if cur > 0 else calc_up(5)
        if target < cur:
            target = calc_up(cur)
        delta = target - cur
    else:
        target = calc_up(max(5, cur + req))
        delta = target - cur

    if delta > 0:
        if handle_side == "tail":
            return target, 0, delta, f"padded {delta} frame(s) at tail to reach {target} ({label})"
        return target, delta, 0, f"padded {delta} frame(s) at head to reach {target} ({label})"
    if delta < 0:
        trim = -delta
        if handle_side == "tail":
            return target, 0, -trim, f"trimmed {trim} frame(s) from tail to reach {target} ({label})"
        return target, -trim, 0, f"trimmed {trim} frame(s) from head to reach {target} ({label})"
    note = f"already {label}-compatible at {cur}"
    if not is_compat(cur) and cur > 0:
        note = f"held at {cur} (not {label}-compatible; increase handles)"
    return cur, 0, 0, note


def apply_frame_padding(
    images: torch.Tensor,
    *,
    head_pad: int = 0,
    tail_pad: int = 0,
) -> torch.Tensor:
    """Repeat boundary frames for head/tail padding or trim."""
    if images.shape[0] == 0:
        return images
    out = images
    if head_pad > 0:
        out = torch.cat([out[0:1].repeat(head_pad, 1, 1, 1), out], dim=0)
    elif head_pad < 0:
        out = out[-head_pad:]
    if tail_pad > 0:
        out = torch.cat([out, out[-1:].repeat(tail_pad, 1, 1, 1)], dim=0)
    elif tail_pad < 0:
        out = out[: tail_pad]
    return out


def _audio_waveform_2d(audio: dict) -> tuple[torch.Tensor, int, bool, int]:
    wf = audio["waveform"]
    sr = int(audio["sample_rate"])
    was_3d = wf.ndim == 3
    batch = int(wf.shape[0]) if was_3d else 1
    if wf.ndim == 3:
        wf2 = wf[0]
    elif wf.ndim == 1:
        wf2 = wf.unsqueeze(0)
    else:
        wf2 = wf
    return wf2, sr, was_3d, batch


def sync_audio_to_frame_change(
    audio: Optional[dict],
    *,
    head_pad: int,
    tail_pad: int,
    fps: float = FPS,
) -> Optional[dict]:
    """Pad/trim audio to match frame padding (silence at head/tail)."""
    if audio is None:
        return None
    wf2, sr, was_3d, batch = _audio_waveform_2d(audio)
    fps = float(fps) if fps > 0 else float(FPS)

    def _samples(frames: int) -> int:
        return max(0, round(frames / fps * sr))

    head_s = _samples(max(0, head_pad))
    tail_s = _samples(max(0, tail_pad))
    trim_head_s = _samples(max(0, -head_pad))
    trim_tail_s = _samples(max(0, -tail_pad))

    parts = []
    if head_s > 0:
        parts.append(torch.zeros(wf2.shape[0], head_s, dtype=wf2.dtype, device=wf2.device))
    mid = wf2
    if trim_head_s > 0:
        mid = mid[:, trim_head_s:]
    if trim_tail_s > 0:
        mid = mid[:, : max(0, mid.shape[1] - trim_tail_s)]
    parts.append(mid)
    if tail_s > 0:
        parts.append(torch.zeros(wf2.shape[0], tail_s, dtype=wf2.dtype, device=wf2.device))
    out_wf = torch.cat(parts, dim=1) if len(parts) > 1 else mid

    if was_3d:
        out_wf = out_wf.unsqueeze(0).repeat(batch, 1, 1)
    elif audio["waveform"].ndim == 1:
        out_wf = out_wf.squeeze(0)
    return {"waveform": out_wf, "sample_rate": sr}


def format_handle_report(
    *,
    original_frames: int,
    final_frames: int,
    head_pad: int,
    tail_pad: int,
    padding_mode: str,
    fps: float,
    policy_note: str,
) -> str:
    fps = float(fps) if fps > 0 else float(FPS)
    lines = [
        policy_note,
        f"frames: {original_frames} -> {final_frames}",
    ]
    if head_pad:
        sec = abs(head_pad) / fps
        verb = "padded" if head_pad > 0 else "trimmed"
        lines.append(f"{verb} {abs(head_pad)} frame(s) ({sec:.3f} s) at head")
    if tail_pad:
        sec = abs(tail_pad) / fps
        verb = "padded" if tail_pad > 0 else "trimmed"
        lines.append(f"{verb} {abs(tail_pad)} frame(s) ({sec:.3f} s) at tail")
    if padding_mode != "disabled":
        lines.append(f"padding_mode={padding_mode}")
    return "\n".join(lines)
