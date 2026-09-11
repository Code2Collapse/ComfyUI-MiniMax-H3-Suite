"""Context-window planning for unbounded-length H3 generation.

H3 cannot be asked for an arbitrarily long clip in one pass, so long output is
made of overlapping passes that are stitched. Getting that right is entirely
about H3's own arithmetic, not about picking round numbers:

  * A legal frame count satisfies ``n % 17 == 5``. The latent grid is a 5-frame
    head (1 + 4) followed by 17-frame cycles of FRAME_PER_TOKEN (1,4,4,4,4),
    each cycle costing 5 latent rows -- see mmx_utils/h3_grid.py. A window whose
    length is not of that form does not tile onto the grid and the layout is
    wrong before sampling starts.

  * Every window after the first must START on a cycle boundary, i.e. the stride
    must be a multiple of 17. Only then does window k have the same internal
    latent structure as window k-1, which is what makes the passes stitchable.

  * Position is carried in H3's temporal RoPE, which runs at 40 Hz against
    24 fps output -- 40/24 RoPE units per frame. A window starting at frame S
    must be offset by S * 40/24 or the model believes every pass starts at t=0
    and the seam drifts. (Constants cross-checked against
    third_party/ComfyUI-MiniMax-H3-LongMedia/temporal_positioning.py.)

This module is pure arithmetic: no torch, no model, no I/O, so the plan can be
tested and drawn without weights.
"""

from __future__ import annotations

from typing import Any

H3_OUTPUT_FPS = 24.0
H3_TEMPORAL_ROPE_HZ = 40.0
H3_ROPE_UNITS_PER_FRAME = H3_TEMPORAL_ROPE_HZ / H3_OUTPUT_FPS

CYCLE = 17          # frames per FRAME_PER_TOKEN cycle
HEAD = 5            # frames in the head group (1 + 4)
ROWS_PER_CYCLE = 5  # latent rows each cycle costs


class ContextWindowError(ValueError):
    """Raised for a request that cannot be expressed on H3's grid."""


def is_legal_frame_count(n: int) -> bool:
    """True when n lands exactly on H3's latent grid."""
    return int(n) % CYCLE == HEAD


def snap_frame_count(n: int, direction: str = "up") -> int:
    """Snap n to the nearest legal count (n % 17 == 5).

    `up` never shortens the clip, which is the safe default: a shorter window
    silently drops frames the artist asked for.
    """
    n = int(n)
    if n < HEAD:
        return HEAD
    if is_legal_frame_count(n):
        return n
    if direction == "down":
        while not is_legal_frame_count(n) and n > HEAD:
            n -= 1
        return max(HEAD, n)
    while not is_legal_frame_count(n):
        n += 1
    return n


def latent_rows(frame_count: int) -> int:
    """Latent rows a window of this many frames occupies."""
    fc = int(frame_count)
    return 2 if fc <= HEAD else ((fc - HEAD) // CYCLE) * ROWS_PER_CYCLE + 2


def rope_offset_for_frame(start_frame: int) -> float:
    """H3 temporal-RoPE offset for a window beginning at this global frame."""
    if int(start_frame) < 0:
        raise ContextWindowError("A window cannot start before frame 0.")
    return int(start_frame) * H3_ROPE_UNITS_PER_FRAME


def plan_context_windows(
    total_frames: int,
    window_frames: int = 81,
    overlap_frames: int = 17,
    snap: str = "up",
) -> dict[str, Any]:
    """Tile `total_frames` with overlapping, H3-legal windows.

    Returns a dict with the window list and everything the report and the
    on-node timeline need. Raises ContextWindowError with a sentence a person
    can act on -- never returns a silently degraded plan.
    """
    total = int(total_frames)
    if total < 1:
        raise ContextWindowError("total_frames must be at least 1.")

    win_req = int(window_frames)
    if win_req < HEAD:
        raise ContextWindowError(
            f"window_frames must be at least {HEAD}; H3's head group alone is "
            f"{HEAD} frames."
        )
    win = snap_frame_count(win_req, snap)

    ov_req = int(overlap_frames)
    if ov_req < 0:
        raise ContextWindowError("overlap_frames cannot be negative.")
    if ov_req >= win:
        raise ContextWindowError(
            f"overlap_frames ({ov_req}) must be smaller than the window "
            f"({win} frames after snapping), otherwise the windows never advance."
        )

    # The stride must be a whole number of cycles or window k stops sharing the
    # latent structure of window k-1 and the passes cannot be stitched.
    stride_req = win - ov_req
    stride = max(CYCLE, (stride_req // CYCLE) * CYCLE)
    overlap = win - stride

    windows: list[dict[str, Any]] = []
    start = 0
    idx = 0
    while True:
        end = min(start + win, total)
        frames = end - start
        windows.append({
            "index": idx,
            "start": int(start),
            "end": int(end),
            "frames": int(frames),
            "requested_frames": int(win),
            "latent_rows": latent_rows(win),
            "rope_offset": round(rope_offset_for_frame(start), 6),
            "overlap_prev": int(overlap) if idx > 0 else 0,
            "is_tail": end >= total,
            "short_tail": frames < win,
        })
        if end >= total:
            break
        start += stride
        idx += 1
        if idx > 100_000:  # pathological guard; stride is always >= 17 so unreachable
            raise ContextWindowError("Window plan did not terminate; check the inputs.")

    covered = windows[-1]["end"]
    return {
        "total_frames": total,
        "window_frames": win,
        "window_frames_requested": win_req,
        "snapped": win != win_req,
        "overlap_frames": overlap,
        "overlap_requested": ov_req,
        "stride_frames": stride,
        "cycle": CYCLE,
        "fps": H3_OUTPUT_FPS,
        "rope_units_per_frame": H3_ROPE_UNITS_PER_FRAME,
        "windows": windows,
        "window_count": len(windows),
        "covered_frames": covered,
        "latent_rows_per_window": latent_rows(win),
    }


def plan_report(plan: dict[str, Any]) -> str:
    """Human summary. Says what was changed and why, not just what was chosen."""
    w = plan["windows"]
    lines = [
        f"{plan['window_count']} window(s) covering {plan['covered_frames']} "
        f"of {plan['total_frames']} frames "
        f"({plan['covered_frames'] / plan['fps']:.2f}s at {plan['fps']:.0f} fps)",
        f"  window        {plan['window_frames']} frames "
        f"-> {plan['latent_rows_per_window']} latent rows",
        f"  stride        {plan['stride_frames']} frames "
        f"({plan['stride_frames'] // plan['cycle']} cycle(s) of {plan['cycle']})",
        f"  overlap       {plan['overlap_frames']} frames",
    ]
    if plan["snapped"]:
        lines.append(
            f"  NOTE: window_frames {plan['window_frames_requested']} is not on "
            f"H3's grid (needs n % 17 == 5) and was snapped to "
            f"{plan['window_frames']}. An unsnapped length does not tile onto "
            f"the latent grid."
        )
    if plan["overlap_requested"] != plan["overlap_frames"]:
        lines.append(
            f"  NOTE: overlap {plan['overlap_requested']} was adjusted to "
            f"{plan['overlap_frames']} so the stride stays a whole number of "
            f"17-frame cycles; without that, later windows stop sharing the "
            f"latent structure of the first and cannot be stitched."
        )
    tail = w[-1]
    if tail["short_tail"]:
        lines.append(
            f"  tail window is {tail['frames']} frames, shorter than "
            f"{plan['window_frames']}; pad or trim it after sampling."
        )
    lines.append(
        f"  rope offset   {w[0]['rope_offset']:.3f} .. {w[-1]['rope_offset']:.3f} "
        f"({plan['rope_units_per_frame']:.4f} units/frame at 40 Hz)"
    )
    return "\n".join(lines)
