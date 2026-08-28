"""H3 temporal grid helpers for mask reduction and frame alignment."""

from __future__ import annotations

from .h3_constants import FRAME_PER_TOKEN, align_frame_count, video_latent_t


def h3_pixel_frame_groups(frame_count: int) -> list[tuple[int, int]]:
    """Map each video-latent index to a half-open pixel-frame range [start, end).

    Head (frames 0-4): groups (1) + (4) -> 2 latents.
    Thereafter each 17-frame cycle uses FRAME_PER_TOKEN (1,4,4,4,4).
    """
    t = max(0, int(frame_count))
    groups: list[tuple[int, int]] = []
    if t <= 0:
        return groups
    groups.append((0, 1))
    if t > 1:
        groups.append((1, min(5, t)))
    pos = 5
    while pos < t:
        for gsize in FRAME_PER_TOKEN:
            if pos >= t:
                break
            end = min(pos + int(gsize), t)
            groups.append((pos, end))
            pos = end
    return groups


def assert_groups_match_latent_t(frame_count: int) -> None:
    groups = h3_pixel_frame_groups(frame_count)
    expected = video_latent_t(frame_count)
    if len(groups) != expected:
        raise ValueError(
            f"h3_pixel_frame_groups({frame_count}) produced {len(groups)} groups "
            f"but video_latent_t expects {expected}"
        )


def snap_frame_count(frame_count: int) -> int:
    return align_frame_count(max(5, int(frame_count)))


def is_h3_compatible(frames: int) -> bool:
    """True when frame count is on the H3 17n+5 grid (n>=0)."""
    n = int(frames)
    return n >= 5 and n % 17 == 5


def calculate_h3_frames(target_frames: int) -> int:
    """Round UP to the next H3-compatible count (17n+5), minimum 5."""
    return snap_frame_count(max(5, int(target_frames)))


def calculate_next_h3_frames(current_frames: int) -> int:
    """Next legal H3 count strictly above *current* when not already compatible."""
    n = max(5, int(current_frames))
    if is_h3_compatible(n):
        return n
    return calculate_h3_frames(n)
