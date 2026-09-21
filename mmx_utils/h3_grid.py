"""H3 temporal grid helpers for mask reduction and frame alignment."""

from __future__ import annotations

from .h3_constants import FRAME_PER_TOKEN, align_frame_count, video_latent_t

# Pixel-frame to audio-latent-sample scale (24 fps video vs 40 Hz audio clock).
# Added for split-upscale temporal stitching; same ratio as motion_context.py.
FRAME_RESCALE = 5.0 / 3.0


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


def snap_frame_count_nearest(frame_count: int) -> int:
    """Snap to the nearest H3 17n+5 count (minimum 5).

    Multishot upstream uses round-to-nearest, not round-up — a 5.2s request at 24fps
    (125 frames) lands on 124, not 141. Added for the native-audio multishot port.
    """
    n = max(5, int(frame_count))
    return max(5, 17 * max(0, round((n - 5) / 17)) + 5)


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


# ── token / frame mapping (split-upscale port; upstream clip_tokens == video_latent_t) ──


def frames_for_tokens(n: int) -> int:
    """Pixel frames covered by the first *n* video-latent rows."""
    return sum(FRAME_PER_TOKEN[i % 5] for i in range(max(0, int(n))))


def tokens_for_frames(f: int) -> int:
    """Smallest latent-row count whose pixel span reaches frame *f*."""
    n, acc = 0, 0
    target = int(f)
    while acc < target:
        acc += FRAME_PER_TOKEN[n % 5]
        n += 1
    return n


def clip_tokens(n: int) -> int:
    """Latent rows for *n* pixel frames — alias of video_latent_t."""
    return video_latent_t(int(n))


def snap_clip_frames(v: int) -> int:
    """Snap pixel-frame count to the nearest H3 grid point (17n+5)."""
    v = int(v)
    if v >= 5:
        return 5 + 17 * max(1, round((v - 5) / 17))
    return max(1, v)


def snap_overlap_frames(v: int) -> int:
    """Snap overlap length to the H3 grid; zero stays zero."""
    v = int(v)
    if v <= 0:
        return 0
    return 5 + 17 * max(0, round((v - 5) / 17))


def steps_for_frames(n: int) -> int | None:
    """Latent rows for exactly *n* pixel frames, or None if *n* is off-grid."""
    k, covered = 0, 0
    target = int(n)
    while covered < target:
        covered += FRAME_PER_TOKEN[k % 5]
        k += 1
    return k if covered == target else None


def token_start_at_or_before(f: int) -> int:
    """Largest latent index whose pixel span still starts at or before frame *f*."""
    k = 0
    while frames_for_tokens(k + 1) <= int(f):
        k += 1
    return k


def audio_range(f0: int, f1: int) -> tuple[int, int]:
    """Map a half-open pixel-frame span to audio-latent sample indices."""
    return round(int(f0) * FRAME_RESCALE), round(int(f1) * FRAME_RESCALE)


# Image-studio temporal packet sizing (short 1/5/9/13/20-frame profiles).
# PORTED FROM: ComfyUI-MiniMax-H3-Image-Studio :: nodes.py
# (_decoded_frames_for_latent_t / _latent_t_for_frame_count).
# Not the same as frames_for_tokens() — that sums the token clock for aligned
# 17n+5 clips; image mode may stop mid-cycle on a shorter decoded span.


def decoded_frames_for_latent_t(latent_t: int) -> int:
    """Natural H3 VAE output length for a temporal latent length.

    This arrived with the Image Studio port carrying its own closed-form
    version of the same grid. Checked exhaustively over latent_t 0..2000, it
    agrees with frames_for_tokens at every value except 0, where this one
    clamps to a single frame and the other returns none. A latent of zero rows
    is not a real input, so the clamp is kept and the arithmetic is not
    duplicated: two implementations of one grid are two things to keep in
    sync, and this pack has been bitten by exactly that before.
    """
    return frames_for_tokens(max(1, int(latent_t)))


def latent_t_for_frame_count(frame_count: int) -> tuple[int, int]:
    """Smallest temporal latent that decodes at least *frame_count* images."""
    requested = max(1, int(frame_count))
    latent_t = 1
    while decoded_frames_for_latent_t(latent_t) < requested:
        latent_t += 1
    return latent_t, decoded_frames_for_latent_t(latent_t)


def compute_h3_segments_adaptive(
    total_tokens: int,
    chunk_frames: int,
    overlap_frames: int,
) -> tuple[list[tuple[int, int, int, int]], int]:
    """Plan temporal segments on the H3 latent grid.

    Returns ``(bounds, total_pixel_frames)`` where each bound is
    ``(k0, f0, k1, f1)`` — half-open token indices and matching pixel frames.
    *total_tokens* is the video latent ``T`` dimension; chunk/overlap are pixel
    frames snapped through clip_tokens.
    """
    tv = int(total_tokens)
    tc = clip_tokens(chunk_frames)
    to = clip_tokens(overlap_frames) if overlap_frames > 0 else 0
    if to >= tc:
        to = max(0, tc - 1)
    if tc >= tv:
        return [(0, 0, tv, frames_for_tokens(tv))], frames_for_tokens(tv)
    hop = max(1, tc - to)
    bounds: list[tuple[int, int, int, int]] = []
    prev_k0, i = -1, 0
    while True:
        k0 = i * hop
        if k0 + tc >= tv:
            k1 = tv
            k0 = max(k0, tv - tc)
            if prev_k0 >= 0 and k0 <= prev_k0:
                k0 = prev_k0 + 1
            if k0 >= tv:
                break
            bounds.append((k0, frames_for_tokens(k0), k1, frames_for_tokens(k1)))
            break
        bounds.append((k0, frames_for_tokens(k0), k0 + tc, frames_for_tokens(k0 + tc)))
        prev_k0 = k0
        i += 1
    return bounds, frames_for_tokens(tv)
