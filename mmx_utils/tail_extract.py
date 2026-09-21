"""Take the tail of a generated clip so the next clip can continue from it.

Chaining clips by hand means decoding the last one, picking off the final
half-second of picture AND the matching half-second of sound, and feeding both
back as references. Doing it by eye goes wrong in two specific ways, and
neither announces itself:

  * PICTURE AND SOUND DRIFT APART. The tail is chosen in frames and the audio
    in samples. Round them independently and the reference video is 12 frames
    while its soundtrack is 0.6s, so the next clip starts with lips already
    out of step and every clip after it inherits the offset.

  * THE FRAME COUNT IS OFF THE GRID. H3 packs video as 5 frames then 17-frame
    chunks. A reference video of any other length is reinterpreted rather than
    refused, and the motion it was supposed to carry arrives at the wrong
    speed.

Both are handled here: the audio length is derived from the FINAL frame count,
after grid alignment, rather than from what was asked for.

Pure torch; the VAEs stay in the node.
"""

from __future__ import annotations

import torch

FPS = 24.0
MIN_H3_FRAMES = 5


class TailExtractError(ValueError):
    pass


def align_up(frames: int) -> int:
    """The smallest legal H3 frame count at or above `frames`.

    Up, not down, because this count is a REQUEST for a reference length: a
    tail shorter than asked for carries less motion, and 5 frames is already
    the floor.
    """
    n = max(MIN_H3_FRAMES, int(frames))
    while n % 17 != 5:
        n += 1
    return n


def tail_frame_count(total: int, seconds: float, *, align: bool) -> int:
    """How many frames the tail actually gets.

    Never more than exist. Alignment rounds up and is then clamped back to
    `total`, so a request that cannot be aligned inside the clip returns the
    whole clip rather than an off-grid slice - the clamp can take the result
    off the grid, which the caller reports rather than hides.
    """
    if total <= 0:
        raise TailExtractError(
            "The decoded clip has no frames. The latent did not contain a "
            "video stream - check that this is an H3 AV latent from the "
            "sampler, not an audio-only or empty latent.")
    wanted = max(1, min(total, int(round(float(seconds) * FPS))))
    if not align:
        return wanted
    return min(total, align_up(max(MIN_H3_FRAMES, wanted)))


def slice_audio_tail(audio: dict, seconds: float) -> dict:
    """The last `seconds` of the waveform.

    Taken from the END, because the point is the join: the next clip starts
    where this one stopped. Slicing from the start would hand it audio from
    half a clip ago.
    """
    sr = int(audio["sample_rate"])
    waveform = audio["waveform"]
    wanted = max(1, int(round(float(seconds) * sr)))
    start = max(0, int(waveform.shape[-1]) - wanted)
    return {"waveform": waveform[..., start:], "sample_rate": sr}


def normalise_audio(audio: torch.Tensor) -> torch.Tensor:
    """H3's audio decode convention: scale by 5x the per-clip std, floored at 1.

    The floor is what stops a quiet or silent passage being multiplied up into
    noise - without it, a clip of near-silence comes back as a hiss at full
    level.
    """
    std = torch.std(audio, dim=[1, 2], keepdim=True) * 5.0
    std[std < 1.0] = 1.0
    return audio / std


def describe(total: int, count: int, seconds_asked: float, *,
             aligned: bool) -> str:
    """What the tail actually is, in the terms the join cares about."""
    duration = count / FPS
    lines = [
        f"Tail: {count} frames ({duration:.3f}s) from a {total}-frame clip.",
        f"The audio was cut to the SAME {duration:.3f}s, measured from the "
        "final frame count rather than the requested length - rounding the two "
        "separately drifts picture against sound, and every later clip "
        "inherits the offset.",
    ]
    asked = int(round(float(seconds_asked) * FPS))
    if aligned and count != asked:
        lines.append(
            f"Rounded {asked} -> {count} frames to land on H3's 17k+5 grid. An "
            "off-grid reference video is reinterpreted, not refused, and its "
            "motion comes out at the wrong speed.")
    if aligned and count % 17 != 5:
        lines.append(
            f"WARNING: {count} frames is NOT on the grid. The clip is only "
            f"{total} frames long, so the aligned length did not fit and the "
            "whole clip was used. Generate a longer clip, or use the last "
            "frame as an image reference instead.")
    if count == total:
        lines.append(
            "This is the entire clip, not a tail. As a reference it carries "
            "the whole shot's motion rather than just the join.")
    return "\n".join(lines)
