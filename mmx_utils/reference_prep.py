"""Geometry and ordering for H3 keyframe + reference conditioning.

H3 accepts two different kinds of image conditioning and the stock nodes make
you pick one:

  KEYFRAMES (fl2va)   first_frame and last_frame occupy real frame positions
                      in the output. They say WHERE the shot starts and ends.
  REFERENCES (ref2va) <Picture i>, <Video k>, <Audio j> have no frame
                      position. They say WHO and WHAT SORT - identity, motion,
                      voice.

They are not alternatives; a shot usually wants both. This module holds the
parts of combining them that are arithmetic rather than tensor work, so they
can be tested without a VAE, a CLIP or a GPU.

Two of them are easy to get subtly wrong:

  * FRAME SNAPPING. H3's latent grid is 17k+5. A reference video whose frame
    count is not on that grid does not error - it is silently reinterpreted,
    and the motion it was supposed to convey comes out at the wrong speed.

  * REFERENCE NUMBERING. The text encoder is told "<Picture 1>", "<Video 2>"
    in one order, and the DiT is handed latent blocks in another list. If the
    two orders disagree the prompt refers to the wrong reference: ask for
    Picture 1's face and get Picture 3's. Nothing raises; the render is just
    of the wrong person.
"""

from __future__ import annotations

import math

# H3's canvas granularity and the reference pipeline's short edge. Mirrored
# from comfy_extras.nodes_minimax_h3 rather than imported, so this module can
# be tested with no ComfyUI present; the node asserts they still agree.
CANVAS_MULTIPLE = 16
REF_IMAGE_SHORT_EDGE = 2048
FPS = 24

# A reference video must be long enough to show motion at all.
MIN_REF_VIDEO_FRAMES = 5


class ReferencePrepError(ValueError):
    """Raised where a silent misinterpretation would otherwise happen."""


def snap_to_latent_grid(frames: int) -> int:
    """The largest usable frame count at or below `frames`.

    H3 packs video as 5 frames then 17-frame chunks, so a legal count
    satisfies n % 17 == 5. Rounding DOWN, never up: padding a reference video
    with frames that do not exist invents motion at its tail.
    """
    if frames < MIN_REF_VIDEO_FRAMES:
        raise ReferencePrepError(
            f"A reference video needs at least {MIN_REF_VIDEO_FRAMES} frames "
            f"(~0.2s at {FPS} fps) to carry any motion, but this one has "
            f"{frames}. Feed a longer clip, or use it as a reference IMAGE "
            "instead - a single frame is a perfectly good identity reference.")
    n = frames
    while n % 17 != 5:
        n -= 1
    return n


def reference_scale(width: int, height: int, ref_w: int, ref_h: int,
                    mode: str) -> float:
    """How much to scale a reference image by. Never above 1.0.

    'match' brings the reference to roughly the generation's pixel area, which
    keeps its token cost proportional to the shot.

    'max' uses the reference pipeline's 2048px short edge, which holds far
    more identity detail - and costs it on EVERY sampling step, because
    reference tokens ride through the whole denoise rather than being consumed
    once.

    Upscaling is refused in both modes: it adds tokens without adding
    information, and a reference is a source of detail, not a place to invent
    it.
    """
    if mode not in ("match", "max"):
        raise ReferencePrepError(
            f"Unknown reference size mode {mode!r}. Use 'match' (scale to the "
            "generation's pixel area) or 'max' (2048px short edge, sharper "
            "identity, much slower).")
    if ref_w <= 0 or ref_h <= 0:
        raise ReferencePrepError(
            f"A reference image has a zero dimension ({ref_w}x{ref_h}).")
    if mode == "match":
        return min(1.0, math.sqrt((width * height) / float(ref_w * ref_h)))
    return min(1.0, REF_IMAGE_SHORT_EDGE / float(min(ref_w, ref_h)))


def canvas_size(ref_w: int, ref_h: int, scale: float) -> tuple[int, int]:
    """Scaled size snapped to the canvas grid, never smaller than one cell."""
    w = max(CANVAS_MULTIPLE,
            round(ref_w * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
    h = max(CANVAS_MULTIPLE,
            round(ref_h * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
    return int(w), int(h)


def soundtrack_key(video_key: str) -> str:
    """The audio slot paired with a reference-video slot.

    The pairing is by trailing NUMBER, not by position: autogrow inputs are a
    dict, a user can fill video 1 and video 3 and leave 2 empty, and matching
    by position would then put video 3's soundtrack on video 1 - a voice on
    the wrong person, with nothing raised.
    """
    return "ref_video_audio_" + video_key.rsplit("_", 1)[-1]


def sample_indices(frame_count: int, per_second: int = 2) -> list[int]:
    """Which frames of a reference video the TEXT encoder sees.

    The DiT gets every frame; the encoder gets a sparse sample, because it is
    reading the clip for gist rather than motion. Two per second matches the
    reference pipeline.
    """
    if frame_count <= 0:
        return []
    stride = max(1, FPS // max(1, per_second))
    return list(range(0, frame_count, stride))


def sample_timestamps(indices: list[int], per_second: int = 2) -> list[float]:
    """Seconds for each sampled frame, so the encoder can order them."""
    return [i / float(per_second) for i in range(len(indices))]


def check_reference_order(items: list[dict], blocks: list[dict]) -> None:
    """Refuse to hand the encoder and the DiT differently-ordered references.

    This is the failure that produces a confident render of the wrong person.
    The encoder numbers references in the order it is given them, so the
    prompt's "<Picture 1>" means whatever came first; the DiT indexes its own
    list. Any drift between the two is silent.

    Audio entries legitimately appear in both lists, and a video with a
    soundtrack contributes an audio item AND a video item to `items` while
    contributing one combined block - so the counts are compared by KIND
    rather than by length.
    """
    item_kinds = [i.get("type") for i in items]
    block_kinds = [b.get("kind") for b in blocks]

    item_images = item_kinds.count("image")
    block_images = block_kinds.count("image")
    if item_images < block_images:
        raise ReferencePrepError(
            f"The text encoder was given {item_images} reference image(s) but "
            f"the model was given {block_images}. The prompt's <Picture n> "
            "would refer to the wrong reference, which renders the wrong "
            "subject without raising anything.")

    item_videos = item_kinds.count("video")
    block_videos = sum(1 for k in block_kinds if k in ("video", "video_audio"))
    if item_videos != block_videos:
        raise ReferencePrepError(
            f"The text encoder was given {item_videos} reference video(s) but "
            f"the model was given {block_videos}. <Video n> would refer to the "
            "wrong clip.")


def describe(n_keyframes: int, n_images: int, n_videos: int, n_audios: int,
             mode: str, length: int, snapped: int) -> str:
    """What was actually built, in the terms the user thinks in."""
    lines: list[str] = []
    if not (n_keyframes or n_images or n_videos or n_audios):
        lines.append(
            "No keyframes and no references: this is a plain text-to-video "
            "conditioning. That works, but the two halves of this node are "
            "both unused - the stock text-to-video node does the same thing.")

    if n_keyframes:
        lines.append(
            f"{n_keyframes} keyframe(s): these occupy real frame positions and "
            "fix where the shot starts and ends.")
    if n_images:
        lines.append(
            f"{n_images} reference image(s) at '{mode}' size. References have "
            "NO frame position - they steer identity and style throughout.")
        if mode == "max":
            lines.append(
                "'max' holds 2048px of identity detail and pays for it on "
                "EVERY sampling step, because reference tokens ride through "
                "the whole denoise. Several times slower than 'match'.")
    if n_videos:
        lines.append(f"{n_videos} reference video(s) for motion and style.")
    if n_audios:
        lines.append(f"{n_audios} reference audio clip(s) for voice.")

    if snapped != length:
        lines.append(
            f"Length snapped {length} -> {snapped} frames to land on H3's "
            "17k+5 latent grid. An unsnapped count is reinterpreted silently "
            "and the motion comes out at the wrong speed.")

    if n_keyframes and n_videos:
        lines.append(
            "Keyframes and reference videos together: the keyframes win at "
            "frames 0 and last, the reference video steers everything between. "
            "If they disagree about the subject, expect a visible handover at "
            "the ends.")
    return "\n".join(lines)
