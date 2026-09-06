"""Frame-range list parsing: "0:24, 40, 100:end" -> selected frame indices.

Original implementation, GPL-3.0, written from the behavioural specification in
docs/cleanroom_maskvid.md (§A2).

Python SLICE semantics throughout, because that is the convention every user of
this tool already knows: the stop frame is EXCLUDED, negatives count back from
the end, and slice endpoints CLAMP rather than erroring.

The one deliberate asymmetry: a slice that runs off the end clamps, but a bare
single frame that is out of range RAISES. That mirrors Python itself
(`xs[0:9999]` is fine, `xs[9999]` is an IndexError) and it is the useful
behaviour - "frames 0:9999" plainly means "all of them", whereas "frame 9999"
on a 24-frame clip is a mistake the author wants told about, not silently dropped.
"""

from __future__ import annotations

import re

_INTEGER = re.compile(r"^-?\d+$")
_SEPARATORS = re.compile(r"[,\n;]")

SYNTAX_HELP = (
    "expected comma-separated frame numbers or Python-style slices, e.g. "
    "'0:24, 40, 100:end' (stop is excluded, negatives count from the end)"
)


class FrameRangeError(ValueError):
    """An authoring mistake in the range text.

    A distinct type so a caller can tell "the user typed something wrong" from
    "this code has a bug", and surface the first as a readable message rather
    than a traceback.
    """


def _bound(token: str, *, is_stop: bool, segment: str):
    """One side of a slice: a number, empty (open), or the word 'end'."""
    if token == "":
        return None
    if token == "end":
        if not is_stop:
            raise FrameRangeError(
                f"'end' only makes sense as the END of a range, but '{segment}' "
                f"uses it as the start. {SYNTAX_HELP}"
            )
        return None
    if not _INTEGER.match(token):
        raise FrameRangeError(f"'{token}' in '{segment}' is not a frame number. {SYNTAX_HELP}")
    return int(token)


def parse_frame_ranges(text: str, frame_count: int) -> list[int]:
    """Return the sorted, de-duplicated frame indices selected by `text`."""
    return parse_frame_ranges_detailed(text, frame_count)["frames"]


def parse_frame_ranges_detailed(text: str, frame_count: int) -> dict:
    """As `parse_frame_ranges`, plus a diagnostic of what each segment matched.

    Returned rather than logged so the node can report exactly what it selected
    without parsing the text a second time.
    """
    frames = int(frame_count)
    if frames <= 0:
        # Checked BEFORE parsing: with 0 frames the modulo used for negative
        # indices would raise ZeroDivisionError, which tells the user nothing.
        raise FrameRangeError(f"frame_count must be at least 1, got {frames}.")

    selected: set[int] = set()
    detail: list[dict] = []

    for raw in _SEPARATORS.split(text or ""):
        segment = "".join(raw.split())      # whitespace inside a segment is insignificant
        if not segment:
            continue

        if _INTEGER.match(segment):
            frame = int(segment)
            if not -frames <= frame < frames:
                raise FrameRangeError(
                    f"frame {frame} is outside a clip of {frames} frames "
                    f"(valid: {-frames} to {frames - 1}). Use a range like "
                    f"'{frame}:end' if you meant to clamp instead."
                )
            picked = [frame % frames]
        else:
            parts = segment.split(":")
            if len(parts) > 3:
                raise FrameRangeError(f"could not read frame range '{segment}'. {SYNTAX_HELP}")
            start = _bound(parts[0], is_stop=False, segment=segment)
            stop = _bound(parts[1], is_stop=True, segment=segment)
            step = _bound(parts[2], is_stop=False, segment=segment) if len(parts) == 3 else None
            if step == 0:
                raise FrameRangeError(f"frame range '{segment}' has a step of zero.")
            # slice().indices() gives Python's own clamping and negative handling,
            # so the semantics match what the user already expects from lists.
            picked = list(range(*slice(start, stop, step).indices(frames)))

        selected.update(picked)
        detail.append({"segment": segment, "count": len(picked)})

    return {"frames": sorted(selected), "segments": detail, "frame_count": frames}
