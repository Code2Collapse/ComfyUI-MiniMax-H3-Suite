"""Temporal windows shared by the finishing nodes.

One helper, because three nodes (PlateRestore, DetailMatch,
ReferenceColorMatch) each had their own copy of the same flaw.
"""

from __future__ import annotations


def centred_window(i: int, n: int, window: int) -> tuple[int, int]:
    """[lo, hi) of a SYMMETRIC window of at most `window` frames around frame i.

    The window shrinks at the ends of the clip rather than being truncated on
    one side. That is the whole point: the drift these nodes correct - the
    ref2va red drift in particular - RAMPS over the clip, and the median of a
    monotonic run is exactly its centre only when the window is centred.

    The copies this replaces truncated one-sidedly: at the last frame the
    window was [n-3, n) and the "median" was frame n-2's value, and a
    4-frame window returned torch.median's LOWER middle. Both lag a ramp, so
    the end of every drifting clip - where the drift is largest - was
    under-corrected. Measured on a 6-frame +0.03 Oklab-a ramp: 0.0093 left
    at the last frame, against 0.005 allowed.
    """
    if n <= 0:
        return 0, 0
    half = max(0, int(window) // 2)
    r = min(half, i, n - 1 - i)
    return i - r, i + r + 1
