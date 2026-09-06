"""Soft-edged per-step denoise masks for differential diffusion.

Original implementation, GPL-3.0, written from the behavioural specification in
docs/cleanroom_maskvid.md (§A1). See that document for the clean-room caveats:
the source studied is GPL-3.0 and so is this pack, so a direct port would have
been permitted; this is written independently so the behaviour could later move
to an Apache-2.0 pack, which a port never could.

WHY A RAMP INSTEAD OF A THRESHOLD
---------------------------------
Differential diffusion turns one greyscale mask into a per-step mask that admits
progressively more of the frame as sampling proceeds. Doing that with a hard
threshold makes EVERY intermediate mask binary, so a carefully feathered input
still produces a razor edge at every step. On video that edge lands on a slightly
different pixel each frame and crawls.

A linear ramp of width `w` in mask VALUE space produces a band of width
`w / |grad m|` in PIXEL space. So the softness of the per-step mask automatically
follows the blur already in the input: feathered stays feathered, sharp stays
sharp. It is per-pixel with no spatial blur, which is what keeps it temporally
stable frame to frame.
"""

from __future__ import annotations

import torch


def schedule_threshold(
    sigma_now: torch.Tensor | float,
    sigma_first: torch.Tensor | float,
    sigma_last: torch.Tensor | float,
    to_timestep,
    sigma_floor: torch.Tensor | float | None = None,
):
    """Normalised sampling progress in TIMESTEP space: 1.0 at the first step, 0.0 at the last.

    Timestep space rather than sigma space because sigma is wildly non-linear -
    a threshold swept linearly in sigma would spend most of the schedule at one
    end. `to_timestep` is the model's own mapping, so this tracks whatever
    schedule the sampler is actually using.

    `sigma_floor` is the model's minimum sigma. The END of the sweep is the
    HIGHER of the floor and the last scheduled sigma, so a partial-denoise run
    (which stops early) still spans a full 1 -> 0 range instead of compressing
    into a fraction of it.
    """
    end = sigma_last
    if sigma_floor is not None and float(sigma_last) < float(sigma_floor):
        end = sigma_floor

    t_now = float(to_timestep(sigma_now))
    t_start = float(to_timestep(sigma_first))
    t_end = float(to_timestep(end))

    span = t_start - t_end
    if abs(span) < 1e-12:
        # Degenerate single-step schedule: treat it as "fully progressed" rather
        # than dividing by ~0 and producing inf.
        return 0.0
    return (t_now - t_end) / span


def soft_step_mask(
    mask: torch.Tensor,
    threshold: float,
    softness: float,
    strength: float = 1.0,
) -> torch.Tensor:
    """Per-step denoise mask: a ramp of width `softness` centred on `threshold`.

    Pure function - no model, no sampler state - so the maths is testable
    directly rather than only through a live sampling run.

    Args:
        mask: the user's greyscale denoise mask, any shape, values in [0, 1].
        threshold: sampling progress, 1.0 at the first step down to 0.0.
        softness: ramp width in MASK-VALUE units. 0 = hard threshold (the stock
            behaviour); 1 = no schedule at all, blend by the raw mask.
        strength: blend the scheduled mask back toward the raw one.
    """
    if not torch.is_tensor(mask):
        raise TypeError(f"mask must be a tensor, got {type(mask).__name__}")
    softness = float(min(max(softness, 0.0), 1.0))
    strength = float(min(max(strength, 0.0), 1.0))

    if softness <= 0.0:
        scheduled = (mask >= threshold).to(mask.dtype)
    else:
        # Explicit band edges. The centre must traverse [s/2, 1 - s/2] rather
        # than [0, 1]: otherwise a mask value of exactly 1 would not be fully
        # open on the FIRST step, and a value of 0 would leak in before the
        # last. Those two endpoints are the whole contract with stock DD.
        lo = threshold * (1.0 - softness)
        hi = lo + softness
        scheduled = ((mask - lo) / (hi - lo)).clamp(0.0, 1.0).to(mask.dtype)

    if strength >= 1.0:
        return scheduled
    return (strength * scheduled + (1.0 - strength) * mask).to(mask.dtype)


def band_width_in_pixels(softness: float, gradient_magnitude: float) -> float:
    """The spatial width the ramp produces for a given mask gradient.

    Exposed because it is the property that justifies the design, and having it
    as a function means the relationship can be asserted in a test rather than
    only claimed in a docstring.
    """
    if gradient_magnitude <= 0:
        return float("inf")     # a flat mask region has no edge to place
    return float(softness) / float(gradient_magnitude)
