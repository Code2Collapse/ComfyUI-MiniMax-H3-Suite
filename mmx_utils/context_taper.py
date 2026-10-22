# PORTED FROM: https://github.com/beijinren/ComfyUI-H3-Context-Noise@7e55312 taper.py (MIT, Copyright (c) 2026 beijinren)
# which carries the palette, the 36x64 grid and alpha_for from
# https://github.com/MacroSony/minimax-h3-chained-character-swap scripts/inject_tail_taper.py (MIT) and mirrors
# FRAME_PER_TOKEN from NikoDemon80/ComfyUI-H3-Motion-Context (GPL-3.0). Logic unchanged; names adapted to the suite.
"""Taper maths for H3 context noise (torch-free, so the schedule is testable on its own).

A flat injection strength ``alpha`` across the carried window, ramping linearly down to ``alpha_end`` over the last
``ramp`` positions, so the frame touching the seam is the least corrupted. An empirical recipe (validated upstream
as 22 frames / 0.45 / 0.10 / 3), not a description of H3 internals.
"""

from __future__ import annotations

import random

from mmx_utils.h3_constants import FRAME_PER_TOKEN

# Desaturated purple / green / grey / pink: destroys chroma without shifting the average luma much, so the carried
# frames stay readable as motion while their colour is gone.
PALETTE = (
    (185, 115, 215),
    (115, 195, 140),
    (150, 148, 162),
    (205, 150, 192),
    (138, 182, 148),
    (160, 120, 175),
)
POC_GRID = (36, 64)          # (w, h): 16 x 16 px blocks at the recipe's 576 x 1024
POC_BLOCK = 16
VIDEO_RUN_GRID = (124, 107, 90, 73, 56, 39, 22, 5, 1)
LATENT_SAFE_RUNS = (5, 22, 39, 56)  # whole numbers of latent steps from a cycle start: 2 / 7 / 12 / 17 steps


class TaperError(ValueError):
    pass


def alpha_for(position: int, tail: int, alpha: float, alpha_end: float, ramp: int) -> float:
    """Alpha at a zero-based position in the tail: 0 = oldest, tail - 1 = the one touching the seam."""
    if not 0 <= position < tail:
        raise TaperError("position must be inside the injected tail")
    from_end = tail - 1 - position
    if from_end >= ramp:
        return alpha
    return alpha + (alpha_end - alpha) * (ramp - from_end) / ramp


def validate(tail: int, alpha: float, alpha_end: float, ramp: int) -> None:
    if tail < 1:
        raise TaperError("the tail must have at least one frame")
    if ramp < 1 or ramp > tail:
        raise TaperError("the ramp must be between 1 and the tail length")
    if not 0.0 <= alpha <= 1.0 or not 0.0 <= alpha_end <= 1.0:
        raise TaperError("alpha and alpha_end must be between 0 and 1")
    if alpha_end > alpha:
        raise TaperError("alpha_end must not be larger than alpha (it is a taper towards the seam)")


def schedule(tail: int, alpha: float, alpha_end: float, ramp: int) -> list[float]:
    validate(tail, alpha, alpha_end, ramp)
    return [alpha_for(p, tail, alpha, alpha_end, ramp) for p in range(tail)]


def new_rng(seed: int) -> random.Random:
    """One stream for every injected frame, as the recipe does."""
    return random.Random(int(seed))


def palette_grid(rng: random.Random, grid_w: int, grid_h: int) -> list:
    """Palette colours drawn row-major (the recipe's order, so a seed reproduces it), [grid_h][grid_w][3]."""
    return [[rng.choice(PALETTE) for _ in range(grid_w)] for _ in range(grid_h)]


def nearest_index_map(out_size: int, in_size: int) -> list[int]:
    """Source indices of a nearest-neighbour resize (PIL's convention: pure block replication at integer scales)."""
    scale = in_size / out_size
    return [min(in_size - 1, int((i + 0.5) * scale)) for i in range(out_size)]


def resolve_grid(width: int, height: int, grid_mode: str, block_size: int) -> tuple[int, int]:
    if grid_mode == "poc_36x64":
        return POC_GRID
    bs = max(1, int(block_size))
    return (max(1, round(width / bs)), max(1, round(height / bs)))


def steps_for_frames(n: int) -> int | None:
    """Latent steps covering exactly n pixel frames from a cycle start, else None (5/22/39/56 -> 2/7/12/17)."""
    k, covered = 0, 0
    while covered < n:
        covered += FRAME_PER_TOKEN[k % 5]
        k += 1
    return k if covered == n else None


def format_schedule(values: list[float], unit: str = "frame", header: str = "") -> str:
    """Runs of equal alpha collapse into one line, so a 22-frame tail reads as four lines."""
    lines = [header] if header else []
    i, total = 0, len(values)
    while i < total:
        j = i
        while j + 1 < total and abs(values[j + 1] - values[i]) < 1e-9:
            j += 1
        span = f"{unit} {i}" if i == j else f"{unit} {i}..{j}"
        lines.append(f"  {span:<18} alpha={values[i]:.3f}{'   <- seam' if j == total - 1 else ''}")
        i = j + 1
    return "\n".join(lines)
