"""Sizing planner for H3 latent upscaling at ANY aspect ratio (work order 2026-11-14 §7; spec
docs/research/H3_aspect_upscale_spec.md in the C2C workspace).

Pure maths, no torch: source size + sizing mode + target ratio + fit policy -> the legal H3 output size, the size the
content is resized to, and the crop / pad box, all in latent CELLS (1 cell = 16 px; H3 patches 2 x 2 cells, so every
side is an even number of cells = a 32 px grid). The H3 3D upscaler network is fully convolutional, so the ratio is
limited only by this grid and by memory - never by a fixed 16:9.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

PX_PER_CELL = 16
GRID_CELLS = 2                      # H3 patch 2 x 2 cells -> 32 px
MIN_PX = 32                         # as the node always clamped (max(32, ...))
MAX_MEGAPIXELS = 33.0
MAX_RATIO = 8.0

PRESETS = {"1:1": 1.0, "5:4": 5 / 4, "4:5": 4 / 5, "4:3": 4 / 3, "3:4": 3 / 4, "3:2": 3 / 2, "2:3": 2 / 3,
           "16:9": 16 / 9, "9:16": 9 / 16, "1.85:1": 1.85, "2:1": 2.0, "2.39:1": 2.39, "21:9": 21 / 9}
ASPECTS = ("source",) + tuple(PRESETS) + ("custom",)
FITS = ("auto", "contain", "crop", "pad", "stretch")
ANCHORS = ("center", "top", "bottom", "left", "right", "top-left", "top-right", "bottom-left", "bottom-right")
PAD_FILLS = ("edge", "zero latent")
SIZE_BY = ("width / height", "scale", "megapixels")
RATIO_TOLERANCE = 0.01              # 'auto' treats ratios within 1 % (plus the grid's own error) as matching


class SizingError(ValueError):
    """A sizing request that cannot be planned, in plain words."""


@dataclass
class Plan:
    out_cells: Tuple[int, int]          # (h, w) of the output latent
    content_cells: Tuple[int, int]      # (h, w) the source is resized to before crop / pad
    offset_cells: Tuple[int, int]       # crop: where the output window starts in the content; pad: where the content
    mode: str                           #   sits in the output.  mode: exact / stretch / contain / crop / pad
    requested_px: Tuple[int, int]       # (w, h) before the grid
    target_ratio: float
    notes: List[str] = field(default_factory=list)
    box_cells: Optional[Tuple[int, int]] = None   # (h, w) the request snapped to the grid (contain fits inside it)

    @property
    def out_px(self) -> Tuple[int, int]:
        return self.out_cells[1] * PX_PER_CELL, self.out_cells[0] * PX_PER_CELL

    @property
    def actual_ratio(self) -> float:
        return self.out_cells[1] / self.out_cells[0]

    def deviation_pct(self) -> float:
        return abs(self.actual_ratio / self.target_ratio - 1.0) * 100.0

    def report(self) -> str:
        rw, rh = self.requested_px
        ow, oh = self.out_px
        ch, cw = self.content_cells
        line = (f"size: requested {rw}x{rh} px (ratio {self.target_ratio:.4f}) -> {ow}x{oh} px = "
                f"{self.out_cells[1]}x{self.out_cells[0]} cells (ratio {self.actual_ratio:.4f}, "
                f"{self.deviation_pct():.2f} % off); fit {self.mode}")
        if self.mode == "crop":
            line += (f": content {cw * PX_PER_CELL}x{ch * PX_PER_CELL} px, window at cell "
                     f"(x {self.offset_cells[1]}, y {self.offset_cells[0]})")
        elif self.mode == "pad":
            line += (f": content {cw * PX_PER_CELL}x{ch * PX_PER_CELL} px placed at cell "
                     f"(x {self.offset_cells[1]}, y {self.offset_cells[0]})")
        return "\n".join([line] + self.notes)


def parse_ratio(text: str) -> float:
    """'2.39:1', '16:9', '1.85' -> float. Errors name the value."""
    s = str(text or "").strip().replace("x", ":").replace("/", ":")
    try:
        if ":" in s:
            a, b = s.split(":", 1)
            r = float(a) / float(b)
        else:
            r = float(s)
    except (ValueError, ZeroDivisionError):
        raise SizingError(f"custom_ratio '{text}' is not a ratio: write it as W:H (e.g. 2.39:1 or 4:5)") from None
    if not math.isfinite(r) or r <= 0:
        raise SizingError(f"custom_ratio '{text}' must be positive")
    if r > MAX_RATIO or r < 1 / MAX_RATIO:
        raise SizingError(f"custom_ratio '{text}' is beyond {MAX_RATIO:g}:1 - not supported")
    return r


def target_ratio(aspect: str, custom_ratio: str, src_w: int, src_h: int) -> float:
    if aspect == "source":
        return src_w / src_h
    if aspect == "custom":
        return parse_ratio(custom_ratio)
    if aspect not in PRESETS:
        raise SizingError(f"aspect must be one of {', '.join(ASPECTS)}; got {aspect!r}")
    return PRESETS[aspect]


def _even(v: float) -> int:
    return max(GRID_CELLS, int(round(v / GRID_CELLS)) * GRID_CELLS)


def snap(w_px: float, h_px: float, ratio: Optional[float] = None,
         within: Optional[Tuple[int, int]] = None) -> Tuple[int, int]:
    """(h, w) cells on the grid nearest the request; when a ratio is given, the candidate with the smallest ratio
    error wins among the neighbours (then the one closest in size). ``within`` = (h, w) cells the result must not
    exceed (contain / pad: the picture has to fit inside the box)."""
    w_c, h_c = w_px / PX_PER_CELL, h_px / PX_PER_CELL
    if ratio is None and within is None:
        return _even(h_c), _even(w_c)
    cands = set()
    for dw in (-GRID_CELLS, 0, GRID_CELLS):
        for dh in (-GRID_CELLS, 0, GRID_CELLS):
            cw, ch = _even(w_c) + dw, _even(h_c) + dh
            if cw >= GRID_CELLS and ch >= GRID_CELLS and (within is None or (ch <= within[0] and cw <= within[1])):
                cands.add((ch, cw))
    if not cands:
        return min(_even(h_c), within[0]), min(_even(w_c), within[1])
    if ratio is None:
        ratio = w_c / h_c
    area = w_c * h_c
    return min(cands, key=lambda hw: (round(abs(hw[1] / hw[0] / ratio - 1.0), 6), abs(hw[0] * hw[1] - area)))


def _offset(space: int, anchor: str, axis: str) -> int:
    """Cells before the window / content along one axis, on the grid."""
    a = anchor.split("-")
    lo, hi = ("top", "bottom") if axis == "y" else ("left", "right")
    if lo in a:
        return 0
    if hi in a:
        return space
    return (space // 2 // GRID_CELLS) * GRID_CELLS


def plan(src_cells: Tuple[int, int], size_by: str = SIZE_BY[0], width: int = 1280, height: int = 704,
         scale: float = 2.0, megapixels: float = 1.0, aspect: str = "source", custom_ratio: str = "2.39:1",
         fit: str = "auto", anchor: str = "center") -> Plan:
    src_h, src_w = int(src_cells[0]), int(src_cells[1])
    if src_h <= 0 or src_w <= 0:
        raise SizingError("the latent has no spatial size")
    if fit not in FITS:
        raise SizingError(f"fit must be one of {', '.join(FITS)}; got {fit!r}")
    if anchor not in ANCHORS:
        raise SizingError(f"anchor must be one of {', '.join(ANCHORS)}; got {anchor!r}")
    sw, sh = src_w * PX_PER_CELL, src_h * PX_PER_CELL
    src_r = sw / sh
    notes: List[str] = []
    if size_by == SIZE_BY[0]:
        bw, bh = float(width), float(height)
        if bw <= 0 or bh <= 0:
            raise SizingError("width and height must be positive")
        r = bw / bh
        if aspect not in ("source", None):
            notes.append(f"aspect '{aspect}' is ignored with size_by = width / height (the box sets the ratio)")
    else:
        r = target_ratio(aspect, custom_ratio, sw, sh)
        if size_by == "scale":
            if not math.isfinite(float(scale)) or float(scale) <= 0:
                raise SizingError("scale must be a positive number")
            k = float(scale)
        elif size_by == "megapixels":
            if not math.isfinite(float(megapixels)) or float(megapixels) <= 0:
                raise SizingError("megapixels must be a positive number")
            k = math.sqrt(float(megapixels) * 1e6 / float(sw * sh))
        else:
            raise SizingError(f"size_by must be one of {', '.join(SIZE_BY)}; got {size_by!r}")
        if aspect == "source":                              # the pre-2026-11-14 maths, so old saves size identically
            bw, bh = float(max(32, round(sw * k))), float(max(32, round(sh * k)))
        else:
            area = sw * sh * k * k
            bw, bh = math.sqrt(area * r), math.sqrt(area / r)
    if bw * bh > MAX_MEGAPIXELS * 1e6:
        raise SizingError(f"{bw:.0f}x{bh:.0f} px is {bw * bh / 1e6:.1f} MP - above the {MAX_MEGAPIXELS:g} MP limit")
    if bw < MIN_PX or bh < MIN_PX:
        notes.append(f"raised to at least {MIN_PX} px per side")
        bw, bh = max(bw, MIN_PX), max(bh, MIN_PX)
    requested = (int(round(bw)), int(round(bh)))
    # the old code rounded each side on its own; that stays for the old cases (a width / height box, or the source
    # ratio), so saved workflows get the same size. A chosen ratio snaps to the grid neighbour closest to that ratio.
    legacy = size_by == SIZE_BY[0] or aspect == "source"
    box = snap(bw, bh, None if legacy else r)               # (h, w) cells
    grid_err = abs((box[1] / box[0]) / r - 1.0)
    same = abs(r / src_r - 1.0) <= RATIO_TOLERANCE + grid_err + 1e-9

    mode = fit
    if fit == "auto":
        if not same:
            raise SizingError(
                f"the target ratio {r:.4f} differs from the source ratio {src_r:.4f}. Choose how to fit: "
                "fit = contain (whole picture, smaller output), crop (fill, cut at 'anchor'), pad (whole picture, "
                "padded) or stretch (distort - the behaviour before 2026-11-14).")
        mode = "exact"
    elif same and fit in ("crop", "pad", "contain"):
        mode = "exact"                                      # nothing to crop / pad: same ratio

    if mode in ("exact", "stretch"):
        return Plan(box, box, (0, 0), mode, requested, r, notes)
    if mode == "contain":
        k = min((box[1] * PX_PER_CELL) / sw, (box[0] * PX_PER_CELL) / sh)
        cont = snap(sw * k, sh * k, src_r, within=box)
        return Plan(cont, cont, (0, 0), mode, requested, r, notes + [
            "contain: the output keeps the source ratio inside the requested box (no crop, no padding)"], box)
    if mode == "crop":
        k = max((box[1] * PX_PER_CELL) / sw, (box[0] * PX_PER_CELL) / sh)
        cont = (max(box[0], _even(sh * k / PX_PER_CELL)), max(box[1], _even(sw * k / PX_PER_CELL)))
        off = (_offset(cont[0] - box[0], anchor, "y"), _offset(cont[1] - box[1], anchor, "x"))
        return Plan(box, cont, off, mode, requested, r, notes)
    # pad
    k = min((box[1] * PX_PER_CELL) / sw, (box[0] * PX_PER_CELL) / sh)
    cont = snap(sw * k, sh * k, src_r, within=box)
    off = (_offset(box[0] - cont[0], anchor, "y"), _offset(box[1] - cont[1], anchor, "x"))
    return Plan(box, cont, off, mode, requested, r, notes + [
        "pad: the padded border must be regenerated by the next sampler pass (its noise-mask cells are set to 1)"])
