"""Rasterise a swap's driving landmarks into a control video.

The Fun ControlNet-Union takes Canny, Depth, HED, MLSD or Pose. This renders
the Pose form, but only for the landmark groups a given swap scope drives -
see swap_regions.py for why the jaw is never one of them.

The edge tables below are per GROUP, so excluding a group excludes its lines
by construction rather than by a filter someone can forget to apply. There is
no jaw chain in this file at all.

Detector-agnostic: it takes COCO-WholeBody 133 keypoints, which is what
DWPose, SDPose and ViTPose-wholebody all emit. SDPose (arXiv 2509.24980) is
the stronger detector - 72.8 AP on COCO-WholeBody against DWPose's ~66, and
notably better out of domain - but it is a different source for the same
array, not a different contract.
"""

from __future__ import annotations

import numpy as np

from .swap_regions import FACE, GROUPS, N_WHOLEBODY, SwapRegionError, scope_groups, select

_F = FACE[0]


def _chain(*idx: int) -> tuple[tuple[int, int], ...]:
    return tuple(zip(idx[:-1], idx[1:]))


def _loop(start: int, stop: int) -> tuple[tuple[int, int], ...]:
    pts = list(range(start, stop))
    return tuple(zip(pts, pts[1:])) + ((pts[-1], pts[0]),)


def _face(edges: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    return tuple((a + _F, b + _F) for a, b in edges)


# COCO-17 body. Head points 0-4 are deliberately omitted: the face groups carry
# the head, and the COCO nose/eye/ear points are far coarser than the 68-point
# face block that sits beside them.
BODY_EDGES = (
    (5, 7), (7, 9), (6, 8), (8, 10),          # arms
    (5, 6), (5, 11), (6, 12), (11, 12),        # torso
    (11, 13), (13, 15), (12, 14), (14, 16),    # legs
)

_HAND_CHAINS = ((0, 1, 2, 3, 4), (0, 5, 6, 7, 8), (0, 9, 10, 11, 12),
                (0, 13, 14, 15, 16), (0, 17, 18, 19, 20))


def _hand(base: int) -> tuple[tuple[int, int], ...]:
    out: list[tuple[int, int]] = []
    for chain in _HAND_CHAINS:
        out.extend((a + base, b + base) for a, b in _chain(*chain))
    return tuple(out)


# NOTE: there is no "jaw" entry, and adding one would defeat the whole design.
GROUP_EDGES: dict[str, tuple[tuple[int, int], ...]] = {
    "body": BODY_EDGES,
    "feet": ((15, 17), (15, 18), (15, 19), (16, 20), (16, 21), (16, 22)),
    "left_hand": _hand(GROUPS["left_hand"][0]),
    "right_hand": _hand(GROUPS["right_hand"][0]),
    "brows": _face(_chain(17, 18, 19, 20, 21) + _chain(22, 23, 24, 25, 26)),
    "nose": _face(_chain(27, 28, 29, 30) + _chain(31, 32, 33, 34, 35)),
    "eyes": _face(_loop(36, 42) + _loop(42, 48)),
    "mouth": _face(_loop(48, 60) + _loop(60, 68)),
}

# Per-group colour. A control image is read by the model as pixels, so giving
# each group its own hue is what lets it tell a mouth line from an eyebrow -
# the same reason OpenPose renders limbs in distinct colours.
GROUP_COLOUR: dict[str, tuple[float, float, float]] = {
    "body": (0.00, 0.60, 1.00),
    "feet": (0.00, 0.85, 0.75),
    "left_hand": (1.00, 0.70, 0.10),
    "right_hand": (0.95, 0.45, 0.10),
    "brows": (0.70, 0.45, 1.00),
    "nose": (0.30, 0.90, 0.45),
    "eyes": (1.00, 0.95, 0.35),
    "mouth": (1.00, 0.30, 0.45),
}


def scope_edges(scope: str) -> list[tuple[tuple[int, int], tuple[float, float, float]]]:
    """Every (edge, colour) a scope draws. Never contains a jaw edge."""
    out = []
    for group in scope_groups(scope):
        colour = GROUP_COLOUR[group]
        for edge in GROUP_EDGES.get(group, ()):
            out.append((edge, colour))
    return out


def _line(img: np.ndarray, x0: int, y0: int, x1: int, y1: int,
          colour: tuple[float, float, float], width: int) -> None:
    """Bresenham with a square nib. No SciPy, no cv2 - this must import on a
    machine with neither, because a control render is not worth a dependency."""
    h, w = img.shape[:2]
    dx, dy = abs(x1 - x0), -abs(y1 - y0)
    sx, sy = (1 if x0 < x1 else -1), (1 if y0 < y1 else -1)
    err = dx + dy
    r = max(0, width // 2)
    while True:
        lo_y, hi_y = max(0, y0 - r), min(h, y0 + r + 1)
        lo_x, hi_x = max(0, x0 - r), min(w, x0 + r + 1)
        if hi_y > lo_y and hi_x > lo_x:
            img[lo_y:hi_y, lo_x:hi_x] = colour
        if x0 == x1 and y0 == y1:
            return
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x0 += sx
        if e2 <= dx:
            err += dx
            y0 += sy


def render_swap_control(
    keypoints: np.ndarray,
    scope: str,
    canvas: tuple[int, int],
    *,
    confidence_gate: float = 0.3,
    line_width: int = 4,
    face_line_width: int | None = None,
) -> np.ndarray:
    """[T,133,3] COCO-WholeBody -> [T,H,W,3] float32 control frames in 0..1.

    `keypoints` are PIXEL coordinates on `canvas`. Landmarks outside the scope
    are suppressed by `select` (their score goes to zero) and then dropped by
    the confidence gate, so an excluded group cannot be drawn even if a caller
    passes its edges by mistake.
    """
    arr = np.asarray(keypoints, dtype=np.float32)
    if arr.ndim != 3:
        raise SwapRegionError(
            f"Expected [frames, {N_WHOLEBODY}, 3] keypoints, got shape {arr.shape}.")
    w, h = int(canvas[0]), int(canvas[1])
    if w <= 0 or h <= 0:
        raise SwapRegionError(f"Canvas must be positive, got {w}x{h}.")

    gated = select(arr, scope)
    edges = scope_edges(scope)
    # Face lines default thinner: a 4px mouth on a 768px frame closes the lip
    # gap and the model then cannot tell an open mouth from a shut one.
    fw = face_line_width if face_line_width is not None else max(1, line_width // 2)
    face_groups = {"brows", "nose", "eyes", "mouth"}
    face_edge_ids = {e for g in face_groups for e in GROUP_EDGES.get(g, ())}

    out = np.zeros((arr.shape[0], h, w, 3), dtype=np.float32)
    for t in range(arr.shape[0]):
        pts = gated[t]
        for edge, colour in edges:
            a, b = edge
            if pts[a, 2] < confidence_gate or pts[b, 2] < confidence_gate:
                continue
            width = fw if edge in face_edge_ids else line_width
            _line(out[t],
                  int(round(pts[a, 0])), int(round(pts[a, 1])),
                  int(round(pts[b, 0])), int(round(pts[b, 1])),
                  colour, width)
    return out


def describe_render(scope: str, drawn: int, total: int) -> str:
    groups = scope_groups(scope)
    return (
        f"Rendered {drawn} of {total} possible edges for scope '{scope}' "
        f"({', '.join(groups)}).\n"
        "No jaw edge exists in the table, so the dupe's face contour cannot "
        "reach the model however the node is configured - the head shape stays "
        "the reference actor's."
    )
