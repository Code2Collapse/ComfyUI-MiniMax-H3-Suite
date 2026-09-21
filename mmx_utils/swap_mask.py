"""Build a swap's region mask from the same landmarks that drive its control.

WHY FROM THE SAME LANDMARKS
---------------------------
The mask says what may change; the control says what drives it. If those two
come from different sources - a mask rotoscoped by hand, a control from a
detector - they drift apart frame by frame, and the boundary walks relative to
the face. Deriving both from one landmark array makes them consistent by
construction, which is the only way to get an offset that stays at zero
without tuning it per shot.

WHY THE JAW IS IN HERE AND NOT IN THE CONTROL
---------------------------------------------
See MASK_GROUPS in swap_regions.py. Masked-but-not-controlled is the pairing
that lets the reference actor's jaw appear: the region is free to change, and
nothing is telling it to look like the dupe's.

No cv2, no SciPy. A convex hull and a scanline fill are forty lines and a
control mask is not worth a dependency that has already broken EXR reading in
this workspace once.
"""

from __future__ import annotations

import numpy as np

from .swap_regions import MASK_PAD, SwapRegionError, mask_indices


def _hull(points: np.ndarray) -> np.ndarray:
    """Monotone chain convex hull. points [N,2] -> hull [M,2] counter-clockwise."""
    if points.shape[0] < 3:
        return points
    pts = points[np.lexsort((points[:, 1], points[:, 0]))]

    def half(seq):
        out: list[np.ndarray] = []
        for p in seq:
            while len(out) >= 2:
                a, b = out[-2], out[-1]
                if (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]) > 0:
                    break
                out.pop()
            out.append(p)
        return out[:-1]

    return np.array(half(pts) + half(pts[::-1]), dtype=np.float32)


def _expand(hull: np.ndarray, pad: float) -> np.ndarray:
    """Push the hull out from its own centroid by `pad` of its size.

    Scaling about the centroid, rather than offsetting each edge, keeps the
    shape's centre exactly where it was - the same reason mask growth happens
    in token units elsewhere in this pack.
    """
    if pad <= 0 or hull.shape[0] == 0:
        return hull
    c = hull.mean(axis=0, keepdims=True)
    return c + (hull - c) * (1.0 + pad)


def _fill(hull: np.ndarray, w: int, h: int) -> np.ndarray:
    """Scanline-fill a convex polygon into an [h, w] float32 mask."""
    out = np.zeros((h, w), dtype=np.float32)
    if hull.shape[0] < 3:
        return out
    y0 = max(0, int(np.floor(hull[:, 1].min())))
    y1 = min(h - 1, int(np.ceil(hull[:, 1].max())))
    if y1 < y0:
        return out
    n = hull.shape[0]
    for y in range(y0, y1 + 1):
        yc = y + 0.5
        xs: list[float] = []
        for i in range(n):
            ax, ay = hull[i]
            bx, by = hull[(i + 1) % n]
            if (ay <= yc < by) or (by <= yc < ay):
                xs.append(ax + (yc - ay) * (bx - ax) / (by - ay))
        if len(xs) < 2:
            continue
        lo = max(0, int(np.floor(min(xs))))
        hi = min(w - 1, int(np.ceil(max(xs))))
        if hi >= lo:
            out[y, lo:hi + 1] = 1.0
    return out


def _feather(mask: np.ndarray, radius: int) -> np.ndarray:
    """Separable box blur, twice - close enough to a gaussian for a matte edge."""
    if radius < 1:
        return mask
    k = 2 * radius + 1
    pad = np.pad(mask, radius, mode="edge")
    for _ in range(2):
        cs = np.cumsum(pad, axis=1)
        pad = np.concatenate([cs[:, k - 1:k], cs[:, k:] - cs[:, :-k]], axis=1) / k
        pad = np.pad(pad, ((0, 0), (radius, radius)), mode="edge")
        cs = np.cumsum(pad, axis=0)
        pad = np.concatenate([cs[k - 1:k, :], cs[k:, :] - cs[:-k, :]], axis=0) / k
        pad = np.pad(pad, ((radius, radius), (0, 0)), mode="edge")
    return pad[radius:radius + mask.shape[0], radius:radius + mask.shape[1]]


def build_swap_mask(
    keypoints: np.ndarray,
    scope: str,
    canvas: tuple[int, int],
    *,
    confidence_gate: float = 0.3,
    pad: float | None = None,
    feather: int = 0,
) -> np.ndarray:
    """[T,133,3] pixel keypoints -> [T,H,W] float32 mask in 0..1."""
    arr = np.asarray(keypoints, dtype=np.float32)
    if arr.ndim != 3:
        raise SwapRegionError(
            f"Expected [frames, 133, 3] keypoints, got shape {arr.shape}.")
    w, h = int(canvas[0]), int(canvas[1])
    if w <= 0 or h <= 0:
        raise SwapRegionError(f"Canvas must be positive, got {w}x{h}.")

    idx = mask_indices(scope)
    amount = MASK_PAD[scope] if pad is None else float(pad)
    out = np.zeros((arr.shape[0], h, w), dtype=np.float32)

    for t in range(arr.shape[0]):
        pts = arr[t, idx]
        good = pts[pts[:, 2] >= confidence_gate][:, :2]
        if good.shape[0] < 3:
            continue              # a frame with no detection is a hole, not a guess
        out[t] = _fill(_expand(_hull(good), amount), w, h)
        if feather > 0:
            out[t] = _feather(out[t], int(feather))
    return np.clip(out, 0.0, 1.0)


def describe_mask(masks: np.ndarray, scope: str) -> str:
    arr = np.asarray(masks)
    covered = float((arr > 0.5).mean()) * 100.0
    empty = int((arr.reshape(arr.shape[0], -1).max(axis=1) <= 0.0).sum())
    lines = [
        f"Mask for scope '{scope}': {covered:.1f}% of frame area on average.",
    ]
    if empty:
        lines.append(
            f"{empty} frame(s) have NO mask - the detector found fewer than "
            "three landmarks there. Those frames will not be regenerated at "
            "all, which shows as the swap flicking back to the original.")
    return "\n".join(lines)
