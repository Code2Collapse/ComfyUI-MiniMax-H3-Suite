# SPDX-License-Identifier: GPL-3.0-or-later
# PORTED FROM: open-source-higgsfield (lucataco, Apache-2.0) :: bodydouble/media.py, bodydouble/qc.py @ 864c8de
#   (third_party/open-source-higgsfield; licence + NOTICE in THIRD_PARTY_LICENSES/open-source-higgsfield-*)
"""Character swap / body double finishing algorithms (owner A21 Rev 3 Part 5 + 10 + 13; ledger L9.78).

The reference recasts a performer with SCAIL-2. These are its model-independent steps, on ComfyUI's float images
(H x W x C in 0..1) instead of 8-bit PIL frames; every threshold is rescaled from the 0..255 originals.

- follow-camera crop plan (``plan_crops``);
- appearance re-identification (``color_hist`` / ``hist_similarity``);
- mesh-warp alignment of a generated crop onto its source crop (``warp`` / ``align`` / ``steady_warps``);
- cast matte reconstruction (``cast_matte``), mask dropout bridging in plate space (``steady_masks``);
- per-channel background colour match smoothed over time (``background_stats`` / ``color_transforms``);
- temporal deflicker (``deflicker``);
- seam ratio against the source's own ratio, background luma (QC).

What is NOT carried over (Part 5): the reference's 81-frame segments, 4n+1 padding, 9-frame minimum - the H3 planner owns
its grid. The 9 x 5 knot layout and the QC limits are the reference's choices for SCAIL-2: here they are parameters,
unmeasured on H3 output.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


# ── small helpers ──────────────────────────────────────────────────────────────────────────────────────────────────

def _np(x) -> np.ndarray:
    return x.detach().cpu().float().numpy() if isinstance(x, torch.Tensor) else np.asarray(x, dtype=np.float32)


def smooth(values, radius: int) -> np.ndarray:
    """Centred moving average that holds the ends."""
    v = np.asarray(values, dtype=np.float64)
    if radius <= 0 or v.size == 0:
        return v.copy()
    padded = np.concatenate([np.full(radius, v[0]), v, np.full(radius, v[-1])])
    kernel = np.ones(2 * radius + 1) / (2 * radius + 1)
    return np.convolve(padded, kernel, mode="valid")


def luma(img) -> np.ndarray:
    a = _np(img)
    return a[..., :3] @ np.array([0.299, 0.587, 0.114], np.float32) if a.ndim == 3 else a


def resize2d(a: np.ndarray, h: int, w: int, mode: str = "bilinear") -> np.ndarray:
    """H x W (x C) float array resized; area for shrinking."""
    t = torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32))
    t = t[None, None] if t.ndim == 2 else t.permute(2, 0, 1)[None]
    if h < t.shape[-2] and w < t.shape[-1] and mode == "bilinear":
        out = F.interpolate(t, size=(h, w), mode="area")
    else:
        out = F.interpolate(t, size=(h, w), mode=mode, **({"align_corners": False} if mode in ("bilinear", "bicubic") else {}))
    out = out[0]
    return (out[0] if a.ndim == 2 else out.permute(1, 2, 0)).numpy()


def dilate(mask: np.ndarray, k: int) -> np.ndarray:
    """Binary max filter with an odd k x k window."""
    k = max(1, int(k) | 1)
    t = torch.from_numpy(np.ascontiguousarray(mask, dtype=np.float32))[None, None]
    return F.max_pool2d(t, k, stride=1, padding=k // 2)[0, 0].numpy()


def bbox(mask, thresh: float = 0.5):
    ys, xs = np.nonzero(_np(mask) > thresh)
    if len(xs) == 0:
        return None
    return float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)


# ── re-identification ───────────────────────────────────────────────────────────────────────────────────────────

def color_hist(frame, mask, bins: int = 8) -> np.ndarray:
    """Normalised RGB histogram of the pixels under a mask: the performer's appearance signature."""
    px = (np.clip(_np(frame)[..., :3], 0, 1) * 255.0).astype(np.int64)[_np(mask) > 0.5]
    if len(px) == 0:
        return np.zeros(bins ** 3)
    q = px // (256 // bins)
    h = np.bincount(q[:, 0] * bins * bins + q[:, 1] * bins + q[:, 2], minlength=bins ** 3).astype(np.float64)
    return h / h.sum()


def hist_similarity(a, b) -> float:
    """Bhattacharyya coefficient: 1 = identical."""
    return float(np.sqrt(np.asarray(a) * np.asarray(b)).sum())


def reacquire_scores(frame, candidates, ref_hist, last_box, width: float, distance_weight: float = 0.5) -> list[float]:
    """Score each candidate mask: appearance similarity to the performer minus ``distance_weight`` x how far its centre
    is from where the performer was (as a fraction of the frame width). The reference picks the best."""
    out = []
    for m in candidates:
        b = bbox(m)
        if b is None:
            out.append(float("-inf"))
            continue
        dist = abs((b[0] + b[2]) / 2 - (last_box[0] + last_box[2]) / 2) / width if last_box else 0.0
        out.append(hist_similarity(color_hist(frame, m), ref_hist) - distance_weight * dist)
    return out


# ── follow-camera crop ──────────────────────────────────────────────────────────────────────────────────────────

def plan_crops(boxes, src_w: int, src_h: int, aspect: float, margin: float = 1.45, min_h: float | None = None,
               radius: int = 12) -> list[list[float]]:
    """One crop rect [x, y, w, h] per frame (w / h = aspect) that follows the performer smoothly. ``boxes`` are per-frame
    (x0, y0, x1, y1) in source pixels, or None where the track lost them (held from the nearest tracked frame). The zoom
    is smoothed but never much tighter than the local need, so limbs are not cut when the performer opens up."""
    idx = [i for i, b in enumerate(boxes) if b is not None]
    if not idx:
        raise ValueError("the performer was never tracked")
    filled = [boxes[min(idx, key=lambda k: abs(k - i))] for i in range(len(boxes))]
    b = np.array(filled, dtype=np.float64)
    cx, cy = (b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2
    need_h = np.maximum((b[:, 3] - b[:, 1]) * margin, (b[:, 2] - b[:, 0]) * margin / aspect)
    max_h = min(src_h, src_w / aspect)
    need_h = np.clip(need_h, min_h or 0, max_h)
    h = np.maximum(smooth(need_h, radius), smooth(need_h, 2))
    cx, cy = smooth(cx, radius), smooth(cy, radius)
    rects = []
    for (x, y, hh), bb in zip(zip(cx, cy, h), b):
        hh = min(hh, max_h)
        ww = hh * aspect
        x0 = min(max(x - ww / 2, 0), src_w - ww)
        y0 = min(max(y - hh / 2, 0), src_h - hh)
        # Deviation from the reference: the smoothed centre lags a fast subject (8 px a frame lags ~25 px with
        # radius 12) and a tight margin then cut it. Slide the rect - same size - just enough to hold this frame's box.
        x0 = min(max(x0, bb[2] - ww), bb[0])
        y0 = min(max(y0, bb[3] - hh), bb[1])
        x0 = min(max(x0, 0), src_w - ww)
        y0 = min(max(y0, 0), src_h - hh)
        rects.append([round(float(x0), 2), round(float(y0), 2), round(float(ww), 2), round(float(hh), 2)])
    return rects


# ── mesh-warp alignment ─────────────────────────────────────────────────────────────────────────────────────────

def _hats(n: int, k: int) -> np.ndarray:
    """n x k weights that interpolate k knots spread evenly over 0..n linearly, at pixel centres."""
    knots = np.round(np.linspace(0, n, k))
    return np.stack([np.interp(np.arange(n) + 0.5, knots, e) for e in np.eye(k)], 1)


def warp(img, u, knots: tuple[int, int] = (9, 5), mode: str = "bilinear") -> np.ndarray:
    """Sample img at (x + dx(x), y + dy(y)); dx and dy are linear between the knots, u = (dx at each of the kx knots,
    dy at each of the ky knots) in pixels. Edge pixels are held outside the frame. (The reference resamples one PIL
    mesh; the displacements are separable, so one grid_sample is the same field.)"""
    a = _np(img)
    two_d = a.ndim == 2
    h, w = a.shape[:2]
    kx, ky = knots
    u = np.asarray(u, dtype=np.float64)
    dx = _hats(w, kx) @ u[:kx]
    dy = _hats(h, ky) @ u[kx:]
    xs = (np.arange(w) + 0.5 + dx) / w * 2 - 1
    ys = (np.arange(h) + 0.5 + dy) / h * 2 - 1
    gy, gx = np.meshgrid(ys, xs, indexing="ij")
    grid = torch.from_numpy(np.stack([gx, gy], -1).astype(np.float32))[None]
    t = torch.from_numpy(np.ascontiguousarray(a[..., None] if two_d else a))
    out = F.grid_sample(t.permute(2, 0, 1)[None], grid, mode=mode, padding_mode="border", align_corners=False)[0]
    out = out.permute(1, 2, 0).numpy()
    return out[..., 0] if two_d else out


def align(out, src, person, u=None, knots: tuple[int, int] = (9, 5), iters: int = 20, stiff: float = 0.1) -> np.ndarray:
    """Knot displacements (see warp) that put a generated crop back on the source crop it was made from: Gauss-Newton on
    the background luminance (away from the person mask), at quarter then half resolution, Huber-weighted so a redrawn
    face does not drag the fit, a curvature penalty carrying the fit across knots the person hides. Warm-start with the
    previous frame's u. Too little background (< 10 % of the crop): returns u unchanged."""
    kx, ky = knots
    u = np.zeros(kx + ky) if u is None else np.asarray(u, np.float64).copy()
    D = np.zeros((kx + ky, kx + ky))
    for k, part in ((kx, slice(0, kx)), (ky, slice(kx, None))):
        d = np.diff(np.eye(k), 2, axis=0)
        D[part, part] = d.T @ d
    lo, ls, pm = luma(out), luma(src), _np(person)
    H0, W0 = lo.shape
    for scale in (0.25, 0.5):
        sw, sh = max(8, round(W0 * scale)), max(8, round(H0 * scale))
        o = _blur(resize2d(lo, sh, sw), 1.0)
        s = _blur(resize2d(ls, sh, sw), 1.0)
        keep = dilate(resize2d(pm.astype(np.float32), sh, sw) > 0.5, int(40 * scale)) < 0.5
        b = max(1, round(16 * scale))
        keep[:b] = keep[-b:] = False
        keep[:, :b] = keep[:, -b:] = False
        if keep.mean() < 0.1:
            return u
        P, Q = _hats(sw, kx), _hats(sh, ky)
        v = u * scale
        for _ in range(iters):
            ow = warp(o, v, knots).astype(np.float64)
            gy, gx = np.gradient(ow)
            r = ow - s
            w = (keep & (np.abs(r) < 40 / 255)) / np.maximum(1, np.abs(r) / (8 / 255))
            xy = w * gx * gy
            Hm = np.block([[P.T @ ((w * gx * gx).sum(0)[:, None] * P), P.T @ xy.T @ Q],
                           [Q.T @ xy @ P, Q.T @ ((w * gy * gy).sum(1)[:, None] * Q)]])
            g = np.concatenate([P.T @ (w * gx * r).sum(0), Q.T @ (w * gy * r).sum(1)])
            lam = stiff * np.median(np.diag(Hm))
            dv = np.linalg.solve(Hm + lam * D + 1e-6 * max(np.diag(Hm).mean(), 1e-12) * np.eye(kx + ky), g + lam * D @ v)
            v -= dv
            if np.abs(dv).max() < 0.01:
                break
        u = v / scale
    return u


def _blur(a: np.ndarray, sigma: float) -> np.ndarray:
    r = max(1, int(round(sigma * 3)))
    x = np.arange(-r, r + 1)
    k = np.exp(-(x ** 2) / (2 * sigma ** 2))
    k /= k.sum()
    t = torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32))[None, None]
    kt = torch.from_numpy(k.astype(np.float32))
    t = F.conv2d(F.pad(t, (r, r, 0, 0), mode="replicate"), kt[None, None, None, :])
    t = F.conv2d(F.pad(t, (0, 0, r, r), mode="replicate"), kt[None, None, :, None])
    return t[0, 0].numpy()


def steady_warps(warps, lengths, radius: int = 6) -> np.ndarray:
    """Smooth per-frame align() fits over time, each segment (lengths, in frames) on its own: drift is slow inside a
    segment and jumps at the seams."""
    warps = np.array(warps, dtype=np.float64)
    start = 0
    for n in lengths:
        if n > 0:
            warps[start:start + n] = np.stack([smooth(c, radius) for c in warps[start:start + n].T], 1)
        start += n
    return warps


# ── mattes ─────────────────────────────────────────────────────────────────────────────────────────────────────

def cast_matte(crop, src, person, seen=None, reach: float = 0.07, thresh: float = 40 / 255, scale: float = 0.25) -> np.ndarray:
    """The person mask grown over the new cast where it sticks out of it: pixels where the ALIGNED generated crop differs
    from the source crop, connected to the person within ``reach`` (a fraction of the crop height), or anywhere inside
    ``seen`` (where a segmenter had the cast in nearby frames). Growing from the person keeps out redrawn faces further
    away. Returns a 0..1 matte at the crop's size."""
    c, s = _np(crop)[..., :3], _np(src)[..., :3]
    H, W = c.shape[:2]
    w, h = max(4, round(W * scale)), max(4, round(H * scale))
    diff = np.abs(resize2d(c, h, w) - resize2d(s, h, w)).max(2)
    grown = resize2d(_np(person).astype(np.float32), h, w) > 0.5
    allowed = grown | (diff > thresh)
    for _ in range(round(reach * h)):
        grown = (dilate(grown, 3) > 0) & allowed
    if seen is not None:
        allowed &= grown | (resize2d(_np(seen).astype(np.float32), h, w) > 0.5)
        while True:
            more = (dilate(grown, 3) > 0) & allowed
            if (more == grown).all():
                break
            grown = more
    return np.clip(resize2d(grown.astype(np.float32), H, W), 0, 1)


def steady_masks(masks, rects, plate_size: tuple[int, int], gap: int = 24, scale: float = 0.25, ends: bool = True):
    """Per-frame crop-space composite masks with short dropouts closed. ``masks``: per-frame crop-space masks (the source
    and cast masks' union, H x W, 0..1); ``rects``: the crop rect [x, y, w, h] of each frame in plate pixels. A plate
    pixel stays covered while it was covered within ``gap`` frames before AND within ``gap`` frames after - judged in
    plate space, where the set holds still while the follow-cam crop moves. ``ends``: the take's ends count as covered
    (the reference's choice, for a cast matte that grows over the first frames); it also paints where a MOVING subject
    goes within ``gap`` frames into the first frames, so the node leaves it off by default.

    Deviation from the reference (L9.78): only the newly bridged plate pixels are added to a frame's own mask. The
    reference re-adds the whole low-res cover, which grows every mask by up to half a cover cell, puts a soft fringe on
    every edge and lifts 0.99 values to 1 - harmless before its 0.5 threshold, wrong for a MASK output."""
    W, H = plate_size
    pw, ph = max(1, round(W * scale)), max(1, round(H * scale))
    n = len(rects)
    cover = np.zeros((n, ph, pw), bool)
    shapes = []
    for i in range(n):
        m = _np(masks[i])
        shapes.append(m.shape[:2])
        x0, y0, x1, y1 = _box(rects[i], pw, ph, scale)
        if x1 > x0 and y1 > y0:
            cover[i, y0:y1, x0:x1] = resize2d(m.astype(np.float32), y1 - y0, x1 - x0) > 0.5
    never = 10 ** 9
    seen = np.full((ph, pw), 0 if ends else -never, np.int64)
    last = np.empty((n, ph, pw), np.int64)
    for i in range(n):
        seen = np.where(cover[i], i, seen)
        last[i] = seen
    seen = np.full((ph, pw), n - 1 if ends else never, np.int64)
    for i in reversed(range(n)):
        seen = np.where(cover[i], i, seen)
        last[i] = (seen - last[i] <= gap) & ~cover[i]                                  # the newly bridged pixels
    for i in range(n):
        h, w = shapes[i]
        x0, y0, x1, y1 = _box(rects[i], pw, ph, scale)
        m = _np(masks[i])
        if x1 > x0 and y1 > y0 and last[i, y0:y1, x0:x1].any():
            m = np.maximum(m, np.clip(resize2d(last[i, y0:y1, x0:x1].astype(np.float32), h, w), 0, 1))
        yield m


def _box(rect, w: int, h: int, scale: float = 1.0) -> tuple[int, int, int, int]:
    x, y, rw, rh = (v * scale for v in rect)
    return max(0, round(x)), max(0, round(y)), min(w, round(x + rw)), min(h, round(y + rh))


# ── colour ────────────────────────────────────────────────────────────────────────────────────────────────────

def background_stats(img, masks, scale: float = 0.25, grow: float = 0.04) -> np.ndarray:
    """Per-channel mean and std of the pixels outside the dilated masks, at quarter resolution: [mean r g b, std r g b]."""
    a = _np(img)[..., :3]
    H, W = a.shape[:2]
    h, w = max(2, round(H * scale)), max(2, round(W * scale))
    small = resize2d(a, h, w)
    keep = np.ones((h, w), bool)
    k = max(3, int(h * grow) | 1)
    for m in masks:
        keep &= dilate(resize2d(_np(m).astype(np.float32), h, w) > 0.5, k) <= 0.5
    px = small[keep] if keep.any() else small.reshape(-1, 3)
    return np.concatenate([px.mean(0), px.std(0) + 1e-3 / 255])


def color_transforms(out_stats, src_stats, radius: int = 12) -> list[tuple[np.ndarray, np.ndarray]]:
    """Per-frame per-channel (gain, offset) mapping the generated background onto the source's, smoothed over time."""
    o, s = np.asarray(out_stats, np.float64), np.asarray(src_stats, np.float64)
    gain = np.clip(s[:, 3:] / o[:, 3:], 0.7, 1.4)
    offset = s[:, :3] - o[:, :3] * gain
    g = np.stack([smooth(gain[:, c], radius) for c in range(3)], 1)
    b = np.stack([smooth(offset[:, c], radius) for c in range(3)], 1)
    return list(zip(g, b))


def apply_color(img, transform) -> np.ndarray:
    g, b = transform
    a = _np(img)
    out = a.copy()
    out[..., :3] = np.clip(a[..., :3] * g + b, 0, 1)
    return out


# ── temporal deflicker ─────────────────────────────────────────────────────────────────────────────────────────

def _box_blur(a: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return a
    t = torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32)).permute(2, 0, 1)[None]
    t = F.avg_pool2d(F.pad(t, (radius,) * 4, mode="replicate"), 2 * radius + 1, stride=1)
    return t[0].permute(1, 2, 0).numpy()


def _deflicker_frame(prev, cur, nxt, amount, radius, lo, hi):
    med = np.median(np.stack((prev, cur, nxt), 0), 0)
    residual = cur - med
    coarse = _box_blur(residual, radius)
    still = np.clip((hi - np.abs(coarse).max(2, keepdims=True)) / (hi - lo), 0, 1)
    return np.clip(cur - amount * still * (residual - coarse), 0, 1)


def deflicker(frames, amount: float = 0.9, radius: int | None = None, lo: float = 4 / 255, hi: float = 8 / 255):
    """Remove fine temporal boil where nothing moves. Boil is random, so it averages out over the box; a coherent coarse
    residual is something moving faster than its own width (an ear, a glasses arm), which a 3-frame median would erase
    and ghost - the filter fades out between ``lo`` and ``hi`` of coarse residual. ``frames``: an iterable of H x W x 3
    float arrays 0..1; yields arrays in order. The first and last frames pass through. ``radius`` defaults to 8 px at
    704 rows and scales with height."""
    prev = cur = None
    for nxt in frames:
        nxt = _np(nxt)
        if radius is None:
            radius = max(2, round(nxt.shape[0] * 8 / 704))
        if cur is not None:
            yield np.clip(cur, 0, 1) if prev is None else _deflicker_frame(prev, cur, nxt, amount, radius, lo, hi)
        prev, cur = cur, nxt
    if cur is not None:
        yield np.clip(cur, 0, 1)


# ── QC ────────────────────────────────────────────────────────────────────────────────────────────────────────

def seam_ratio(lumas, b: int, window: int = 12) -> float:
    """Mean |dY| across a boundary over the median |dY| of the nearby frames. ``lumas``: per-frame luma arrays (quarter
    resolution is enough); ``b``: index of the first frame after the seam."""
    diffs = [float(np.abs(lumas[i] - lumas[i - 1]).mean()) for i in range(1, len(lumas))]
    across = diffs[b - 1]
    nearby = [d for i, d in enumerate(diffs) if abs(i - (b - 1)) <= window and i != b - 1]
    return across / max(float(np.median(nearby)) if nearby else 0.0, 1e-3 / 255)


def background_luma(img, masks, scale: float = 0.25, grow: float = 0.04) -> float:
    """Mean luma outside the dilated masks (background drift = output / source - 1)."""
    y = luma(img)
    H, W = y.shape
    h, w = max(2, round(H * scale)), max(2, round(W * scale))
    ys = resize2d(y, h, w)
    m = None
    k = max(3, int(h * grow) | 1)
    for mk in masks:
        a = dilate(resize2d(_np(mk).astype(np.float32), h, w) > 0.5, k) > 0.5
        m = a if m is None else (m | a)
    bg = ys[~m] if m is not None and (~m).any() else ys.ravel()
    return float(bg.mean())
