# Copyright 2025-2026 Code2Collapse — ComfyUI-WanAnimatePreprocessV2
# Licensed under the Apache License, Version 2.0
# PORTED FROM: ComfyUI-WanAnimatePreprocessV2 :: nodes.py @ feat/v2-ai-spine
#
# Subset: One-Euro / EMA smoothing, lock_anchor_size, build_jitterless_boxes (centre path).

from __future__ import annotations

import math
from typing import Optional

import numpy as np

_JITTERLESS_MAX_CENTER_DRIFT_FRAC = 0.12


class OneEuroFilter:
    """Casiez et al. 2012 — minimal 1-D implementation for offline clips."""

    def __init__(self, freq: float = 24.0, min_cutoff: float = 1.0, beta: float = 0.05):
        self.freq = float(freq)
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self._x_prev: Optional[float] = None
        self._dx_prev: float = 0.0

    def __call__(self, x: float) -> float:
        if self._x_prev is None:
            self._x_prev = float(x)
            return float(x)
        dt = 1.0 / max(self.freq, 1e-6)
        dx = (float(x) - self._x_prev) / dt
        alpha_d = _smoothing_factor(dt, 1.0)
        dx_hat = alpha_d * dx + (1.0 - alpha_d) * self._dx_prev
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        alpha = _smoothing_factor(dt, cutoff)
        x_hat = alpha * float(x) + (1.0 - alpha) * self._x_prev
        self._x_prev = x_hat
        self._dx_prev = dx_hat
        return x_hat


def _smoothing_factor(dt: float, cutoff: float) -> float:
    tau = 1.0 / (2.0 * math.pi * max(cutoff, 1e-6))
    return 1.0 / (1.0 + tau / max(dt, 1e-9))


def _gaussian_window(window: int) -> np.ndarray:
    w = max(int(window) | 1, 3)
    xs = np.arange(w, dtype=np.float64) - (w // 2)
    sigma = max(w / 6.0, 0.5)
    k = np.exp(-(xs ** 2) / (2.0 * sigma ** 2))
    return k / k.sum()


def smooth_centers(
    centers_xy: np.ndarray,
    method: str = "one_euro",
    *,
    ema_strength: float = 0.6,
    image_diag: float = 1.0,
    one_euro_min_cutoff: float = 1.0,
    one_euro_beta: float = 0.05,
    gaussian_window: int = 7,
    zero_phase: bool = True,
    freq: float = 24.0,
) -> np.ndarray:
    centers_xy = np.asarray(centers_xy, dtype=np.float32)
    if centers_xy.ndim != 2 or centers_xy.shape[0] < 2 or method == "none":
        return centers_xy.copy()

    if zero_phase and method in ("ema", "one_euro"):
        kw = dict(
            ema_strength=ema_strength,
            image_diag=image_diag,
            one_euro_min_cutoff=one_euro_min_cutoff,
            one_euro_beta=one_euro_beta,
            gaussian_window=gaussian_window,
            zero_phase=False,
            freq=freq,
        )
        fwd = smooth_centers(centers_xy, method, **kw)
        bwd = smooth_centers(centers_xy[::-1], method, **kw)[::-1]
        return (0.5 * (fwd + bwd)).astype(np.float32)

    if method == "ema":
        out = np.empty_like(centers_xy)
        out[0] = centers_xy[0]
        norm = max(1.0, image_diag)
        base = float(np.clip(ema_strength, 0.0, 1.0))
        for i in range(1, len(centers_xy)):
            curr = centers_xy[i]
            prev = out[i - 1]
            motion = float(np.mean(np.abs(curr - prev)) / norm)
            dyn = base * np.exp(-motion * 5.0)
            alpha = 1.0 - dyn
            out[i] = alpha * curr + (1.0 - alpha) * prev
        return out

    if method == "gaussian":
        k = _gaussian_window(gaussian_window)
        pad = len(k) // 2
        padded = np.pad(centers_xy, ((pad, pad), (0, 0)), mode="edge")
        out = np.empty_like(centers_xy)
        out[:, 0] = np.convolve(padded[:, 0], k, mode="valid")
        out[:, 1] = np.convolve(padded[:, 1], k, mode="valid")
        return out

    fx = OneEuroFilter(freq=freq, min_cutoff=one_euro_min_cutoff, beta=one_euro_beta)
    fy = OneEuroFilter(freq=freq, min_cutoff=one_euro_min_cutoff, beta=one_euro_beta)
    out = np.empty_like(centers_xy)
    for i in range(len(centers_xy)):
        out[i, 0] = fx(float(centers_xy[i, 0]))
        out[i, 1] = fy(float(centers_xy[i, 1]))
    return out


def lock_anchor_size(
    sizes_raw: np.ndarray,
    *,
    safety_margin: float = 1.12,
    max_side: float,
) -> float:
    """Round the crop side once so integer widths cannot wobble frame-to-frame."""
    if sizes_raw.size == 0:
        return float(max(8.0, min(max_side, 256.0)))
    anchor = float(np.median(np.asarray(sizes_raw, dtype=np.float64)))
    margin = float(safety_margin) if safety_margin and safety_margin > 0 else 1.0
    return float(np.clip(anchor * margin, 8.0, max_side))


def build_jitterless_boxes(
    *,
    target_centers: np.ndarray,
    anchor_size: float,
    crop_w: float,
    crop_h: float,
    W: int,
    H: int,
    smoothing_method: str = "one_euro",
    one_euro_min_cutoff: float = 1.0,
    one_euro_beta: float = 0.05,
    max_center_drift_frac: float = _JITTERLESS_MAX_CENTER_DRIFT_FRAC,
    hold_mask: Optional[np.ndarray] = None,
    freq: float = 24.0,
) -> tuple[list[tuple[float, float, float, float]], np.ndarray]:
    """Preview-only causal planner. Returns float boxes (x, y, bw, bh)."""
    target_centers = np.asarray(target_centers, dtype=np.float32).reshape(-1, 2).copy()
    b = int(target_centers.shape[0])
    if b == 0:
        return [], np.zeros((0, 2), np.float32)

    bw = float(crop_w)
    bh = float(crop_h)
    image_diag = float((W * W + H * H) ** 0.5)
    centers = smooth_centers(
        target_centers,
        smoothing_method,
        image_diag=image_diag,
        one_euro_min_cutoff=one_euro_min_cutoff,
        one_euro_beta=one_euro_beta,
        freq=freq,
    )

    if max_center_drift_frac and max_center_drift_frac > 0:
        limit = float(max_center_drift_frac) * float(max(bw, bh))
        for i in range(b):
            dx = float(centers[i, 0] - target_centers[i, 0])
            dy = float(centers[i, 1] - target_centers[i, 1])
            dist = math.hypot(dx, dy)
            if dist > limit > 0:
                k = limit / dist
                centers[i, 0] = target_centers[i, 0] + dx * k
                centers[i, 1] = target_centers[i, 1] + dy * k

    if hold_mask is not None:
        for i in range(1, b):
            if bool(hold_mask[i]):
                centers[i] = centers[i - 1]

    boxes: list[tuple[float, float, float, float]] = []
    for i in range(b):
        cx, cy = float(centers[i, 0]), float(centers[i, 1])
        x = cx - bw / 2.0
        y = cy - bh / 2.0
        x = min(max(x, 0.0), max(0.0, W - bw))
        y = min(max(y, 0.0), max(0.0, H - bh))
        boxes.append((float(x), float(y), float(bw), float(bh)))
    return boxes, centers
