# PORTED FROM: https://github.com/HerrgottMargott/Herrgotts-H3-Infinite-Continuation-Suite@4b1edd6 motion_analysis.py
# (analyze_freeze_tail, the v0.4.6 / v1.4 "stable-tail consensus" detector), GPL-3.0. The detector is kept with its
# calibrated defaults; the suite's own H3 Continuation decides where to cut (a legal 5 + 17k length), so the upstream
# phase-snapping helpers are not needed.
"""Find H3's frozen ending: FL2VA often slows down, locks onto (nearly) the final image and repeats it, sometimes with
a brightness landing. Continuing from those frames carries the freeze into the next segment.

A trailing run counts as frozen only if BOTH hold all the way to the clip end:
1. each frame matches a robust median of the last frames (the stable final state), within strict mean and
   changed-area limits, with only short runs of outliers;
2. the frame-to-frame changes inside the run are near-static.
Slow, sustained convergence towards the last frame is therefore not called a freeze.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _prepare(images: torch.Tensor, max_edge: int) -> torch.Tensor:
    x = images[..., :3].detach().float().movedim(-1, 1)
    h, w = int(x.shape[-2]), int(x.shape[-1])
    scale = min(1.0, max(32, int(max_edge)) / float(max(h, w)))
    th, tw = max(16, int(round(h * scale))), max(16, int(round(w * scale)))
    if (th, tw) != (h, w):
        x = F.interpolate(x, size=(th, tw), mode="bilinear", align_corners=False, antialias=True)
    return F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)      # suppresses single-pixel shimmer


def _gray(rgb: torch.Tensor) -> torch.Tensor:
    return 0.2126 * rgb[:, 0:1] + 0.7152 * rgb[:, 1:2] + 0.0722 * rgb[:, 2:3]


def _max_false_streak(flags) -> int:
    best = cur = 0
    for f in flags:
        cur = 0 if f else cur + 1
        best = max(best, cur)
    return best


def analyze_freeze_tail(images: torch.Tensor, analysis_window: int = 72, freeze_hold: int = 12,
                        analysis_size: int = 192, final_mean_diff_threshold: float = 0.0120,
                        final_active_pixel_threshold: float = 0.025, max_final_active_area_percent: float = 3.0,
                        transition_mean_diff_threshold: float = 0.0020, transition_active_pixel_threshold: float = 0.010,
                        max_transition_active_area_percent: float = 1.0, min_static_transition_percent: float = 70.0,
                        max_consecutive_motion_outliers: int = 2, final_reference_frames: int = 15,
                        min_final_match_percent: float = 75.0, max_consecutive_final_outliers: int = 3) -> dict:
    """{freeze_detected, freeze_start_frame (-1 if none), trailing_frames, confidence, reason} for IMAGE [T,H,W,C]."""
    if images.ndim != 4:
        raise ValueError(f"expected IMAGE [frames, H, W, C], got {tuple(images.shape)}")
    n = int(images.shape[0])
    if n < 2:
        return {"freeze_detected": False, "freeze_start_frame": -1, "trailing_frames": 0, "confidence": 0.0,
                "reason": "too few frames"}
    window = max(2, min(n, int(analysis_window)))
    hold = max(2, int(freeze_hold))
    f_area = max(0.0, max_final_active_area_percent) / 100.0
    t_area = max(0.0, max_transition_active_area_percent) / 100.0
    min_static = max(0.0, min(1.0, min_static_transition_percent / 100.0))
    min_match = max(0.0, min(1.0, min_final_match_percent / 100.0))
    ref = _prepare(images[max(0, n - max(1, int(final_reference_frames))):], analysis_size).median(dim=0, keepdim=True).values

    def stats(start: int) -> dict:
        rgb = _prepare(images[start:], analysis_size)
        gray = _gray(rgb)
        fd = (rgb - ref).abs()
        f_mean = fd.mean(dim=(1, 2, 3))
        f_act = (fd.mean(dim=1) >= final_active_pixel_threshold).float().mean(dim=(1, 2))
        td = (gray[1:] - gray[:-1]).abs()
        t_mean = td.mean(dim=(1, 2, 3))
        t_act = (td[:, 0] >= transition_active_pixel_threshold).float().mean(dim=(1, 2))
        return {"final_match": ((f_mean <= final_mean_diff_threshold) & (f_act <= f_area)).tolist(),
                "final_mean": f_mean.tolist(), "final_active": f_act.tolist(),
                "trans_static": ((t_mean <= transition_mean_diff_threshold) & (t_act <= t_area)).tolist(),
                "trans_mean": t_mean.tolist(), "trans_active": t_act.tolist()}

    def primary(st, i):
        flags = st["final_match"][i:]
        if not flags:
            return False, 0.0
        ratio = sum(1 for x in flags if x) / float(len(flags))
        return bool(flags[0] and ratio >= min_match and _max_false_streak(flags) <= max_consecutive_final_outliers), ratio

    def residual(st, i):
        flags = st["trans_static"][i:]
        if not flags:
            return True
        ratio = sum(1 for x in flags if x) / float(len(flags))
        return bool(ratio >= min_static and _max_false_streak(flags) <= max_consecutive_motion_outliers)

    def find(st):
        latest = len(st["final_match"]) - hold
        first = None
        for c in range(0, latest + 1):
            ok, _ = primary(st, c)
            if not ok:
                continue
            first = c if first is None else first
            if residual(st, c):
                return c, first
        return None, first

    start = n - window
    st = stats(start)
    lock, first = find(st)
    if start > 0 and primary(st, 0)[0]:          # the window began inside the lock: look further back
        start = 0
        st = stats(start)
        lock, first = find(st)
    if lock is None:
        return {"freeze_detected": False, "freeze_start_frame": -1, "trailing_frames": 0, "confidence": 0.0,
                "reason": "residual motion" if first is not None else "no stable final state"}
    trailing = len(st["final_match"]) - lock
    fm = st["final_mean"][lock:]
    fa = st["final_active"][lock:]
    tm = st["trans_mean"][lock:]
    ta = st["trans_active"][lock:]
    avg = lambda xs: sum(xs) / max(1, len(xs))  # noqa: E731
    conf = (0.35 * (1 - min(1.0, avg(fm) / max(final_mean_diff_threshold, 1e-9)))
            + 0.20 * ((1 - min(1.0, avg(fa) / max(f_area, 1e-9))) if f_area > 0 else 1.0)
            + 0.10 * (1 - min(1.0, avg(tm) / max(transition_mean_diff_threshold, 1e-9)))
            + 0.10 * ((1 - min(1.0, avg(ta) / max(t_area, 1e-9))) if t_area > 0 else 1.0)
            + 0.10 * min(1.0, trailing / float(max(hold * 2, 1)))
            + 0.15 * primary(st, lock)[1])
    return {"freeze_detected": True, "freeze_start_frame": int(start + lock), "trailing_frames": int(trailing),
            "confidence": round(max(0.0, min(1.0, conf)), 3), "reason": "stable final state, near-static"}


def legal_cut(freeze_start: int, overlap: int, safety: int = 3, min_new: int = 17) -> int | None:
    """Longest H3 length (5 + 17k) that ends `safety` frames before the freeze and still adds `min_new` new frames
    after the pinned overlap; None when no such length exists."""
    limit = int(freeze_start) - int(safety)
    if limit < 5:
        return None
    cut = 5 + ((limit - 5) // 17) * 17
    return cut if cut - int(overlap) >= int(min_new) else None
