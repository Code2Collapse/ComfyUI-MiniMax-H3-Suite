"""Performance-preservation scores for H3 character replacement (C2C L8.25, owner A12 s1.3 / s1.6).

Pure numpy (+ cv2 for head pose, torch only for the pack's own Oklab conversion). Inputs are per-frame landmark
arrays (FaceMesh 478 x 3, normalised x/y) and uint8 / float frames; outputs are plain dicts that serialise to JSON.
Design and definitions: C2C docs/research/l825_h3_performance_bench.md.

Rules: frames without a face in either clip are counted and excluded, never imputed; every score says what it
measured on; nothing here claims identity for a face it did not see.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np

# FaceMesh indices (refine_landmarks=True adds the irises 468-477)
MOUTH_V, MOUTH_H = (13, 14), (78, 308)
EYE_L_V, EYE_L_H = (159, 145), (33, 133)
EYE_R_V, EYE_R_H = (386, 374), (362, 263)
IRIS_L, IRIS_R = 468, 473
POSE_IDX = (1, 152, 33, 263, 61, 291)
# a generic face model in mm for those six points (nose tip, chin, eye outer corners, mouth corners), in OpenCV's
# camera convention (x right, y DOWN, z AWAY from the camera) so a frontal face solves to angles near 0
POSE_3D = np.array([[0.0, 0.0, 0.0], [0.0, 63.6, 12.5], [-43.3, -32.7, 26.0], [43.3, -32.7, 26.0],
                    [-28.9, 28.9, 24.1], [28.9, 28.9, 24.1]], dtype=np.float64)

SIGNALS = ("mouth_open", "eye_open_l", "eye_open_r", "gaze_x", "gaze_y", "yaw", "pitch", "roll")


def _dist(p: np.ndarray, i: int, j: int) -> float:
    return float(np.linalg.norm(p[i, :2] - p[j, :2]))


def _ratio(p: np.ndarray, v: tuple[int, int], h: tuple[int, int]) -> float:
    w = _dist(p, *h)
    return _dist(p, *v) / w if w > 1e-9 else float("nan")


def head_pose(p: np.ndarray, width: int, height: int) -> tuple[float, float, float]:
    """(yaw, pitch, roll) in degrees from 6 landmarks by solvePnP; NaNs when it fails."""
    import cv2

    img = np.array([[p[i, 0] * width, p[i, 1] * height] for i in POSE_IDX], dtype=np.float64)
    f = float(width)
    cam = np.array([[f, 0, width / 2.0], [0, f, height / 2.0], [0, 0, 1]], dtype=np.float64)
    ok, rvec, _t = cv2.solvePnP(POSE_3D, img, cam, np.zeros(4), flags=cv2.SOLVEPNP_EPNP)
    if not ok:
        return (float("nan"),) * 3
    rot, _ = cv2.Rodrigues(rvec)
    sy = math.hypot(rot[0, 0], rot[1, 0])
    pitch = math.degrees(math.atan2(rot[2, 1], rot[2, 2]))
    yaw = math.degrees(math.atan2(-rot[2, 0], sy))
    roll = math.degrees(math.atan2(rot[1, 0], rot[0, 0]))      # + = the face turned clockwise in the image
    return yaw, pitch, roll


def _gaze(p: np.ndarray, iris: int, outer: int, inner: int) -> tuple[float, float]:
    a, b = p[outer, :2], p[inner, :2]
    centre = (a + b) / 2.0
    half = np.linalg.norm(b - a) / 2.0
    if half < 1e-9 or p.shape[0] <= iris:
        return float("nan"), float("nan")
    d = (p[iris, :2] - centre) / half
    return float(d[0]), float(d[1])


def face_signals(landmarks: Sequence[np.ndarray | None], width: int, height: int) -> dict[str, np.ndarray]:
    """Per-frame signals; NaN where no face was found."""
    out = {k: np.full(len(landmarks), np.nan) for k in SIGNALS}
    for t, p in enumerate(landmarks):
        if p is None:
            continue
        p = np.asarray(p, dtype=np.float64)
        out["mouth_open"][t] = _ratio(p, MOUTH_V, MOUTH_H)
        out["eye_open_l"][t] = _ratio(p, EYE_L_V, EYE_L_H)
        out["eye_open_r"][t] = _ratio(p, EYE_R_V, EYE_R_H)
        gl, gr = _gaze(p, IRIS_L, 33, 133), _gaze(p, IRIS_R, 263, 362)
        out["gaze_x"][t] = np.nanmean([gl[0], gr[0]])
        out["gaze_y"][t] = np.nanmean([gl[1], gr[1]])
        out["yaw"][t], out["pitch"][t], out["roll"][t] = head_pose(p, width, height)
    return out


def _pearson(a: np.ndarray, b: np.ndarray) -> float | None:
    if a.size < 3 or np.std(a) < 1e-9 or np.std(b) < 1e-9:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def mouth_events(series: np.ndarray, threshold: float) -> list[int]:
    """Frames where the mouth crosses open (+) or closed (-) with hysteresis (0.8 x threshold to close)."""
    events, state = [], False
    for t, v in enumerate(series):
        if np.isnan(v):
            continue
        if not state and v > threshold:
            state = True
            events.append(t)
        elif state and v < 0.8 * threshold:
            state = False
            events.append(-t - 1)
    return events


def event_f1(ref: list[int], got: list[int], tol: int = 1) -> float | None:
    if not ref and not got:
        return None
    used, hits = set(), 0
    for e in ref:
        for j, g in enumerate(got):
            if j not in used and (e >= 0) == (g >= 0) and abs(abs(e) - abs(g)) <= tol:
                used.add(j)
                hits += 1
                break
    p = hits / len(got) if got else 0.0
    r = hits / len(ref) if ref else 0.0
    return 2 * p * r / (p + r) if p + r > 0 else 0.0


def performance_scores(src: dict[str, np.ndarray], out: dict[str, np.ndarray]) -> dict[str, Any]:
    """How closely the output's face performance follows the source's, per signal."""
    n = min(len(src["mouth_open"]), len(out["mouth_open"]))
    both = ~np.isnan(src["mouth_open"][:n]) & ~np.isnan(out["mouth_open"][:n])
    res: dict[str, Any] = {"frames": n, "frames_with_face_in_both": int(both.sum()),
                           "frames_missing_face": int(n - both.sum()), "signals": {}}
    for k in SIGNALS:
        a, b = src[k][:n], out[k][:n]
        m = both & ~np.isnan(a) & ~np.isnan(b)
        if m.sum() == 0:
            res["signals"][k] = {"measured_on": 0}
            continue
        res["signals"][k] = {"measured_on": int(m.sum()), "pearson_r": _pearson(a[m], b[m]),
                             "mae": float(np.mean(np.abs(a[m] - b[m])))}
    mo_src, mo_out = src["mouth_open"][:n].copy(), out["mouth_open"][:n].copy()
    mo_src[~both], mo_out[~both] = np.nan, np.nan
    if both.sum() >= 3:
        thr = float(np.nanmedian(mo_src) * 1.5) or 0.05
        ev_s, ev_o = mouth_events(mo_src, thr), mouth_events(mo_out, thr)
        s_std, o_std = float(np.nanstd(mo_src)), float(np.nanstd(mo_out))
        res["mouth_events"] = {"threshold": thr, "source": len(ev_s), "output": len(ev_o),
                               "f1_tol1": event_f1(ev_s, ev_o, 1)}
        res["frozen_expression_ratio"] = (o_std / s_std) if s_std > 1e-9 else None
        res["frozen"] = bool(s_std > 1e-9 and o_std / s_std < 0.5)
    return res


# ── pixel scores ─────────────────────────────────────────────────────────────

def _f32(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    return x.astype(np.float32) / 255.0 if x.dtype == np.uint8 else x.astype(np.float32)


def _dilate(mask: np.ndarray, px: int) -> np.ndarray:
    import cv2

    if px <= 0:
        return mask > 0.5
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * px + 1, 2 * px + 1))
    return np.stack([cv2.dilate((m > 0.5).astype(np.uint8), k) > 0 for m in mask])


def pixel_scores(plate: np.ndarray, output: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    """plate / output [T,H,W,3], mask [T,H,W] (1 = replaced). Background, edge leakage, excess flicker, colour."""
    plate, output, mask = _f32(plate), _f32(output), _f32(mask)
    t = min(len(plate), len(output), len(mask))
    plate, output, mask = plate[:t], output[:t], mask[:t]
    diff = np.abs(output - plate).max(axis=-1)                         # [T,H,W]
    d8, d2, d12, d24 = _dilate(mask, 8), _dilate(mask, 2), _dilate(mask, 12), _dilate(mask, 24)
    outside = ~d8
    res: dict[str, Any] = {"frames": t}
    if outside.any():
        mse = float(np.mean((output - plate)[outside] ** 2))
        res["background"] = {"psnr_db": (float("inf") if mse == 0 else 10 * math.log10(1.0 / mse)),
                             "max_abs": float(diff[outside].max()), "measured_px": int(outside.sum())}
    band, far = d12 & ~d2, d24 & ~d12
    if band.any() and far.any():
        res["edge_leakage"] = {"band_2_12px": float(diff[band].mean()), "band_12_24px": float(diff[far].mean())}
    inside = mask > 0.5
    if t >= 2 and inside[1:].any():
        m = inside[1:] & inside[:-1]
        f_out = np.abs(output[1:] - output[:-1]).mean(axis=-1)[m].mean()
        f_src = np.abs(plate[1:] - plate[:-1]).mean(axis=-1)[m].mean()
        res["flicker"] = {"output": float(f_out), "plate": float(f_src), "excess": float(f_out - f_src)}
    res["colour"] = colour_drift(plate, output, inside, outside)
    return res


def _oklab(x: np.ndarray) -> np.ndarray:
    import torch

    from .color_space import linear_srgb_to_oklab, srgb_to_linear

    tt = torch.from_numpy(np.ascontiguousarray(x)).float()
    return linear_srgb_to_oklab(srgb_to_linear(tt)).numpy()


def colour_drift(plate: np.ndarray, output: np.ndarray, inside: np.ndarray, outside: np.ndarray) -> dict[str, Any]:
    lab_p, lab_o = _oklab(plate), _oklab(output)
    res: dict[str, Any] = {}
    traj_p = np.array([lab_p[i][inside[i]].mean(axis=0) if inside[i].any() else [np.nan] * 3 for i in range(len(plate))])
    traj_o = np.array([lab_o[i][inside[i]].mean(axis=0) if inside[i].any() else [np.nan] * 3 for i in range(len(plate))])
    if np.isfinite(traj_o).any():
        res["inside_trajectory_std"] = {"output": float(np.nanstd(traj_o, axis=0).mean()),
                                        "plate": float(np.nanstd(traj_p, axis=0).mean())}
        res["inside_trajectory_excess"] = res["inside_trajectory_std"]["output"] - res["inside_trajectory_std"]["plate"]
    if outside.any():
        res["outside_mean_shift"] = float(np.linalg.norm(lab_o[outside].mean(axis=0) - lab_p[outside].mean(axis=0)))
    return res
