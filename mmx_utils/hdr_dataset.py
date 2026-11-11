"""HDR training data for H3 (ledger L9.180; plan docs/research/l9178_minimax_hdr_rnd.md sections 5 and 7).

Building blocks of tools/h3_hdr_dataset.py, all numpy float32, scene-linear in ACEScg unless a name says otherwise:

- ``orbit_frames``: a camera orbit through an equirectangular HDRI (Poly Haven, CC0) - LumiVid's training source;
- ``normalise_exposure``: per-clip exposure (Dark Matters' p99-to-1, an 18 % grey median, or none for camera plates);
- ``to_training_codes``: the log code (ACEScct by default) as dithered 8-bit, because ai-toolkit reads video as 8-bit;
- ``sdr_variant``: a degraded display-referred SDR rendering for Track C (tone curve, contrast, highlight blur, shadow
  softening, exposure shift; H.264 is applied by the writer), after LumiVid appendix E and Dark Matters section 4;
- ``split_by_parent``: a parent-disjoint train / val split (no crop of one source in both).
"""
from __future__ import annotations

import hashlib
import math

import numpy as np

from . import hdr_log as H

H3_FRAME_GRID = (22, 39, 56, 73, 90)            # 17n+5
SIDE_MULTIPLE = 32                              # 16x VAE x 2x2 patch (ai-toolkit get_bucket_divisibility)
EXPOSURE_MODES = ("p99_to_1", "grey_0.18", "none")
TONE_CURVES = ("aces2_sdr", "reinhard_w4", "hable")      # the last one is held out for testing by default


def snap_frames(n: int) -> int:
    """The largest 17n+5 count <= n (>= 22)."""
    if n < 22:
        raise ValueError(f"an H3 clip needs at least 22 frames (17n+5), got {n}")
    return (n - 5) // 17 * 17 + 5


def snap_side(n: int) -> int:
    return max(SIDE_MULTIPLE, n // SIDE_MULTIPLE * SIDE_MULTIPLE)


# ---- HDRI orbits ---------------------------------------------------------------------------------------------------

def orbit_frames(equirect: np.ndarray, *, frames: int, width: int, height: int, fov_deg: float = 60.0,
                 yaw0_deg: float = 0.0, yaw_span_deg: float = 30.0, pitch_deg: float = 0.0,
                 pitch_wobble_deg: float = 2.0) -> np.ndarray:
    """[frames, height, width, 3] perspective views of an equirect panorama [H, W, 3] (longitude across, latitude
    down, +y up), the camera turning yaw0 -> yaw0 + span with a small pitch wobble. Bilinear, wraps in longitude."""
    eq = np.asarray(equirect, np.float32)
    eh, ew = eq.shape[:2]
    f = 0.5 * width / math.tan(math.radians(fov_deg) / 2.0)
    xs = (np.arange(width, dtype=np.float32) - (width - 1) / 2.0) / f
    ys = ((height - 1) / 2.0 - np.arange(height, dtype=np.float32)) / f
    gx, gy = np.meshgrid(xs, ys)
    d = np.stack([gx, gy, np.ones_like(gx)], -1)
    d /= np.linalg.norm(d, axis=-1, keepdims=True)
    out = np.empty((frames, height, width, 3), np.float32)
    for i in range(frames):
        t = i / max(1, frames - 1)
        yaw = math.radians(yaw0_deg + yaw_span_deg * t)
        pitch = math.radians(pitch_deg + pitch_wobble_deg * math.sin(2 * math.pi * t))
        cp, sp, cy, sy = math.cos(pitch), math.sin(pitch), math.cos(yaw), math.sin(yaw)
        rx = np.array([[1, 0, 0], [0, cp, sp], [0, -sp, cp]], np.float32)          # +pitch looks up
        ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], np.float32)
        v = d @ (ry @ rx).T
        lon = np.arctan2(v[..., 0], v[..., 2])                       # -pi .. pi
        lat = np.arcsin(np.clip(v[..., 1], -1.0, 1.0))               # -pi/2 .. pi/2
        u = (lon / (2 * np.pi) + 0.5) * ew - 0.5
        w = (0.5 - lat / np.pi) * eh - 0.5
        out[i] = _bilinear_wrap(eq, u, w)
    return out


def _bilinear_wrap(img: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    u0 = np.floor(u).astype(np.int64)
    v0 = np.floor(v).astype(np.int64)
    fu, fv = (u - u0)[..., None], (v - v0)[..., None]
    u1 = (u0 + 1) % w
    u0 = u0 % w
    v1 = np.clip(v0 + 1, 0, h - 1)
    v0 = np.clip(v0, 0, h - 1)
    top = img[v0, u0] * (1 - fu) + img[v0, u1] * fu
    bot = img[v1, u0] * (1 - fu) + img[v1, u1] * fu
    return (top * (1 - fv) + bot * fv).astype(np.float32)


def rec709_to_acescg(lin: np.ndarray) -> np.ndarray:
    from mmx_vendor.hdr_curves._primaries import gamut_matrix
    return H._apply(gamut_matrix("sRGB / Rec.709", H.WORKING_GAMUT), lin)


# ---- exposure ------------------------------------------------------------------------------------------------------

def normalise_exposure(clip: np.ndarray, mode: str = "p99_to_1") -> tuple[np.ndarray, float]:
    """(clip * scale, scale). p99_to_1: the 75th percentile of the per-frame 99th-percentile luminance becomes 1.0
    (Dark Matters); grey_0.18: the clip's median luminance becomes 0.18; none: as shot (camera plates)."""
    if mode not in EXPOSURE_MODES:
        raise ValueError(f"exposure mode must be one of {EXPOSURE_MODES}, got {mode!r}")
    if mode == "none":
        return clip, 1.0
    y = H.luminance(clip)
    if mode == "p99_to_1":
        ref = float(np.percentile(np.percentile(y.reshape(y.shape[0], -1), 99, axis=1), 75))
        target = 1.0
    else:
        ref, target = float(np.median(y)), 0.18
    scale = target / max(ref, 1e-6)
    return (clip * np.float32(scale)).astype(np.float32), float(scale)


# ---- the training codes ----------------------------------------------------------------------------------------------

def to_training_codes(clip: np.ndarray, code: str = "ACEScct", *, seed: int = 0) -> tuple[np.ndarray, dict]:
    """(uint8 [T, H, W, 3], stats): ACEScg -> log code, clipped to [0, 1] (what the VAE can hold), TPDF-dithered to 8 bits
    so the model does not learn banding. Stats report what was clipped."""
    c = H.encode(clip, code)
    stats = {"clipped_above_1": float((c > 1.0).mean()), "clipped_below_0": float((c < 0.0).mean()),
             "stops_above_white": H.stops_above_white(clip)}
    rng = np.random.default_rng(seed)
    tpdf = (rng.random(c.shape, dtype=np.float32) - rng.random(c.shape, dtype=np.float32))
    q = np.clip(np.round(np.clip(c, 0.0, 1.0) * 255.0 + tpdf), 0, 255).astype(np.uint8)
    return q, stats


# ---- SDR variants (Track C) ------------------------------------------------------------------------------------------

def _srgb_oetf(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1 / 2.4) - 0.055).astype(np.float32)


def _hable(x: np.ndarray) -> np.ndarray:
    a, b, c, d, e, f = 0.15, 0.50, 0.10, 0.20, 0.02, 0.30
    return ((x * (a * x + c * b) + d * e) / (x * (a * x + b) + d * f)) - e / f


def tone_map(acescg: np.ndarray, curve: str) -> np.ndarray:
    """Display-referred sRGB / Rec.709 code values in [0, 1]."""
    if curve == "aces2_sdr":
        import PyOpenColorIO as ocio  # type: ignore[import-not-found]
        cfg = ocio.Config.CreateFromFile("ocio://studio-config-latest")
        view = cfg.getDefaultView("sRGB - Display")
        proc = cfg.getProcessor(ocio.DisplayViewTransform(src="ACEScg", display="sRGB - Display", view=view))
        a = np.ascontiguousarray(acescg.reshape(-1, 3), np.float32).copy()
        proc.getDefaultCPUProcessor().applyRGB(a)
        return np.clip(a.reshape(acescg.shape), 0.0, 1.0)
    from mmx_vendor.hdr_curves._primaries import gamut_matrix
    lin709 = np.maximum(H._apply(gamut_matrix(H.WORKING_GAMUT, "sRGB / Rec.709"), acescg), 0.0)
    if curve == "reinhard_w4":                                      # extended Reinhard, white 4 (Dark Matters)
        tm = lin709 * (1.0 + lin709 / 16.0) / (1.0 + lin709)
    elif curve == "hable":                                          # Hable filmic, white 11.2
        tm = _hable(lin709 * 2.0) / _hable(np.float32(11.2))
    else:
        raise ValueError(f"tone curve must be one of {TONE_CURVES}, got {curve!r}")
    return _srgb_oetf(tm)


def sdr_variant(acescg: np.ndarray, *, curve: str, ev: float = 0.0, contrast: float = 1.0,
                highlight_blur_sigma: float = 0.0, shadow_soften: bool = False) -> np.ndarray:
    """One degraded SDR rendering [T, H, W, 3] in [0, 1] (8-bit quantisation and H.264 happen in the writer)."""
    sdr = tone_map(acescg * np.float32(2.0 ** ev), curve)
    if contrast != 1.0:                                             # anchored at mid grey (LumiVid E.2)
        sdr = np.clip(0.5 + (sdr - 0.5) * contrast, 0.0, 1.0)
    if highlight_blur_sigma > 0:
        sdr = _blur_where(sdr, sdr.max(-1) > 0.95, highlight_blur_sigma)
    if shadow_soften:                                               # 2x area down, upsample (LumiVid E.1)
        sdr = _blur_where(sdr, sdr.max(-1) < 0.05, 1.0)
    return sdr.astype(np.float32)


def _blur_where(img: np.ndarray, mask: np.ndarray, sigma: float) -> np.ndarray:
    import cv2
    k = 2 * int(math.ceil(2 * sigma)) + 1
    out = img.copy()
    for t in range(img.shape[0]):
        m = cv2.GaussianBlur(mask[t].astype(np.float32), (k, k), sigma)[..., None]
        b = cv2.GaussianBlur(img[t], (k, k), sigma)
        out[t] = img[t] * (1 - m) + b * m
    return out


def random_sdr_recipe(rng: np.random.Generator, curves=("aces2_sdr", "reinhard_w4")) -> dict:
    """LumiVid appendix E / Dark Matters section 4 ranges. crf: 70 % 12, 30 % 23-35."""
    return {"curve": str(rng.choice(list(curves))), "ev": float(rng.choice([-2.0, 0.0, 0.0, 2.0])),
            "contrast": float(rng.uniform(1.0, 1.5)) if rng.random() < 0.3 else 1.0,
            "highlight_blur_sigma": float(rng.uniform(5.0, 12.0)) if rng.random() < 0.7 else 0.0,
            "shadow_soften": bool(rng.random() < 0.7),
            "crf": 12 if rng.random() < 0.7 else int(rng.integers(23, 36))}


# ---- split -------------------------------------------------------------------------------------------------------------

def split_by_parent(parents: list[str], val_fraction: float = 0.08) -> dict[str, str]:
    """parent -> 'train' | 'val', deterministic by name hash, so crops of one source never straddle the split."""
    out = {}
    for p in sorted(set(parents)):
        h = int(hashlib.sha1(p.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
        out[p] = "val" if h < val_fraction else "train"
    if len(out) > 1 and "val" not in out.values():                  # a tiny set still gets one val parent
        out[sorted(out)[0]] = "val"
    return out
