"""HDR log code spaces for H3 (ledger L9.178 / L9.179): scene-linear ACEScg <-> the log code a video VAE sees, and the
measurements that decide which code H3 should generate in (docs/research/l9178_minimax_hdr_rnd.md section 4).

The curves and gamuts are NukeMax's (mmx_vendor/hdr_curves, kept byte-identical by tools/sync_vendored.py), so a
LogC3 here is the same LogC3 NukeMax decodes. Every function is numpy float32 in, float32 out; nothing clips the
scene-linear signal except where a code must fit the VAE's [0, 1] input.

PU21 (Mantiuk & Azimi 2021) constants are the official 'banding_glare' variant of gfxdisp/pu21
(matlab/pu21_encoder.m, BSD-3-Clause, (c) 2021 Graphics and Displays group, University of Cambridge); PU21-PSNR uses
the repo's peak of 256 (100 nit -> 256) on absolute luminance.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_PKG = Path(__file__).resolve().parents[1]
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from mmx_vendor.hdr_curves._curves import resolve_curve  # noqa: E402
from mmx_vendor.hdr_curves._primaries import gamut_matrix  # noqa: E402

WORKING_GAMUT = "ACEScg (AP1)"
# the code spaces the probe compares by default (doc section 4); any NukeMax curve name works
DEFAULT_CODES = ("ACEScct", "ARRI LogC3", "ARRI LogC4")

# Rec.709 / sRGB luminance weights applied to ACEScg via the AP1 -> Rec.709 matrix (luminance is gamut-dependent)
_REC709_Y = np.array([0.2126, 0.7152, 0.0722], np.float64)


def _apply(m: np.ndarray, rgb: np.ndarray) -> np.ndarray:
    return np.einsum("ij,...j->...i", m.astype(np.float64), rgb.astype(np.float64)).astype(np.float32)


def code_gamut(code: str) -> str:
    return resolve_curve(code)[2]


def encode(lin_acescg: np.ndarray, code: str) -> np.ndarray:
    """ACEScg scene-linear [..., 3] -> the code space's log values (camera gamut first, then the curve). Unclipped."""
    to_log, _from_log, gamut = resolve_curve(code)
    x = lin_acescg if gamut in ("", WORKING_GAMUT) else _apply(gamut_matrix(WORKING_GAMUT, gamut), lin_acescg)
    return to_log(x)


def decode(codes: np.ndarray, code: str) -> np.ndarray:
    """The inverse of encode: log values -> ACEScg scene-linear."""
    _to_log, from_log, gamut = resolve_curve(code)
    x = from_log(codes)
    return x if gamut in ("", WORKING_GAMUT) else _apply(gamut_matrix(gamut, WORKING_GAMUT), x)


def code_of_black(code: str) -> float:
    """The code linear 0 maps to (the footroom floor). <= 0 means the curve has no footroom (cannot be checked)."""
    return float(resolve_curve(code)[0](np.zeros(1, np.float32))[0])


def ceiling(code: str) -> float:
    """Scene-linear value at code 1.0 (the brightest value a [0, 1] VAE input can carry), in the curve's gamut."""
    return float(resolve_curve(code)[1](np.ones(1, np.float32))[0])


def footroom_fraction(codes: np.ndarray, code: str, margin: float = 0.002) -> float | None:
    """Share of channel values below the curve's black code (negative light, impossible in a genuine log image).
    None for a curve without footroom (DaVinci Intermediate, ACEScc): no code value could prove anything there."""
    black = code_of_black(code)
    if black <= 0.0:
        return None
    return float((np.asarray(codes)[..., :3] < black - margin).mean())


def luminance(lin_acescg: np.ndarray) -> np.ndarray:
    """Relative luminance Y of ACEScg linear RGB (through Rec.709: the weights belong to that gamut)."""
    rec709 = _apply(gamut_matrix(WORKING_GAMUT, "sRGB / Rec.709"), lin_acescg)
    return (rec709.astype(np.float64) @ _REC709_Y).astype(np.float32)


def stops_above_white(lin_acescg: np.ndarray, pct: float = 99.9) -> float:
    """log2 of the pct-th percentile luminance (LumiVid's dynamic-range figure; diffuse white = 1.0 = 0 stops)."""
    y = luminance(lin_acescg)
    return float(np.log2(max(float(np.percentile(y, pct)), 1e-6)))


# ---- PU21 (gfxdisp/pu21, banding_glare, BSD-3-Clause) ---------------------------------------------------------------
_PU21 = (0.353487901, 0.3734658629, 8.277049286e-05, 0.9062562627, 0.09150303166, 0.9099517204, 596.3148142)
_PU21_LMIN, _PU21_LMAX, _PU21_PEAK = 0.005, 10000.0, 256.0


def pu21_encode(nits: np.ndarray) -> np.ndarray:
    p = _PU21
    y = np.clip(np.asarray(nits, np.float64), _PU21_LMIN, _PU21_LMAX)
    v = p[6] * (((p[0] + p[1] * y ** p[3]) / (1.0 + p[2] * y ** p[3])) ** p[4] - p[5])
    return np.maximum(v, 0.0)


def pu21_psnr(test_lin: np.ndarray, ref_lin: np.ndarray, white_nits: float = 100.0) -> float:
    """Colour PU21-PSNR (the repo's 'PSNR', peak 256) on absolute RGB: scene-linear 1.0 = white_nits cd/m2."""
    a = pu21_encode(np.asarray(test_lin, np.float64) * white_nits)
    b = pu21_encode(np.asarray(ref_lin, np.float64) * white_nits)
    mse = float(np.mean((a - b) ** 2))
    return float("inf") if mse == 0.0 else float(10.0 * np.log10(_PU21_PEAK ** 2 / mse))


# ---- colour survival in deep shadows ----------------------------------------------------------------------------------

def uv_prime(lin_acescg: np.ndarray) -> np.ndarray:
    """CIE 1976 u'v' of ACEScg linear RGB [..., 3] -> [..., 2] (RGB floored at 1e-8 like Dark Matters)."""
    xyz = _apply(_ap1_to_xyz(), np.maximum(lin_acescg, 1e-8)).astype(np.float64)
    d = xyz[..., 0] + 15.0 * xyz[..., 1] + 3.0 * xyz[..., 2]
    d = np.where(d <= 1e-12, 1e-12, d)
    return np.stack([4.0 * xyz[..., 0] / d, 9.0 * xyz[..., 1] / d], axis=-1).astype(np.float32)


def _ap1_to_xyz() -> np.ndarray:
    from mmx_vendor.hdr_curves import _primaries as P
    r, g, b, w, _ = P.PRIMARIES[WORKING_GAMUT]
    return P._npm(r, g, b, w)


def per_stop_error(test_lin: np.ndarray, ref_lin: np.ndarray, lo: int = -8, hi: int = 8) -> dict[str, float]:
    """Mean |log2(test Y / ref Y)| (stops of error) per reference brightness band, in stops relative to 18% grey."""
    yt, yr = luminance(test_lin), luminance(ref_lin)
    ok = yr > 1e-6
    band = np.floor(np.log2(np.maximum(yr, 1e-12) / 0.18))
    err = np.abs(np.log2(np.maximum(yt, 1e-6) / np.maximum(yr, 1e-6)))
    out: dict[str, float] = {}
    for s in range(lo, hi + 1):
        m = ok & (band == s)
        if m.sum() >= 16:
            out[f"{s:+d}"] = float(err[m].mean())
    return out


def value_range(x: np.ndarray) -> dict[str, float]:
    x = np.asarray(x, np.float32)
    return {"min": float(x.min()), "p0.1": float(np.percentile(x, 0.1)), "p99.9": float(np.percentile(x, 99.9)),
            "max": float(x.max())}


def region_error(test_lin: np.ndarray, ref_lin: np.ndarray, lo: float, hi: float) -> float | None:
    """Mean |log2(test Y / ref Y)| (stops) over reference pixels with lo <= Y < hi; None when fewer than 16 pixels."""
    yt, yr = luminance(test_lin), luminance(ref_lin)
    m = (yr >= lo) & (yr < hi)
    if m.sum() < 16:
        return None
    return float(np.abs(np.log2(np.maximum(yt[m], 1e-6) / np.maximum(yr[m], 1e-6))).mean())


def delta_uv_in(test_lin: np.ndarray, ref_lin: np.ndarray, lo: float, hi: float) -> float | None:
    """Mean Δu'v' over reference pixels with lo <= Y < hi (colour drift in one brightness band)."""
    yr = luminance(ref_lin)
    m = (yr >= lo) & (yr < hi)
    if m.sum() < 16:
        return None
    return float(np.linalg.norm(uv_prime(test_lin) - uv_prime(ref_lin), axis=-1)[m].mean())


def temporal_consistency(test_lin: np.ndarray, ref_lin: np.ndarray) -> dict | None:
    """Frame-to-frame behaviour of a round trip on VIDEO [T, H, W, 3] (T >= 2), in stops:
    ``mean_luma_flicker`` = mean |Δ_t log2 mean-Y(out) - Δ_t log2 mean-Y(ref)| (exposure pumping / flicker added);
    ``pixel_temporal_error`` = mean over pixels (ref Y >= 1e-3) of |Δ_t log2 Y(out) - Δ_t log2 Y(ref)| (shimmer)."""
    if test_lin.ndim != 4 or test_lin.shape[0] < 2:
        return None
    yt, yr = luminance(test_lin).astype(np.float64), luminance(ref_lin).astype(np.float64)
    mt = np.log2(np.maximum(yt.reshape(yt.shape[0], -1).mean(1), 1e-9))
    mr = np.log2(np.maximum(yr.reshape(yr.shape[0], -1).mean(1), 1e-9))
    lt, lr = np.log2(np.maximum(yt, 1e-6)), np.log2(np.maximum(yr, 1e-6))
    m = (yr[1:] >= 1e-3) & (yr[:-1] >= 1e-3)
    d = np.abs(np.diff(lt, axis=0) - np.diff(lr, axis=0))
    return {"mean_luma_flicker": float(np.abs(np.diff(mt) - np.diff(mr)).mean()),
            "max_luma_flicker": float(np.abs(np.diff(mt) - np.diff(mr)).max()),
            "pixel_temporal_error": float(d[m].mean()) if m.any() else None}


# provisional acceptance values (A26 3.2): the final ones are chosen from real H3 probe runs and recorded with them
PROBE_THRESHOLDS = {"pu21_psnr_min": 35.0, "highlight_error_max_stops": 0.25, "shadow_delta_uv_max": 0.02,
                    "footroom_violations_max": 0.001, "luma_flicker_max_stops": 0.05, "clipped_pixels_max": 0.001}


def verdict(rep: dict, th: dict | None = None) -> dict:
    """PASS / FAIL / NOT VERIFIED for one code space of a ``roundtrip_report``. NOT VERIFIED when the input has no
    highlights above diffuse white (nothing would prove the code carries HDR) or a metric could not be measured."""
    th = dict(PROBE_THRESHOLDS, **(th or {}))
    e0 = rep["exposures"].get("+0")
    if not e0:
        return {"verdict": "NOT VERIFIED", "why": ["no +0 exposure measured"], "thresholds": th}
    low = rep["exposures"][min(rep["exposures"], key=float)]
    why, unknown = [], []

    def check(name, value, ok):
        if value is None:
            unknown.append(f"{name}: not measurable on this input")
        elif not ok(value):
            why.append(f"{name} = {value:.4g}")
    check("pu21_psnr", e0["pu21_psnr"], lambda v: v >= th["pu21_psnr_min"])
    check("highlight_error_stops", e0.get("highlight_error_stops"), lambda v: v <= th["highlight_error_max_stops"])
    check("shadow_delta_uv (lowest exposure)", low.get("delta_uv_shadows"), lambda v: v <= th["shadow_delta_uv_max"])
    if e0.get("footroom_violations_out") is not None:
        check("footroom_violations_out", e0["footroom_violations_out"], lambda v: v <= th["footroom_violations_max"])
    check("clipped_pixels", e0.get("clipped_pixels"), lambda v: v <= th["clipped_pixels_max"])
    t = e0.get("temporal")
    if t is not None:
        check("mean_luma_flicker", t["mean_luma_flicker"], lambda v: v <= th["luma_flicker_max_stops"])
    if why:
        return {"verdict": "FAIL", "why": why + unknown, "thresholds": th}
    if unknown:
        return {"verdict": "NOT VERIFIED", "why": unknown, "thresholds": th}
    return {"verdict": "PASS", "why": [], "thresholds": th}


def roundtrip_report(ref_lin: np.ndarray, roundtrip, code: str, *, exposures=(0.0, -6.0, -8.0),
                     white_nits: float = 100.0) -> dict:
    """Push ACEScg scene-linear frames [T, H, W, 3] through ``roundtrip`` (codes in [0, 1] -> codes, e.g. a VAE encode +
    decode) in the given code space, at each exposure shift, and measure what comes back.

    Returned per exposure: PU21-PSNR, code-space PSNR, mean Δu'v' (pixels with reference Y >= 1e-3 after the shift),
    per-stop luminance error, the share of codes clipped at 0 / 1 going in, and footroom violations coming out."""
    ref_lin = np.asarray(ref_lin, np.float32)
    rep: dict = {"code": code, "gamut": code_gamut(code), "black_code": code_of_black(code),
                 "ceiling_linear": ceiling(code), "white_nits": white_nits, "exposures": {}}
    for ev in exposures:
        ref = ref_lin * np.float32(2.0 ** ev)
        c_raw = encode(ref, code)
        clipped_hi = float((c_raw > 1.0).mean())
        clipped_lo = float((c_raw < 0.0).mean())
        clipped_px = float(((c_raw > 1.0) | (c_raw < 0.0)).any(axis=-1).mean())
        black = code_of_black(code)
        c_in = np.clip(c_raw, 0.0, 1.0)
        c_out = np.asarray(roundtrip(c_in), np.float32)
        if c_out.shape != c_in.shape:
            raise ValueError(f"the round trip changed the shape {c_in.shape} -> {c_out.shape}")
        out = decode(c_out, code)
        ref_c = decode(c_in, code)                      # what a perfect round trip would give back (after the clip)
        y = luminance(ref_c)
        m = y >= 1e-3
        duv = np.linalg.norm(uv_prime(out) - uv_prime(ref_c), axis=-1)
        mse_code = float(np.mean((c_out - c_in) ** 2))
        rep["exposures"][f"{ev:+g}"] = {
            "pu21_psnr": pu21_psnr(out, ref_c, white_nits),
            "code_psnr": float("inf") if mse_code == 0 else float(10 * np.log10(1.0 / mse_code)),
            "delta_uv_mean": float(duv[m].mean()) if m.any() else None,
            "delta_uv_pixels": int(m.sum()),
            "per_stop_error": per_stop_error(out, ref_c),
            "clipped_above_1": clipped_hi,
            "clipped_below_0": clipped_lo,
            "footroom_violations_out": footroom_fraction(c_out, code),
            "stops_above_white_ref": stops_above_white(ref_c),
            "stops_above_white_out": stops_above_white(out),
            # A26 3.2 additions
            "range_linear_in": value_range(ref), "range_linear_out": value_range(out),
            "range_codes_in": value_range(c_raw), "range_codes_out": value_range(c_out),
            "clipped_pixels": clipped_px,
            "below_black_in": float((c_raw < black).mean()) if black > 0 else None,
            "saturated_out_hi": float((c_out >= 0.999).mean()), "saturated_out_lo": float((c_out <= 0.001).mean()),
            "share_below_white_ref": float((y < 1.0).mean()), "share_above_white_ref": float((y >= 1.0).mean()),
            "error_below_white_stops": region_error(out, ref_c, 1e-3, 1.0),
            "highlight_error_stops": region_error(out, ref_c, 1.0, np.inf),
            "delta_uv_shadows": delta_uv_in(out, ref_c, 1e-3, 0.18 / 8),
            "delta_uv_highlights": delta_uv_in(out, ref_c, 1.0, np.inf),
            "temporal": temporal_consistency(out, ref_c),
        }
    return rep
