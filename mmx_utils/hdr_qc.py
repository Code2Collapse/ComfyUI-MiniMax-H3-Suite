"""HDR QC detectors (ledger L9.212, A26 3.7): what makes an HDR file real, measured on scene-linear ACEScg frames
[T, H, W, 3] (+ optional alpha [T, H, W]).

A file being EXR / 16-bit DPX / HLG proves nothing. These checks look for the ways a picture only LOOKS HDR:
- ``quantised_source``: an 8-bit SDR picture stored in a deeper file keeps <= ~256 distinct values per channel;
- ``highlight_plateau``: SDR highlights clipped at one level stay flat after any expansion (a plateau at the maximum);
- ``highlight_texture``: real highlights carry spatial detail above diffuse white; an expanded clip does not;
- ``negative_light``: values below zero (footroom / black violations);
- exposure flicker and pumping, highlight flicker, chroma instability, temporal discontinuities (VIDEO, frame to frame);
- gamut excursions (negative components in AP1 and in Rec.2020 for HLG delivery);
- alpha corruption (non-finite, outside 0..1, colour under fully transparent pixels when premultiplied);
- with an aligned reference: spatial shift and frame offset, PU21-PSNR, per-stop error, colour drift, flicker delta.

``verdict`` is PASS only when highlight information above diffuse white is MEASURED and nothing fails.
"""
from __future__ import annotations

import numpy as np

from . import hdr_log as H

LIMITS = {"stops_above_white_min": 1.0, "quantised_levels_max": 300, "plateau_max": 0.2, "highlight_texture_min": 0.02,
          "negative_light_max": 0.001, "exposure_flicker_max": 0.05, "exposure_pumping_max": 0.15,
          "highlight_flicker_max": 0.15, "chroma_flicker_max": 0.002, "discontinuity_stops": 0.5,
          "gamut_ap1_max": 0.001, "alignment_px_max": 0.5}


def _detrend(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, np.float64)
    if v.size < 3:
        return v - v.mean() if v.size else v
    t = np.arange(v.size, dtype=np.float64)
    a, b = np.polyfit(t, v, 1)
    return v - (a * t + b)


def quantised_source(lin: np.ndarray, sample: int = 200_000) -> dict:
    """Distinct values per channel (max over channels) on a sample of pixels. An 8-bit source, however it was
    expanded, has at most 256 per channel."""
    flat = lin.reshape(-1, lin.shape[-1])
    if flat.shape[0] > sample:
        idx = np.random.default_rng(0).choice(flat.shape[0], sample, replace=False)
        flat = flat[idx]
    levels = int(max(np.unique(flat[:, c]).size for c in range(min(3, flat.shape[1]))))
    return {"distinct_levels_per_channel": levels, "pixels_sampled": int(flat.shape[0])}


def highlights(lin: np.ndarray) -> dict:
    """Above diffuse white (Y >= 1): share of pixels, plateau share (pixels at the clip's maximum channel value) and
    texture (median per-frame spatial std of log2 Y inside the above-white area, in stops)."""
    y = H.luminance(lin)
    above = y >= 1.0
    share = float(above.mean())
    if above.sum() < 16:
        return {"share_above_white": share, "plateau_fraction": None, "texture_stops": None}
    mx = lin[..., :3].max(-1)
    top = float(mx[above].max())
    plateau = float((mx[above] >= top * (1 - 1e-3)).mean())
    tex = []
    for t in range(lin.shape[0]):
        a = above[t]
        if a.sum() >= 16:
            tex.append(float(np.log2(np.maximum(y[t][a], 1e-6)).std()))
    return {"share_above_white": share, "plateau_fraction": plateau,
            "texture_stops": float(np.median(tex)) if tex else None, "peak_linear": top}


def negative_light(lin: np.ndarray) -> float:
    return float((lin[..., :3] < -1e-4).any(-1).mean())


def temporal(lin: np.ndarray) -> dict | None:
    if lin.shape[0] < 3:
        return None
    y = H.luminance(lin).reshape(lin.shape[0], -1).astype(np.float64)
    mean_st = np.log2(np.maximum(y.mean(1), 1e-9))
    hi_st = np.log2(np.maximum(np.percentile(y, 99.9, axis=1), 1e-9))
    dm = _detrend(mean_st)
    lit = y >= 1e-3
    uv = H.uv_prime(lin).reshape(lin.shape[0], -1, 2).astype(np.float64)
    mu = np.stack([uv[t][lit[t]].mean(0) if lit[t].any() else np.zeros(2) for t in range(lin.shape[0])])
    chroma = float(np.sqrt(_detrend(mu[:, 0]).std() ** 2 + _detrend(mu[:, 1]).std() ** 2))
    ly = np.log2(np.maximum(y, 1e-6))
    jumps = np.abs(np.diff(ly, axis=0))
    both = lit[1:] & lit[:-1]
    per = np.array([float(jumps[t][both[t]].mean()) if both[t].any() else 0.0 for t in range(jumps.shape[0])])
    return {"exposure_flicker_stops": float(dm.std()), "exposure_pumping_stops": float(np.abs(dm).max()),
            "highlight_flicker_stops": float(_detrend(hi_st).std()), "chroma_flicker_uv": chroma,
            "frame_change_stops": per.round(4).tolist(),
            "discontinuities": [int(i) + 1 for i in np.nonzero(per > LIMITS["discontinuity_stops"])[0]]}


def gamut(lin: np.ndarray) -> dict:
    from mmx_vendor.hdr_curves._primaries import gamut_matrix
    rec2020 = H._apply(gamut_matrix(H.WORKING_GAMUT, "Rec.2020"), lin[..., :3])
    return {"negative_ap1": float((lin[..., :3] < -1e-4).any(-1).mean()),
            "negative_rec2020": float((rec2020 < -1e-4).any(-1).mean())}


def alpha_check(rgb: np.ndarray, alpha: np.ndarray | None, premultiplied: bool) -> dict:
    if alpha is None:
        return {"present": False}
    a = np.asarray(alpha, np.float32)
    finite = bool(np.isfinite(a).all())
    out_of_range = float(((a < -1e-4) | (a > 1 + 1e-4)).mean()) if finite else None
    rep = {"present": True, "finite": finite, "outside_0_1": out_of_range, "premultiplied": bool(premultiplied)}
    if premultiplied and finite:
        clear = a <= 1e-6
        rep["colour_under_transparent"] = float((np.abs(rgb[..., :3][clear]) > 1e-4).any(-1).mean()) \
            if clear.any() else 0.0
    rep["ok"] = finite and (out_of_range or 0) == 0 and rep.get("colour_under_transparent", 0) == 0
    return rep


def _shift(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Translation of b relative to a (phase correlation, integer + parabolic subpixel)."""
    fa, fb = np.fft.fft2(a - a.mean()), np.fft.fft2(b - b.mean())
    r = fa * np.conj(fb)
    r /= np.maximum(np.abs(r), 1e-12)
    c = np.real(np.fft.ifft2(r))
    i, j = np.unravel_index(int(np.argmax(c)), c.shape)

    def sub(cm, cp, c0):
        d = cm - 2 * c0 + cp
        return 0.0 if abs(d) < 1e-12 else 0.5 * (cm - cp) / d
    h, w = c.shape
    dy = i + sub(c[(i - 1) % h, j], c[(i + 1) % h, j], c[i, j])
    dx = j + sub(c[i, (j - 1) % w], c[i, (j + 1) % w], c[i, j])
    dy = dy - h if dy > h / 2 else dy
    dx = dx - w if dx > w / 2 else dx
    return float(-dy), float(-dx)


def alignment(lin: np.ndarray, ref: np.ndarray, max_lag: int = 3) -> dict:
    """Frame offset (cross-correlation of the per-frame mean log exposure) and spatial shift (phase correlation of
    log luminance, median over frames) of the output against the aligned reference."""
    ly = np.log2(np.maximum(H.luminance(lin), 1e-6)).astype(np.float64)
    lr = np.log2(np.maximum(H.luminance(ref), 1e-6)).astype(np.float64)
    n = min(ly.shape[0], lr.shape[0])
    lag = 0
    if n > 2 * max_lag + 2:
        ma, mr = ly.reshape(ly.shape[0], -1).mean(1), lr.reshape(lr.shape[0], -1).mean(1)
        best = None
        for k in range(-max_lag, max_lag + 1):
            a = ma[max(0, k):n + min(0, k)]
            b = mr[max(0, -k):n - max(0, k)]
            m = min(a.size, b.size)
            if m < 3 or a[:m].std() < 1e-9 or b[:m].std() < 1e-9:
                continue
            cc = float(np.corrcoef(a[:m], b[:m])[0, 1])
            if best is None or cc > best[0] + 1e-9:
                best = (cc, k)
        lag = best[1] if best else 0
    shifts = [_shift(lr[t], ly[t]) for t in range(n)]
    dy = float(np.median([s[0] for s in shifts]))
    dx = float(np.median([s[1] for s in shifts]))
    return {"frame_offset": int(lag), "shift_px": [dx, dy], "shift_magnitude_px": float(np.hypot(dx, dy))}


def compare(lin: np.ndarray, ref: np.ndarray, white_nits: float = 100.0) -> dict:
    n = min(lin.shape[0], ref.shape[0])
    lin, ref = lin[:n], ref[:n]
    if lin.shape[1:3] != ref.shape[1:3]:
        raise ValueError(f"output {lin.shape[1:3]} and reference {ref.shape[1:3]} differ in size")
    y = H.luminance(ref)
    duv = np.linalg.norm(H.uv_prime(lin) - H.uv_prime(ref), axis=-1)
    lit, shadow, hi = y >= 1e-3, (y >= 1e-3) & (y < 0.0036), y >= 1.0
    to, tr = temporal(lin), temporal(ref)
    rep = {"frames_compared": int(n), "pu21_psnr": H.pu21_psnr(lin, ref, white_nits),
           "per_stop_error": H.per_stop_error(lin, ref),
           "highlight_error_stops": H.region_error(lin, ref, 1.0, np.inf) if hasattr(H, "region_error") else None,
           "delta_uv_mean": float(duv[lit].mean()) if lit.any() else None,
           "delta_uv_shadows": float(duv[shadow].mean()) if shadow.sum() >= 16 else None,
           "delta_uv_highlights": float(duv[hi].mean()) if hi.sum() >= 16 else None,
           "stops_above_white_reference": H.stops_above_white(ref), "alignment": alignment(lin, ref)}
    if to and tr:
        rep["flicker_added_stops"] = to["exposure_flicker_stops"] - tr["exposure_flicker_stops"]
    return rep


def run(lin: np.ndarray, *, alpha: np.ndarray | None = None, premultiplied: bool = False,
        ref: np.ndarray | None = None, log_codes: np.ndarray | None = None, code: str | None = None,
        white_nits: float = 100.0) -> dict:
    lin = np.asarray(lin, np.float32)
    per_frame = [H.stops_above_white(f[None]) for f in lin]
    rep = {"frames": int(lin.shape[0]), "size": [int(lin.shape[2]), int(lin.shape[1])],
           "stops_above_white": float(np.median(per_frame)), "highlights": highlights(lin),
           "quantised": quantised_source(lin), "negative_light": negative_light(lin), "temporal": temporal(lin),
           "gamut": gamut(lin), "alpha": alpha_check(lin, alpha, premultiplied)}
    if log_codes is not None and code:
        rep["footroom_violations"] = H.footroom_fraction(log_codes, code)
    if ref is not None:
        rep["reference"] = compare(lin, np.asarray(ref, np.float32), white_nits)
    rep["verdict"] = verdict(rep)
    return rep


def verdict(rep: dict, lim: dict | None = None) -> dict:
    lim = dict(LIMITS, **(lim or {}))
    fail, check, info = [], [], []
    hl = rep["highlights"]
    if rep["stops_above_white"] < lim["stops_above_white_min"] or hl["share_above_white"] < 1e-4:
        fail.append(f"no highlight range: {rep['stops_above_white']:+.2f} stops above diffuse white")
    if rep["quantised"]["distinct_levels_per_channel"] <= lim["quantised_levels_max"]:
        fail.append(f"quantised source: only {rep['quantised']['distinct_levels_per_channel']} distinct values per "
                    "channel (an 8-bit picture in a deeper file)")
    if hl["plateau_fraction"] is not None and hl["plateau_fraction"] > lim["plateau_max"]:
        fail.append(f"clipped highlights: {hl['plateau_fraction']:.0%} of above-white pixels sit at one level")
    if hl["texture_stops"] is not None and hl["texture_stops"] < lim["highlight_texture_min"]:
        fail.append(f"no highlight detail: {hl['texture_stops']:.3f} stops of variation above white (a numeric "
                    "maximum, not highlight information)")
    if rep["negative_light"] > lim["negative_light_max"]:
        fail.append(f"negative light: {rep['negative_light']:.2%} of pixels below zero")
    fr = rep.get("footroom_violations")
    if fr is not None and fr > 0.01:
        fail.append(f"footroom: {fr:.2%} of log values below the black code (fake HDR)")
    t = rep["temporal"]
    if t:
        for k, lk, word in (("exposure_flicker_stops", "exposure_flicker_max", "exposure flicker"),
                            ("exposure_pumping_stops", "exposure_pumping_max", "exposure pumping"),
                            ("highlight_flicker_stops", "highlight_flicker_max", "highlight flicker"),
                            ("chroma_flicker_uv", "chroma_flicker_max", "chroma instability")):
            if t[k] > lim[lk]:
                check.append(f"{word}: {t[k]:.4f} (> {lim[lk]})")
        if t["discontinuities"]:
            check.append(f"temporal discontinuities at frames {t['discontinuities'][:8]}")
    else:
        info.append("single frame: temporal checks need video")
    if rep["gamut"]["negative_ap1"] > lim["gamut_ap1_max"]:
        check.append(f"gamut excursions: {rep['gamut']['negative_ap1']:.2%} of pixels outside ACEScg")
    a = rep["alpha"]
    if a.get("present") and not a.get("ok"):
        fail.append("alpha corrupted: " + ", ".join(k for k in ("finite", "outside_0_1", "colour_under_transparent")
                                                     if (k == "finite" and not a.get(k)) or (k != "finite" and a.get(k))))
    r = rep.get("reference")
    if r:
        al = r["alignment"]
        if al["frame_offset"] != 0 or al["shift_magnitude_px"] > lim["alignment_px_max"]:
            fail.append(f"misaligned with the reference: frame offset {al['frame_offset']}, shift "
                        f"{al['shift_magnitude_px']:.2f} px")
        info.append(f"PU21-PSNR {r['pu21_psnr']:.2f} dB vs the reference")
    status = "FAIL" if fail else ("CHECK" if check else "PASS")
    return {"status": status, "fail": fail, "check": check, "info": info,
            "meaning": "PASS = measured highlight information above diffuse white and no failed check; the file format "
                       "alone never passes"}
