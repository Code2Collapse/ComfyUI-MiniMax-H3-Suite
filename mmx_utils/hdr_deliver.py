"""H3 HDR delivery (ledger L9.212, A26 3.7): decoded frames -> validated log interpretation -> scene-linear ACEScg ->
exposure -> optional alpha composite -> HDR QC -> EXR / DPX / HLG, each with colour metadata and the QC report beside it.

- EXR: half float, ACEScg scene-linear, alpha kept, ``chromaticities`` = AP1 + ACES white, c2c:* attributes (transfer,
  exposure, source code space, QC verdict). (OIIO rewrites oiio:ColorSpace to its own id, so chromaticities is the
  authoritative tag.)
- DPX: the log code (any NukeMax curve, e.g. 'ARRI LogC3' in ARRI Wide Gamut 3) at 10 or 16 bits, transfer "Logarithmic";
  DPX's description field holds 32 characters, so the full curve / gamut name is in the sidecar JSON.
- HLG: ITU-R BT.2100 HLG, BT.2020 primaries, BT.2020 non-constant-luminance Y'CbCr, narrow range, 10-bit 4:4:4 HEVC
  (libx265), container tags primaries 9 / transfer 18 (ARIB STD-B67) / matrix 9 / range limited. Scene light is mapped
  so diffuse white (1.0) = 75 % HLG signal (BT.2408: 203 cd/m2 on a 1000 cd/m2 display), so HLG holds about 1.9
  stops above diffuse white; everything brighter is clipped and the report says how much.
After writing, every file is read back and its metadata and values are checked.
"""
from __future__ import annotations

import json
import math
from fractions import Fraction
from pathlib import Path

import numpy as np

from . import hdr_log as H
from . import hdr_qc as Q

AP1_CHROMATICITIES = [0.713, 0.293, 0.165, 0.830, 0.128, 0.044, 0.32168, 0.33767]
# BT.2100 HLG OETF constants
HLG_A, HLG_B, HLG_C = 0.17883277, 1.0 - 4 * 0.17883277, 0.5 - 0.17883277 * math.log(4 * 0.17883277)
HLG_REF_WHITE_SIGNAL = 0.75                                        # BT.2408 reference white
KR, KB = 0.2627, 0.0593                                            # BT.2020 non-constant luminance


def hlg_oetf(e: np.ndarray) -> np.ndarray:
    e = np.clip(np.asarray(e, np.float64), 0.0, 1.0)
    return np.where(e <= 1 / 12, np.sqrt(3 * e), HLG_A * np.log(np.maximum(12 * e - HLG_B, 1e-12)) + HLG_C)


def hlg_inverse_oetf(v: np.ndarray) -> np.ndarray:
    v = np.clip(np.asarray(v, np.float64), 0.0, 1.0)
    return np.where(v <= 0.5, v * v / 3, (np.exp((v - HLG_C) / HLG_A) + HLG_B) / 12)


HLG_E_PER_WHITE = float(hlg_inverse_oetf(np.array(HLG_REF_WHITE_SIGNAL)))  # ~0.2650: scene E of diffuse white
HLG_STOPS_ABOVE_WHITE = math.log2(1.0 / HLG_E_PER_WHITE)                   # ~1.92


def to_scene_linear(values: np.ndarray, input_space: str) -> np.ndarray:
    """'ACEScg' | a curve name (decoded log code values, e.g. H3 + a log LoRA) | 'linear:<gamut>'."""
    s = input_space.strip()
    if s == "ACEScg":
        return np.asarray(values, np.float32)
    if s.startswith("linear:"):
        from mmx_vendor.hdr_curves._primaries import gamut_matrix
        return H._apply(gamut_matrix(s.split(":", 1)[1].strip(), H.WORKING_GAMUT), values)
    return H.decode(np.asarray(values, np.float32), s)


def prepare(values, input_space, *, exposure_ev=0.0, alpha=None, background=None, premultiplied=False):
    """(scene-linear ACEScg, alpha out, notes). ``background`` (same size, ACEScg) = composite over it."""
    lin = to_scene_linear(values, input_space) * np.float32(2.0 ** float(exposure_ev))
    notes = {"input_space": input_space, "exposure_ev": float(exposure_ev)}
    if alpha is not None and background is not None:
        a = np.clip(np.asarray(alpha, np.float32), 0, 1)[..., None]
        fg = lin if premultiplied else lin * a
        lin = fg + np.asarray(background, np.float32) * (1 - a)
        notes["composite"] = "over background (" + ("premultiplied" if premultiplied else "straight") + " alpha)"
        alpha = None
    return lin.astype(np.float32), alpha, notes


# ── writers ─────────────────────────────────────────────────────────────────────────────────────────────────────
def write_exr(folder: Path, name: str, lin: np.ndarray, alpha=None, attrs: dict | None = None) -> list[Path]:
    import OpenImageIO as oiio
    folder.mkdir(parents=True, exist_ok=True)
    out = []
    for t, f in enumerate(lin):
        ch = 4 if alpha is not None else 3
        px = np.concatenate([f, alpha[t][..., None]], -1) if alpha is not None else f
        spec = oiio.ImageSpec(f.shape[1], f.shape[0], ch, oiio.HALF)
        if alpha is not None:
            spec.alpha_channel = 3
        spec.attribute("chromaticities", oiio.TypeDesc("float[8]"), AP1_CHROMATICITIES)
        spec.attribute("oiio:ColorSpace", "ACEScg")
        for k, v in (attrs or {}).items():
            spec.attribute(f"c2c:{k}", v if isinstance(v, (int, float)) else str(v))
        p = folder / f"{name}.{t + 1001:04d}.exr"
        o = oiio.ImageOutput.create(str(p))
        if not o.open(str(p), spec):
            raise OSError(f"cannot write {p}: {oiio.geterror()}")
        o.write_image(np.ascontiguousarray(px, np.float32))
        o.close()
        out.append(p)
    return out


def write_dpx(folder: Path, name: str, codes: np.ndarray, bits: int, tag: str) -> list[Path]:
    import OpenImageIO as oiio
    if bits not in (10, 16):
        raise ValueError("DPX bit depth must be 10 or 16.")
    folder.mkdir(parents=True, exist_ok=True)
    out = []
    for t, f in enumerate(codes):
        spec = oiio.ImageSpec(f.shape[1], f.shape[0], 3, oiio.UINT16)
        spec.attribute("oiio:BitsPerSample", bits)
        spec.attribute("dpx:Transfer", "Logarithmic")
        spec.attribute("dpx:Colorimetric", "User defined")
        spec.attribute("ImageDescription", tag[:32])
        p = folder / f"{name}.{t + 1001:04d}.dpx"
        o = oiio.ImageOutput.create(str(p))
        if not o.open(str(p), spec):
            raise OSError(f"cannot write {p}: {oiio.geterror()}")
        o.write_image(np.ascontiguousarray(np.clip(f, 0, 1), np.float32))
        o.close()
        out.append(p)
    return out


def hlg_signal(lin: np.ndarray) -> tuple[np.ndarray, float]:
    """ACEScg scene-linear -> BT.2020 HLG non-linear R'G'B' in 0..1, and the share of pixels clipped at signal 1."""
    from mmx_vendor.hdr_curves._primaries import gamut_matrix
    rgb = H._apply(gamut_matrix(H.WORKING_GAMUT, "Rec.2020"), lin).astype(np.float64)
    e = np.maximum(rgb, 0.0) * HLG_E_PER_WHITE
    clipped = float((e > 1.0).any(-1).mean())
    return hlg_oetf(e), clipped


def ycbcr10(rgbp: np.ndarray) -> np.ndarray:
    """BT.2020 NCL R'G'B' (0..1) -> [3, H, W] uint16 narrow-range 10-bit Y'CbCr."""
    r, g, b = rgbp[..., 0], rgbp[..., 1], rgbp[..., 2]
    y = KR * r + (1 - KR - KB) * g + KB * b
    cb = (b - y) / (2 * (1 - KB))
    cr = (r - y) / (2 * (1 - KR))
    q = [np.round(64 + 876 * y), np.round(512 + 896 * cb), np.round(512 + 896 * cr)]
    return np.clip(np.stack(q), 4, 1019).astype(np.uint16)


def rgb_from_ycbcr10(ycc: np.ndarray) -> np.ndarray:
    y = (ycc[0].astype(np.float64) - 64) / 876
    cb = (ycc[1].astype(np.float64) - 512) / 896
    cr = (ycc[2].astype(np.float64) - 512) / 896
    r = y + 2 * (1 - KR) * cr
    b = y + 2 * (1 - KB) * cb
    g = (y - KR * r - KB * b) / (1 - KR - KB)
    return np.stack([r, g, b], -1)


def write_hlg(path: Path, lin: np.ndarray, fps: float = 24.0, lossless: bool = True, crf: int = 8) -> dict:
    import av
    path.parent.mkdir(parents=True, exist_ok=True)
    sig, clipped = hlg_signal(lin)
    t, h, w = sig.shape[:3]
    with av.open(str(path), "w") as c:
        s = c.add_stream("libx265", rate=Fraction(fps).limit_denominator(1001))
        s.width, s.height, s.pix_fmt = w, h, "yuv444p10le"
        s.options = {"x265-params": ("lossless=1" if lossless else f"crf={int(crf)}") +
                     ":colorprim=bt2020:transfer=arib-std-b67:colormatrix=bt2020nc:range=limited:log-level=error"}
        cc = s.codec_context
        cc.color_primaries, cc.color_trc, cc.colorspace, cc.color_range = 9, 18, 9, 1
        for i in range(t):
            planes = ycbcr10(sig[i])
            vf = av.VideoFrame(w, h, "yuv444p10le")
            for pl, a in zip(vf.planes, planes):
                buf = np.zeros((h, pl.line_size // 2), np.uint16)
                buf[:, :w] = a
                pl.update(buf.tobytes())
            vf.pts = i
            for pkt in s.encode(vf):
                c.mux(pkt)
        for pkt in s.encode(None):
            c.mux(pkt)
    return {"clipped_above_hlg_peak": clipped, "stops_above_white_representable": HLG_STOPS_ABOVE_WHITE}


# ── read-back validation ───────────────────────────────────────────────────────────────────────────────────────
def check_exr(files: list[Path], lin: np.ndarray, alpha) -> dict:
    import OpenImageIO as oiio
    errs, worst = [], 0.0
    for t, p in enumerate(files):
        b = oiio.ImageBuf(str(p))
        sp = b.spec()
        px = np.asarray(b.get_pixels(oiio.FLOAT), np.float32)
        chrom = sp.getattribute("chromaticities")
        if chrom is None or max(abs(a - b_) for a, b_ in zip(chrom, AP1_CHROMATICITIES)) > 1e-4:
            errs.append(f"{p.name}: chromaticities missing or not AP1")
        if str(sp.format) != "half":
            errs.append(f"{p.name}: not half float")
        if alpha is not None and sp.alpha_channel != 3:
            errs.append(f"{p.name}: alpha channel lost")
        ref = lin[t]
        rel = np.abs(px[..., :3] - ref) / np.maximum(np.abs(ref), 1e-3)
        worst = max(worst, float(rel.max()))
    if worst > 2e-3:
        errs.append(f"values differ by up to {worst:.2e} (relative) from the half-float ideal")
    return {"ok": not errs, "problems": errs, "max_relative_error": worst}


def check_dpx(files: list[Path], codes: np.ndarray, bits: int) -> dict:
    import OpenImageIO as oiio
    errs, worst = [], 0.0
    for t, p in enumerate(files):
        b = oiio.ImageBuf(str(p))
        sp = b.spec()
        if sp.getattribute("oiio:BitsPerSample") != bits:
            errs.append(f"{p.name}: {sp.getattribute('oiio:BitsPerSample')} bits, expected {bits}")
        if sp.getattribute("dpx:Transfer") != "Logarithmic":
            errs.append(f"{p.name}: transfer is not Logarithmic")
        px = np.asarray(b.get_pixels(oiio.FLOAT), np.float32)
        worst = max(worst, float(np.abs(px - np.clip(codes[t], 0, 1)).max()))
    if worst > 1.0 / ((1 << bits) - 1):
        errs.append(f"code values differ by up to {worst:.2e} (more than one {bits}-bit step)")
    return {"ok": not errs, "problems": errs, "max_code_error": worst}


def check_hlg(path: Path, lin: np.ndarray, lossless: bool) -> dict:
    import av
    errs = []
    sig, _ = hlg_signal(lin)
    want = np.stack([ycbcr10(f) for f in sig])
    with av.open(str(path)) as c:
        st = c.streams.video[0]
        cc = st.codec_context
        tags = {"pix_fmt": cc.pix_fmt, "primaries": int(cc.color_primaries), "transfer": int(cc.color_trc),
                "matrix": int(cc.colorspace), "range": int(cc.color_range)}
        got = []
        for f in c.decode(video=0):
            got.append(np.stack([np.frombuffer(bytes(pl), np.uint16).reshape(f.height, -1)[:, :f.width]
                                 for pl in f.planes]))
    got = np.stack(got)
    if tags["pix_fmt"] != "yuv444p10le":
        errs.append(f"pixel format {tags['pix_fmt']}, expected yuv444p10le (10-bit)")
    for k, v in (("primaries", 9), ("transfer", 18), ("matrix", 9), ("range", 1)):
        if tags[k] != v:
            errs.append(f"{k} tag {tags[k]}, expected {v}")
    if got.shape != want.shape:
        errs.append(f"decoded {got.shape} vs written {want.shape}")
        diff = None
    else:
        diff = int(np.abs(got.astype(np.int32) - want.astype(np.int32)).max())
        if lossless and diff != 0:
            errs.append(f"lossless HLG decoded with code differences up to {diff}")
    return {"ok": not errs, "problems": errs, "tags": tags, "max_code_difference": diff}


def deliver(values, input_space, out_dir, name, *, exr=True, dpx_code=None, dpx_bits=16, hlg=False, fps=24.0,
            exposure_ev=0.0, alpha=None, background=None, premultiplied=False, reference=None, hlg_lossless=True):
    """Run the whole path; returns the delivery report (also written as <name>.delivery.json beside the outputs)."""
    out_dir = Path(out_dir)
    lin, alpha_out, notes = prepare(values, input_space, exposure_ev=exposure_ev, alpha=alpha, background=background,
                                    premultiplied=premultiplied)
    log_codes = np.asarray(values, np.float32) if input_space not in ("ACEScg",) and \
        not input_space.startswith("linear:") else None
    qc = Q.run(lin, alpha=alpha_out, premultiplied=premultiplied, ref=reference, log_codes=log_codes,
               code=input_space if log_codes is not None else None)
    rep = {"tool": "h3_hdr_deliver", "version": 1, "name": name, "prepare": notes, "qc": qc, "outputs": {}}
    meta = {"transfer": "linear", "gamut": "ACEScg (AP1)", "exposure_ev": float(exposure_ev),
            "source_code_space": input_space, "qc_verdict": qc["verdict"]["status"]}
    if exr:
        files = write_exr(out_dir / "exr", name, lin, alpha_out, meta)
        rep["outputs"]["exr"] = {"files": len(files), "first": str(files[0]), "metadata": meta,
                                 "readback": check_exr(files, lin, alpha_out)}
    if dpx_code:
        codes = H.encode(lin, dpx_code)
        clipped = float(((codes > 1) | (codes < 0)).any(-1).mean())
        tag = f"{dpx_code} / {H.code_gamut(dpx_code)}"
        files = write_dpx(out_dir / "dpx", name, codes, dpx_bits, tag)
        rep["outputs"]["dpx"] = {"files": len(files), "first": str(files[0]), "curve": dpx_code,
                                 "gamut": H.code_gamut(dpx_code), "bits": dpx_bits, "header_description": tag[:32],
                                 "clipped_codes": clipped, "readback": check_dpx(files, codes, dpx_bits)}
    if hlg:
        p = out_dir / f"{name}_hlg.mkv"
        info = write_hlg(p, lin, fps=fps, lossless=hlg_lossless)
        rep["outputs"]["hlg"] = {"file": str(p), **info, "readback": check_hlg(p, lin, hlg_lossless)}
    rep["ok"] = all(o["readback"]["ok"] for o in rep["outputs"].values())
    rep["hdr"] = qc["verdict"]["status"]
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{name}.delivery.json").write_text(json.dumps(rep, indent=1, default=float), encoding="utf-8")
    return rep
