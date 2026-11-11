"""Build an H3 HDR training set (ledger L9.180; plan docs/research/l9178_minimax_hdr_rnd.md section 7).

Sources (any mix):
  --hdri FILE_OR_DIR      equirectangular .exr / .hdr panoramas (Poly Haven, CC0: polyhaven.com/hdris). Each one becomes
                          --orbits-per-hdri camera pans; Rec.709 linear, converted to ACEScg.
  --sequence PATTERN      your own plates, written from Nuke as ACEScg scene-linear half EXR (shot.####.exr). Cut into
                          17n+5-frame windows. A caption goes in <pattern stem>.txt next to the frames (optional).

Output (--out DIR), laid out for ai-toolkit (docs/research/l9178_h3_hdr_log_lora.yaml):
  train/clipNNNN.mkv + .txt   Track B target: the log code (ACEScct by default), dithered 8-bit RGB, FFV1 lossless
  val/clipNNNN.mkv + .txt     held-out parents (no source in both halves)
  sdr/<split>/clipNNNN_vK.mp4 Track C inputs: degraded SDR renderings (H.264); val also gets the held-out tone curve
  trackc/<split>/clipNNNN_vK.mkv + .txt   with --trackc: Track C pairs, ONE clip twice as wide - left the SDR
                              rendering as encoded (H.264 decoded back), right the same frames in the log code
                              (In-Context LoRA, arXiv 2410.23775; train as plain t2v). Use 832 x 480 panels.
  manifest.json               every clip's parent, exposure scale, clipping, stops above white and SDR recipes

    python tools/h3_hdr_dataset.py --hdri D:/hdri --sequence "D:/plates/sh010/sh010.####.exr" --out D:/h3_hdr_set

Nothing is downloaded. Clips are processed one at a time (about 250 MB of RAM for 39 x 544 x 960).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np

PKG = Path(__file__).resolve().parents[1]
if str(PKG) not in sys.path:
    sys.path.insert(0, str(PKG))

from mmx_utils import hdr_dataset as D  # noqa: E402
from mmx_utils import hdr_log as H  # noqa: E402

FPS = 24


# ---- reading ---------------------------------------------------------------------------------------------------------

def read_float_image(path: str) -> np.ndarray:
    import OpenImageIO as oiio  # type: ignore[import-not-found]
    buf = oiio.ImageBuf(path)
    if buf.has_error:
        raise SystemExit(f"cannot read {path}: {buf.geterror()}")
    px = np.asarray(buf.get_pixels(oiio.FLOAT), np.float32)
    if px.ndim == 2:
        px = px[..., None]
    if px.shape[-1] == 1:
        px = np.repeat(px, 3, axis=-1)
    return px[..., :3]


def hdri_files(paths: list[str]) -> list[str]:
    out = []
    for p in paths:
        if os.path.isdir(p):
            out += [os.path.join(p, f) for f in sorted(os.listdir(p)) if f.lower().endswith((".exr", ".hdr"))]
        elif os.path.isfile(p):
            out.append(p)
        else:
            raise SystemExit(f"no HDRI at {p}")
    return out


def sequence_files(pattern: str) -> list[str]:
    files = sorted(glob.glob(re.sub(r"%0?\d*d", "*", re.sub(r"#+", "*", pattern))))
    if not files:
        raise SystemExit(f"no frames match {pattern}")
    return files


def fit(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    """Scale to cover width x height (area filter, scene-linear safe), then centre-crop."""
    import cv2
    h, w = frame.shape[:2]
    s = max(width / w, height / h)
    nw, nh = max(width, round(w * s)), max(height, round(h * s))
    r = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
    y0, x0 = (nh - height) // 2, (nw - width) // 2
    return np.ascontiguousarray(r[y0:y0 + height, x0:x0 + width])


# ---- writing ---------------------------------------------------------------------------------------------------------

def write_ffv1(path: str, frames_u8: np.ndarray) -> None:
    import av  # type: ignore[import-not-found]
    with av.open(path, mode="w") as c:
        s = c.add_stream("ffv1", rate=Fraction(FPS, 1))
        s.width, s.height, s.pix_fmt = frames_u8.shape[2], frames_u8.shape[1], "bgr0"
        for i, fr in enumerate(frames_u8):
            vf = av.VideoFrame.from_ndarray(np.ascontiguousarray(fr), format="rgb24").reformat(format="bgr0")
            vf.pts = i
            for pkt in s.encode(vf):
                c.mux(pkt)
        for pkt in s.encode(None):
            c.mux(pkt)


def read_rgb_u8(path: str) -> np.ndarray:
    """[T, H, W, 3] uint8 of a video file (the SDR as a model will really see it, compression included)."""
    import av  # type: ignore[import-not-found]
    with av.open(path) as c:
        return np.stack([fr.to_ndarray(format="rgb24") for fr in c.decode(video=0)])


TRACKC_TAG = "[HDRPAIR]"


def trackc_caption(caption: str, code: str) -> str:
    return (f"{TRACKC_TAG} left: an SDR video; right: the same shot as {code} HDR log footage, frame for frame. "
            f"{caption}")


def write_h264(path: str, frames01: np.ndarray, crf: int) -> None:
    import av  # type: ignore[import-not-found]
    u8 = np.clip(np.round(frames01 * 255.0), 0, 255).astype(np.uint8)
    with av.open(path, mode="w") as c:
        s = c.add_stream("libx264", rate=Fraction(FPS, 1))
        s.width, s.height, s.pix_fmt = u8.shape[2], u8.shape[1], "yuv420p"
        s.options = {"crf": str(int(crf)), "preset": "medium"}
        s.codec_context.color_primaries, s.codec_context.color_trc, s.codec_context.colorspace = 1, 13, 1
        for i, fr in enumerate(u8):
            vf = av.VideoFrame.from_ndarray(fr, format="rgb24")
            vf.pts = i
            for pkt in s.encode(vf):
                c.mux(pkt)
        for pkt in s.encode(None):
            c.mux(pkt)


# ---- the build -------------------------------------------------------------------------------------------------------

def clip_sources(a) -> list[dict]:
    """Every clip to make, before any pixel is touched: (parent, kind, caption, how to make the frames)."""
    rng = np.random.default_rng(a.seed)
    out = []
    for f in hdri_files(a.hdri or []):
        name = Path(f).stem
        words = re.sub(r"[_\-]+", " ", re.sub(r"_\d+k$", "", name)).strip()
        for k in range(a.orbits_per_hdri):
            out.append({"parent": f"hdri:{name}", "kind": "hdri_orbit", "source": f,
                        "caption": f"{words}, real HDR environment, camera pan",
                        "orbit": {"yaw0_deg": float(rng.uniform(0, 360)), "yaw_span_deg": float(rng.uniform(15, 50)),
                                  "pitch_deg": float(rng.uniform(-10, 20)), "fov_deg": float(rng.uniform(45, 80))}})
    for pat in a.sequence or []:
        files = sequence_files(pat)
        stem = re.sub(r"[._]?(#+|%0?\d*d|\*)\.\w+$", "", pat)
        cap_file = stem + ".txt"
        caption = Path(cap_file).read_text(encoding="utf-8").strip() if os.path.isfile(cap_file) else a.caption
        for w0 in range(0, len(files) - a.frames + 1, a.frames):
            out.append({"parent": f"seq:{stem}", "kind": "plate", "source": pat, "caption": caption,
                        "files": files[w0:w0 + a.frames], "first": w0})
    if not out:
        raise SystemExit("nothing to build: give --hdri and / or --sequence")
    return out


def make_frames(src: dict, a, eq_cache: dict) -> np.ndarray:
    if src["kind"] == "hdri_orbit":
        if src["source"] not in eq_cache:
            eq_cache.clear()                                         # one panorama in RAM at a time
            eq_cache[src["source"]] = D.rec709_to_acescg(read_float_image(src["source"]))
        o = src["orbit"]
        lin = D.orbit_frames(eq_cache[src["source"]], frames=a.frames, width=a.width, height=a.height,
                             fov_deg=o["fov_deg"], yaw0_deg=o["yaw0_deg"], yaw_span_deg=o["yaw_span_deg"],
                             pitch_deg=o["pitch_deg"])
        return D.normalise_exposure(lin, a.exposure)
    lin = np.stack([fit(read_float_image(f), a.width, a.height) for f in src["files"]])
    return D.normalise_exposure(plate_to_acescg(lin, a.plate_space), a.plate_exposure)


def plate_to_acescg(values: np.ndarray, space: str) -> np.ndarray:
    """The probe's input-space rules: 'ACEScg'; a curve name (code values in that curve's gamut); 'linear:<gamut>'."""
    s = space.strip()
    if s == "ACEScg":
        return values
    if s.startswith("linear:"):
        from mmx_vendor.hdr_curves._primaries import gamut_matrix
        return H._apply(gamut_matrix(s.split(":", 1)[1].strip(), H.WORKING_GAMUT), values)
    return H.decode(values, s)


def build(a) -> dict:
    a.frames, a.width, a.height = D.snap_frames(a.frames), D.snap_side(a.width), D.snap_side(a.height)
    out = Path(a.out)
    sources = clip_sources(a)
    split = D.split_by_parent([s["parent"] for s in sources], a.val_fraction)
    rng = np.random.default_rng(a.seed + 1)
    train_curves = tuple(c for c in D.TONE_CURVES if c != a.held_out_curve)
    manifest = {"tool": "h3_hdr_dataset", "version": 1, "code": a.code, "frames": a.frames, "fps": FPS,
                "size": [a.width, a.height], "held_out_curve": a.held_out_curve, "clips": []}
    eq_cache: dict = {}
    for n, src in enumerate(sources, start=1):
        part = split[src["parent"]]
        cid = f"clip{n:04d}"
        (out / part).mkdir(parents=True, exist_ok=True)
        lin, scale = make_frames(src, a, eq_cache)
        codes, stats = D.to_training_codes(lin, a.code, seed=a.seed + n)
        write_ffv1(str(out / part / f"{cid}.mkv"), codes)
        (out / part / f"{cid}.txt").write_text(src["caption"], encoding="utf-8")
        rec = {"id": cid, "split": part, "parent": src["parent"], "kind": src["kind"], "source": src["source"],
               "caption": src["caption"], "exposure_scale": scale, **stats, "sdr": []}
        if src["kind"] == "plate":
            rec["first_frame_index"] = src["first"]
        if src["kind"] == "hdri_orbit":
            rec["orbit"] = src["orbit"]
        recipes = [D.random_sdr_recipe(rng, train_curves) for _ in range(a.sdr_variants)]
        if part == "val" and a.sdr_variants and a.held_out_curve:
            recipes.append({**D.random_sdr_recipe(rng, (a.held_out_curve,)), "held_out": True})
        for k, r in enumerate(recipes):
            sdir = out / "sdr" / part
            sdir.mkdir(parents=True, exist_ok=True)
            sdr = D.sdr_variant(lin, curve=r["curve"], ev=r["ev"], contrast=r["contrast"],
                                highlight_blur_sigma=r["highlight_blur_sigma"], shadow_soften=r["shadow_soften"])
            name = f"{cid}_v{k}.mp4"
            write_h264(str(sdir / name), sdr, r["crf"])
            entry = {"file": f"sdr/{part}/{name}", **r}
            if a.trackc:
                left = read_rgb_u8(str(sdir / name))[: codes.shape[0]]
                if left.shape != codes.shape:
                    raise SystemExit(f"{cid}: the decoded SDR {left.shape} does not match the codes {codes.shape}")
                cdir = out / "trackc" / part
                cdir.mkdir(parents=True, exist_ok=True)
                write_ffv1(str(cdir / f"{cid}_v{k}.mkv"), np.concatenate([left, codes], axis=2))
                (cdir / f"{cid}_v{k}.txt").write_text(trackc_caption(src["caption"], a.code), encoding="utf-8")
                entry["trackc"] = f"trackc/{part}/{cid}_v{k}.mkv"
            rec["sdr"].append(entry)
        manifest["clips"].append(rec)
        print(f"{cid} {part:5s} {src['kind']:10s} {rec['stops_above_white']:+.2f} stops, clipped>1 "
              f"{rec['clipped_above_1']:.3%}, {len(rec['sdr'])} SDR", flush=True)
        del lin, codes
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    return manifest


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hdri", action="append", help="equirect .exr / .hdr file or folder (repeatable)")
    ap.add_argument("--sequence", action="append", help="ACEScg EXR sequence pattern (repeatable)")
    ap.add_argument("--plate-space", default="ACEScg", help="what --sequence frames hold (see the probe's --input-space)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--code", default="ACEScct", help="the log code to train in (decide with h3_vae_log_probe)")
    ap.add_argument("--frames", type=int, default=39, help="snapped down to 17n+5")
    ap.add_argument("--width", type=int, default=960, help="snapped down to a multiple of 32")
    ap.add_argument("--height", type=int, default=544)
    ap.add_argument("--orbits-per-hdri", type=int, default=4)
    ap.add_argument("--exposure", default="p99_to_1", choices=D.EXPOSURE_MODES, help="for HDRI orbits")
    ap.add_argument("--plate-exposure", default="none", choices=D.EXPOSURE_MODES, help="for your plates")
    ap.add_argument("--sdr-variants", type=int, default=3, help="Track C SDR renderings per clip (0 = none)")
    ap.add_argument("--held-out-curve", default="hable", choices=D.TONE_CURVES)
    ap.add_argument("--trackc", action="store_true",
                    help="also write Track C side-by-side [SDR | log] clips (needs --sdr-variants > 0)")
    ap.add_argument("--val-fraction", type=float, default=0.08)
    ap.add_argument("--caption", default="real HDR camera footage")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    m = build(a)
    print(f"{len(m['clips'])} clips -> {a.out} (manifest.json)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
