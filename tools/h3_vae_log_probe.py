"""H3 VAE log-encoding probe (ledger L9.179; plan docs/research/l9178_minimax_hdr_rnd.md section 4).

Which log code should H3 learn to generate HDR in? The one its frozen VAE carries best. This round-trips REAL HDR
footage through the H3 video VAE in each candidate code space (ACEScct, ARRI LogC3, ARRI LogC4 by default) at several
exposure shifts and writes one JSON: PU21-PSNR, code-space PSNR, mean delta u'v' (colour drift; Dark Matters found
LogC3 loses shadow colour at -8 EV through LTX's VAE), per-stop luminance error and footroom violations.

Run it on the GPU box, from the ComfyUI folder's Python (the one with torch):

    python tools/h3_vae_log_probe.py --comfy /path/to/ComfyUI --vae /path/to/minimax_h3_vae.safetensors \
        --input /plates/shot010/shot010.####.exr --input-space ACEScg --out probe_shot010.json

    # a LogC3 ProRes straight from the camera / Nuke (the file holds LogC3 code values, AWG3):
    python tools/h3_vae_log_probe.py --comfy ... --vae ... --input A001C003.mov --input-space "ARRI LogC3" --out p.json

Run it on 3-5 different shots (night with lamps, sky, faces, fire) and send the JSON files back. Nothing is written
except the JSON. ``--fake-vae`` replaces the VAE by 8-bit quantisation + a small blur, to check the tool itself on
any machine (its numbers mean nothing about H3).
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import platform
import re
import sys
import time
from pathlib import Path

import numpy as np

PKG = Path(__file__).resolve().parents[1]
if str(PKG) not in sys.path:
    sys.path.insert(0, str(PKG))

from mmx_utils import hdr_log as H  # noqa: E402

_SEQ_EXT = (".exr", ".dpx", ".tif", ".tiff", ".png")


# ---- input ---------------------------------------------------------------------------------------------------------

def _sequence_files(path: str) -> list[str]:
    if os.path.isdir(path):
        files = [os.path.join(path, f) for f in os.listdir(path) if f.lower().endswith(_SEQ_EXT)]
    else:
        pat = re.sub(r"#+", "*", path)
        pat = re.sub(r"%0?\d*d", "*", pat)
        files = glob.glob(pat)
    return sorted(files)


def _read_still(path: str) -> np.ndarray:
    try:
        import OpenImageIO as oiio  # type: ignore[import-not-found]
    except ImportError as exc:
        raise SystemExit("Reading EXR / DPX / TIFF needs OpenImageIO (pip install OpenImageIO).") from exc
    buf = oiio.ImageBuf(path)
    if buf.has_error:
        raise SystemExit(f"cannot read {path}: {buf.geterror()}")
    px = np.asarray(buf.get_pixels(oiio.FLOAT), np.float32)
    if px.ndim == 2:
        px = px[..., None]
    if px.shape[-1] == 1:
        px = np.repeat(px, 3, axis=-1)
    return px[..., :3]


def _read_movie(path: str, frames: int) -> np.ndarray:
    try:
        import av  # type: ignore[import-not-found]
    except ImportError as exc:
        raise SystemExit("Reading a movie needs PyAV (pip install av).") from exc
    out = []
    with av.open(path) as c:
        for f in c.decode(video=0):
            out.append(f.to_ndarray(format="rgb48le").astype(np.float32) / 65535.0)
            if len(out) >= frames:
                break
    if not out:
        raise SystemExit(f"no video frames in {path}")
    return np.stack(out)


def read_frames(path: str, frames: int) -> np.ndarray:
    """[T, H, W, 3] float32 as stored in the file (scene-linear for EXR, code values for log movies / DPX)."""
    ext = os.path.splitext(path)[1].lower()
    if os.path.isfile(path) and ext not in _SEQ_EXT:
        return _read_movie(path, frames)
    files = _sequence_files(path) if not os.path.isfile(path) else [path]
    if not files:
        raise SystemExit(f"no image files match {path}")
    return np.stack([_read_still(f) for f in files[:frames]])


def to_acescg(values: np.ndarray, input_space: str) -> np.ndarray:
    """'ACEScg' = already ACEScg scene-linear; a curve name ('ARRI LogC3', 'ACEScct' ...) = code values in that curve's
    camera gamut; 'linear:<gamut>' = scene-linear in another gamut (e.g. 'linear:ARRI Wide Gamut 3')."""
    s = input_space.strip()
    if s == "ACEScg":
        return values.astype(np.float32)
    if s.startswith("linear:"):
        from mmx_vendor.hdr_curves._primaries import gamut_matrix
        m = gamut_matrix(s.split(":", 1)[1].strip(), H.WORKING_GAMUT)
        return np.einsum("ij,...j->...i", m, values.astype(np.float64)).astype(np.float32)
    return H.decode(values.astype(np.float32), s)


def conform(lin: np.ndarray, frames: int, width: int, height: int) -> np.ndarray:
    """H3's VAE grid: 17k+5 frames (22, 39, 56 ...), sides multiple of 16. Centre crop; never resample (a resize would
    change the very detail being measured)."""
    t = lin.shape[0]
    n = t if t < 22 else ((min(t, frames) - 5) // 17) * 17 + 5
    if n < 22:
        raise SystemExit(f"need at least 22 frames (H3 clips are 17k+5), the input has {t}")
    lin = lin[:n]
    h, w = lin.shape[1:3]
    ch, cw = min(height, h) // 16 * 16, min(width, w) // 16 * 16
    y0, x0 = (h - ch) // 2, (w - cw) // 2
    return np.ascontiguousarray(lin[:, y0:y0 + ch, x0:x0 + cw])


# ---- the round trip ------------------------------------------------------------------------------------------------

def fake_vae():
    """8-bit quantisation + a 3-tap blur: exercises the tool, says nothing about H3."""
    def rt(codes: np.ndarray) -> np.ndarray:
        q = np.round(codes * 255.0) / 255.0
        b = q.copy()
        b[:, 1:-1] = (q[:, :-2] + 2 * q[:, 1:-1] + q[:, 2:]) / 4.0
        return b.astype(np.float32)
    return rt, {"vae": "fake (8-bit + blur)"}


def comfy_vae(comfy_root: str, vae_path: str, device: str | None):
    if comfy_root:
        sys.path.insert(0, os.path.abspath(comfy_root))
    try:
        import torch
        import comfy.sd
        import comfy.utils
    except Exception as exc:  # noqa: BLE001 - the comfy_kitchen skew raises AttributeError, not ImportError
        raise SystemExit(f"cannot import ComfyUI from {comfy_root or 'sys.path'} ({exc}); pass --comfy <ComfyUI folder>")
    sd, metadata = comfy.utils.load_torch_file(vae_path, return_metadata=True)
    quant = comfy.utils.detect_layer_quantization(sd, "")
    vae = comfy.sd.VAE(sd=sd, metadata=metadata, **({"device": torch.device(device)} if device else {}))
    if type(vae.first_stage_model).__name__ != "MiniMaxH3VideoVAE":
        print(f"note: {os.path.basename(vae_path)} is a {type(vae.first_stage_model).__name__}, not the H3 video VAE",
              file=sys.stderr)
    with open(vae_path, "rb") as fh:
        head = hashlib.sha256(fh.read(1 << 20)).hexdigest()[:16]
    info = {"vae": os.path.basename(vae_path), "vae_bytes": os.path.getsize(vae_path), "vae_sha256_first_mb": head,
            "vae_class": type(vae.first_stage_model).__name__,
            "quantized": quant is not None,
            "torch": torch.__version__, "cuda": torch.cuda.is_available()}

    def rt(codes: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(np.ascontiguousarray(codes)).float()
        with torch.inference_mode():
            lat = vae.encode(x)
            img = vae.decode(lat)
        out = img.float().cpu().numpy().reshape(-1, *codes.shape[1:])
        if out.shape[0] < codes.shape[0]:
            raise SystemExit(f"the VAE returned {out.shape[0]} frames for {codes.shape[0]}")
        return out[: codes.shape[0]]
    return rt, info


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="EXR / DPX / TIFF sequence (shot.####.exr, a folder) or a movie")
    ap.add_argument("--input-space", default="ACEScg",
                    help="ACEScg (linear EXR) | a curve name the file is encoded in ('ARRI LogC3') | linear:<gamut>")
    ap.add_argument("--vae", help="the H3 video VAE file (.safetensors) you generate with")
    ap.add_argument("--comfy", default=os.environ.get("COMFYUI_PATH", ""), help="the ComfyUI folder")
    ap.add_argument("--device", default=None)
    ap.add_argument("--fake-vae", action="store_true", help="test the tool without H3 (numbers are meaningless)")
    ap.add_argument("--codes", default=",".join(H.DEFAULT_CODES))
    ap.add_argument("--exposures", default="0,-3,-6,-8", help="exposure shifts in stops")
    ap.add_argument("--frames", type=int, default=22)
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=544)
    ap.add_argument("--white-nits", type=float, default=100.0, help="scene-linear 1.0 in cd/m2 for PU21")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if not a.fake_vae and not a.vae:
        ap.error("--vae is required (or --fake-vae to test the tool)")

    raw = read_frames(a.input, max(a.frames, 22))
    lin = conform(to_acescg(raw, a.input_space), a.frames, a.width, a.height)
    rt, info = fake_vae() if a.fake_vae else comfy_vae(a.comfy, a.vae, a.device)
    report = {"tool": "h3_vae_log_probe", "version": 1, "input": os.path.basename(a.input),
              "input_space": a.input_space, "frames": int(lin.shape[0]), "size": [int(lin.shape[2]), int(lin.shape[1])],
              "input_stops_above_white": H.stops_above_white(lin), "machine": platform.platform(), **info,
              "codes": []}
    for code in [c.strip() for c in a.codes.split(",") if c.strip()]:
        t0 = time.time()
        rep = H.roundtrip_report(lin, rt, code, exposures=[float(e) for e in a.exposures.split(",")],
                                 white_nits=a.white_nits)
        rep["seconds"] = round(time.time() - t0, 2)
        report["codes"].append(rep)
        e0 = rep["exposures"].get("+0", {})
        print(f"{code:12s} PU21-PSNR {e0.get('pu21_psnr', float('nan')):6.2f} dB  "
              f"du'v' {e0.get('delta_uv_mean') or 0:.4f}  ({rep['seconds']} s)")
    Path(a.out).write_text(json.dumps(report, indent=1, default=float), encoding="utf-8")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
