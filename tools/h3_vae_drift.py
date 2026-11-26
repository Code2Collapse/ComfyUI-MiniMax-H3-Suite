"""H3 VAE round-trip drift measurement (ledger L8.17): colour shift, red drift, softness, chunk seams, out-of-range
values over repeated round trips, for one or more H3 video VAEs (e.g. the original and lovartai's H3-MT-VAE, whose
decoder is fine-tuned for repeated round trips), and what the Suite's colour_lock removes.

Run it on a machine with a GPU, from ComfyUI's Python (the one with torch). Nothing is downloaded, only the JSON is
written:

    python tools/h3_vae_drift.py --comfy /path/to/ComfyUI --video clip.mp4 --out drift.json \
        --vae models/vae/minimax_h3_video_vae_fp16.safetensors models/vae/minimax_h3_video_vae_mt_fp16.safetensors

Measured per VAE and round (1, 3, 5 by default), against the source frames:
  psnr, mean shift per channel (8-bit units; positive red = red drift), red minus green, sharpness ratio (Laplacian
  energy out / in; below 1 = softer), largest error, share of values outside 0..1 (what a clamp would cut), the
  lowest per-frame PSNR, the PSNR dip at the 17-frame chunk seams (frames 17, 34, ...), and the same after
  colour_lock (global method, strength 1).

Measured on CPU only as far as it goes: on a 16 GB CPU box the 5.2 GB fp16 VAE did not finish one 5-frame 128 px
encode in 25 minutes in fp16 or bf16 (CPU half-precision 3-D convolutions), and fp32 needs more than 10 GB.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import time
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]
if str(PKG) not in sys.path:
    sys.path.insert(0, str(PKG))


def read_video(path, frames, width, height):
    import av
    import torch
    import comfy.utils
    c = av.open(path)
    out = []
    for f in c.decode(video=0):
        out.append(torch.from_numpy(f.to_ndarray(format="rgb24")).float() / 255.0)
        if len(out) >= frames:
            break
    c.close()
    if not out:
        raise SystemExit(f"no frames in {path}")
    x = torch.stack(out)
    while x.shape[0] < frames:                       # ping-pong a short clip up to the asked length
        x = torch.cat([x, x.flip(0)])[:frames]
    x = comfy.utils.common_upscale(x.movedim(-1, 1), width, height, "bilinear", "center").movedim(1, -1)
    return x.clamp(0, 1)


def lap_energy(x):
    g = x.mean(-1)
    lap = 4 * g[:, 1:-1, 1:-1] - g[:, :-2, 1:-1] - g[:, 2:, 1:-1] - g[:, 1:-1, :-2] - g[:, 1:-1, 2:]
    return float(lap.pow(2).mean())


def psnr(a, b):
    mse = float((a - b).pow(2).mean())
    return 99.0 if mse == 0 else 10 * math.log10(1.0 / mse)


def stats(out, ref):
    d = (out - ref).mean(dim=(0, 1, 2)) * 255.0
    per_frame = [psnr(out[i], ref[i]) for i in range(out.shape[0])]
    seams = [s for s in range(17, len(per_frame), 17)]
    dips = [round(sum(per_frame[s - 2:s]) / 2 - per_frame[s], 3) for s in seams]
    return {"psnr": round(psnr(out, ref), 3),
            "mean_shift_rgb_8bit": [round(float(v), 3) for v in d],
            "red_minus_green_8bit": round(float(d[0] - d[1]), 3),
            "sharpness_ratio": round(lap_energy(out) / max(lap_energy(ref), 1e-12), 4),
            "max_abs_8bit": round(float((out - ref).abs().max()) * 255, 2),
            "outside_0_1_pct": round(float(((out < 0) | (out > 1)).float().mean()) * 100, 4),
            "per_frame_psnr_min": round(min(per_frame), 3),
            "seam_psnr_dip_db": dips}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--comfy", default="", help="ComfyUI folder (if it is not on sys.path)")
    ap.add_argument("--video", required=True)
    ap.add_argument("--vae", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames", type=int, default=39, help="17k+5 (39 covers two chunk seams)")
    ap.add_argument("--width", type=int, default=768)
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--rounds", default="1,3,5")
    a = ap.parse_args(argv)
    if a.frames % 17 != 5:
        raise SystemExit("--frames must be 17k+5 (5, 22, 39, 56...)")
    if a.width % 32 or a.height % 32:
        raise SystemExit("--width / --height must be multiples of 32")
    if a.comfy:
        sys.path.insert(0, os.path.abspath(a.comfy))
    import torch
    import comfy.sd
    import comfy.utils
    from mmx_utils import colour_lock as CL

    rounds = sorted({int(r) for r in a.rounds.split(",")})
    src = read_video(a.video, a.frames, a.width, a.height)
    report = {"tool": "h3_vae_drift", "video": os.path.basename(a.video), "size": [a.width, a.height, a.frames],
              "torch": torch.__version__, "cuda": torch.cuda.is_available(),
              "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu", "vaes": {}}
    for path in a.vae:
        t0 = time.time()
        sd, meta = comfy.utils.load_torch_file(path, return_metadata=True)
        vae = comfy.sd.VAE(sd=sd, metadata=meta)
        del sd
        res = {"class": type(vae.first_stage_model).__name__, "vae_dtype": str(getattr(vae, "vae_dtype", ""))}
        x = src
        with torch.inference_mode():
            for r in range(1, max(rounds) + 1):
                y = vae.decode(vae.encode(x))
                y = y.reshape(-1, *y.shape[-3:])[:a.frames].float().cpu()
                if r in rounds:
                    entry = stats(y, src)
                    entry["after_colour_lock"] = stats(CL.lock(y, src, strength=1.0, method="global"), src)
                    res[f"round_{r}"] = entry
                    print(os.path.basename(path), "round", r, json.dumps(entry), flush=True)
                x = y.clamp(0, 1)                    # the next round sees what a user would save
        res["seconds"] = round(time.time() - t0, 1)
        report["vaes"][os.path.basename(path)] = res
        del vae
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    Path(a.out).write_text(json.dumps(report, indent=1), encoding="utf-8")
    print("wrote", a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
