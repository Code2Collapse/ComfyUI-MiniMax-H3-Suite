"""HDR QC report for H3 HDR LoRA checkpoints (ledger L9.185; acceptance = docs/research/l9178_minimax_hdr_rnd.md 7.6).

Give it what a checkpoint generated, and, when there is one, the real HDR it should have produced:

    # a generated log video straight from H3 (the LoRA's code space), no reference:
    python tools/h3_hdr_qc.py --input sample_step2500.mp4 --input-space ACEScct --out qc_2500.json
    # decoded EXRs against the ground-truth plate:
    python tools/h3_hdr_qc.py --input out.####.exr --reference plate.####.exr --out qc.json

Reports (and a PASS / CHECK line per acceptance criterion):
  - fake-HDR footroom check (only when the input is in a log code: pixels below the curve's black = negative light);
  - stops above diffuse white (log2 of the 99.9th-percentile luminance), median over frames;
  - flicker: frame-to-frame std of the mean exposure and of the highlight level, in stops (trend removed);
  - with --reference: PU21-PSNR, per-stop luminance error, mean delta u'v' overall and in the shadows (< 2 % grey).
Send the JSON back; the frames stay with you.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
for p in (str(HERE.parent), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

from mmx_utils import hdr_log as H  # noqa: E402
import h3_vae_log_probe as PROBE  # noqa: E402  (same readers and input-space rules)

ACCEPT = {"stops_above_white_min": 2.0, "footroom_max": 0.01, "exposure_flicker_max_stops": 0.05,
          "highlight_flicker_max_stops": 0.15, "pu21_psnr_min": None}


def _detrend_std(v: np.ndarray) -> float:
    if v.size < 3:
        return 0.0
    t = np.arange(v.size, dtype=np.float64)
    a, b = np.polyfit(t, v.astype(np.float64), 1)
    return float(np.std(v - (a * t + b)))


def flicker(lin: np.ndarray) -> dict:
    y = H.luminance(lin).reshape(lin.shape[0], -1)
    mean_stops = np.log2(np.maximum(y.mean(axis=1), 1e-6))
    hi_stops = np.log2(np.maximum(np.percentile(y, 99.9, axis=1), 1e-6))
    return {"exposure_flicker_stops": _detrend_std(mean_stops), "highlight_flicker_stops": _detrend_std(hi_stops)}


def compare(lin: np.ndarray, ref: np.ndarray, white_nits: float) -> dict:
    n = min(lin.shape[0], ref.shape[0])
    lin, ref = lin[:n], ref[:n]
    if lin.shape != ref.shape:
        raise SystemExit(f"output {lin.shape[1:3]} and reference {ref.shape[1:3]} differ in size: crop them alike")
    duv = np.linalg.norm(H.uv_prime(lin) - H.uv_prime(ref), axis=-1)
    y = H.luminance(ref)
    lit, shadow = y >= 1e-3, (y >= 1e-3) & (y < 0.0036)            # 0.0036 = 2 % of 18 % grey
    return {"frames_compared": int(n), "pu21_psnr": H.pu21_psnr(lin, ref, white_nits),
            "per_stop_error": H.per_stop_error(lin, ref),
            "delta_uv_mean": float(duv[lit].mean()) if lit.any() else None,
            "delta_uv_shadows": float(duv[shadow].mean()) if shadow.sum() >= 16 else None,
            "stops_above_white_reference": H.stops_above_white(ref)}


def report(values: np.ndarray, input_space: str, ref_lin: np.ndarray | None = None, white_nits: float = 100.0) -> dict:
    lin = PROBE.to_acescg(values, input_space)
    per_frame = [H.stops_above_white(f[None]) for f in lin]
    rep = {"input_space": input_space, "frames": int(lin.shape[0]), "size": [int(lin.shape[2]), int(lin.shape[1])],
           "stops_above_white": float(np.median(per_frame)), "peak": float(lin.max()), **flicker(lin)}
    if input_space not in ("ACEScg",) and not input_space.startswith("linear:"):
        rep["footroom_violations"] = H.footroom_fraction(values, input_space)
        rep["at_ceiling"] = float((values[..., :3] >= 0.999).mean())
    if ref_lin is not None:
        rep["reference"] = compare(lin, ref_lin, white_nits)
    rep["checks"] = checks(rep)
    return rep


def checks(rep: dict) -> list[str]:
    out = []
    s = rep["stops_above_white"]
    out.append(f"{'PASS' if s >= ACCEPT['stops_above_white_min'] else 'CHECK'} highlight range {s:+.2f} stops above "
               f"white (target >= +{ACCEPT['stops_above_white_min']:.0f}; LumiVid +2.75)")
    f = rep.get("footroom_violations")
    if f is not None:
        out.append(f"{'PASS' if f <= ACCEPT['footroom_max'] else 'FAIL'} footroom: {f:.3%} of values below the black "
                   f"code ({'real log picture' if f <= ACCEPT['footroom_max'] else 'FAKE HDR: an SDR picture'})")
    for k, lim in (("exposure_flicker_stops", "exposure_flicker_max_stops"),
                   ("highlight_flicker_stops", "highlight_flicker_max_stops")):
        out.append(f"{'PASS' if rep[k] <= ACCEPT[lim] else 'CHECK'} {k.replace('_', ' ')} {rep[k]:.3f} "
                   f"(<= {ACCEPT[lim]})")
    r = rep.get("reference")
    if r:
        out.append(f"INFO PU21-PSNR {r['pu21_psnr']:.2f} dB vs the reference (compare with the probe's VAE ceiling)")
        if r.get("delta_uv_shadows") is not None:
            out.append(f"INFO shadow colour drift du'v' {r['delta_uv_shadows']:.4f} (overall {r['delta_uv_mean']:.4f})")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True)
    ap.add_argument("--input-space", default="ACEScg", help="ACEScg | a curve name (generated log) | linear:<gamut>")
    ap.add_argument("--reference", help="ground-truth HDR (same frames / size)")
    ap.add_argument("--reference-space", default="ACEScg")
    ap.add_argument("--frames", type=int, default=200)
    ap.add_argument("--white-nits", type=float, default=100.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    vals = PROBE.read_frames(a.input, a.frames)
    ref = PROBE.to_acescg(PROBE.read_frames(a.reference, a.frames), a.reference_space) if a.reference else None
    rep = {"tool": "h3_hdr_qc", "version": 1, "input": Path(a.input).name, **report(vals, a.input_space, ref,
                                                                                    a.white_nits)}
    Path(a.out).write_text(json.dumps(rep, indent=1, default=float), encoding="utf-8")
    print("\n".join(rep["checks"]))
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
