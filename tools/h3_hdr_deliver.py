"""Deliver H3 HDR output (ledger L9.212, A26 3.7): decoded frames -> log interpretation -> scene-linear ACEScg ->
exposure -> optional alpha composite -> HDR QC -> EXR / DPX / HLG with metadata, every file read back and checked,
and <name>.delivery.json (QC + outputs) written beside them.

    # H3 + a log LoRA, decoded frames in ACEScct, to EXR + LogC3 DPX + HLG:
    python tools/h3_hdr_deliver.py --input gen.####.png --input-space ACEScct --out D:/deliver/shot010 \\
        --name shot010 --exr --dpx "ARRI LogC3" --hlg --fps 24
    # with the aligned HDR plate as reference (alignment + PU21 + per-stop + colour drift + flicker delta):
    python tools/h3_hdr_deliver.py --input out.####.exr --reference plate.####.exr --out D:/deliver/a --name a --exr

The verdict is HDR PASS only when highlight information above diffuse white is measured and no check fails; an EXR
or 16-bit file alone never passes. Exit code 0 = written and verified (whatever the verdict), 2 = a read-back failed.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
for p in (str(HERE.parent), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np  # noqa: E402

import h3_vae_log_probe as PROBE  # noqa: E402  (same readers)
from mmx_utils import hdr_deliver as DLV  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="frames: EXR / DPX / TIFF / PNG sequence or a movie")
    ap.add_argument("--input-space", default="ACEScg", help="ACEScg | the log code of the frames | linear:<gamut>")
    ap.add_argument("--alpha", help="alpha / matte sequence (single channel or the first channel is used)")
    ap.add_argument("--premultiplied", action="store_true")
    ap.add_argument("--background", help="ACEScg plate to composite the alpha over (then no alpha is written)")
    ap.add_argument("--exposure", type=float, default=0.0, help="exposure change in stops before delivery")
    ap.add_argument("--reference", help="aligned HDR reference (ACEScg) for the comparison")
    ap.add_argument("--reference-space", default="ACEScg")
    ap.add_argument("--frames", type=int, default=1000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--exr", action="store_true")
    ap.add_argument("--dpx", metavar="CURVE", help="log curve for DPX, e.g. 'ARRI LogC3' (any NukeMax curve name)")
    ap.add_argument("--dpx-bits", type=int, default=16, choices=(10, 16))
    ap.add_argument("--hlg", action="store_true")
    ap.add_argument("--hlg-crf", type=int, default=None, help="lossy HLG at this CRF (default: lossless)")
    ap.add_argument("--fps", type=float, default=24.0)
    a = ap.parse_args(argv)
    if not (a.exr or a.dpx or a.hlg):
        ap.error("choose at least one of --exr, --dpx, --hlg")
    vals = PROBE.read_frames(a.input, a.frames)
    alpha = PROBE.read_frames(a.alpha, a.frames)[..., 0] if a.alpha else None
    bg = PROBE.read_frames(a.background, a.frames) if a.background else None
    ref = PROBE.to_acescg(PROBE.read_frames(a.reference, a.frames), a.reference_space) if a.reference else None
    rep = DLV.deliver(vals[..., :3].astype(np.float32), a.input_space, a.out, a.name, exr=a.exr, dpx_code=a.dpx,
                      dpx_bits=a.dpx_bits, hlg=a.hlg, fps=a.fps, exposure_ev=a.exposure, alpha=alpha,
                      background=bg, premultiplied=a.premultiplied, reference=ref,
                      hlg_lossless=a.hlg_crf is None)
    v = rep["qc"]["verdict"]
    print(f"HDR {v['status']}")
    for line in v["fail"] + v["check"] + v["info"]:
        print("  -", line)
    for kind, o in rep["outputs"].items():
        print(f"{kind}: {'verified' if o['readback']['ok'] else 'READ-BACK FAILED: ' + '; '.join(o['readback']['problems'])}")
    print(f"report: {Path(a.out) / (a.name + '.delivery.json')}")
    return 0 if rep["ok"] else 2


if __name__ == "__main__":
    sys.exit(main())
