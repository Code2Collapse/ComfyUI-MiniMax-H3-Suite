"""H3 character-replacement benchmark: does the result keep the SOURCE performance? (C2C L8.25, owner A12 s1.6)

    # score one output against its plate (CPU; run with ComfyUI's Python)
    python tools/h3_replace_bench.py score --plate plate.mp4 --output out.mp4 --mask mask.mp4 [--performance drive.mp4]
                                           [--reference character.png] [--id T5] [--json T5.json]

    # run the matrix on a ComfyUI server, one entry at a time, then score each result
    python tools/h3_replace_bench.py run --manifest bench.json --server http://127.0.0.1:8188 --out results/

    # one table for all results
    python tools/h3_replace_bench.py report results/*.json [--md report.md]

bench.json: {"plate": "...", "mask": "...", "reference": "...", "performance": "... (optional)",
             "comfy_output_dir": "C:/ComfyUI/output",
             "runs": [{"id": "T1", "label": "Ref2VA baseline", "workflow": "t1_api.json", "output_prefix": "bench/T1"}]}
`workflow` is an API-format export ("Export (API)"). Seeds are whatever the workflow fixes - fix them.

Scores (definitions: C2C docs/research/l825_h3_performance_bench.md): face performance vs the source (mouth, eyes,
gaze, head pose: Pearson r + MAE per signal, mouth open/close event F1, frozen-expression ratio), background
preservation outside the mask, mask-edge leakage, excess flicker inside the mask, colour drift, identity (only when
insightface + a model are already installed), wall time and peak VRAM (nvidia-smi, whole GPU). Nothing is downloaded.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

import numpy as np

PACK = Path(__file__).resolve().parents[1]
if str(PACK) not in sys.path:
    sys.path.insert(0, str(PACK))

from mmx_utils import face_landmarks as FL  # noqa: E402
from mmx_utils import performance_metrics as PM  # noqa: E402

VIDEO_EXT = (".mp4", ".mov", ".mkv", ".webm", ".avi")


def read_video(path: str, max_frames: int = 0) -> np.ndarray:
    """[T,H,W,3] uint8 RGB from a video or an image (one frame)."""
    import cv2

    if Path(path).suffix.lower() not in VIDEO_EXT:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise SystemExit(f"cannot read {path}")
        return img[None, ..., ::-1].copy()
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, f = cap.read()
        if not ok or (max_frames and len(frames) >= max_frames):
            break
        frames.append(f[..., ::-1])
    cap.release()
    if not frames:
        raise SystemExit(f"cannot read {path}")
    return np.stack(frames)


def _fit(x: np.ndarray, h: int, w: int, nearest: bool) -> np.ndarray:
    import cv2

    if x.shape[1:3] == (h, w):
        return x
    interp = cv2.INTER_NEAREST if nearest else cv2.INTER_AREA
    return np.stack([cv2.resize(f, (w, h), interpolation=interp) for f in x])


def insightface_root(explicit: str | None = None) -> tuple[str | None, str]:
    """A root that ALREADY holds models/buffalo_l/*.onnx, or (None, why). FaceAnalysis downloads a missing model by
    itself, so it is only ever constructed on a root where the files exist."""
    roots = [explicit] if explicit else [os.environ.get("INSIGHTFACE_HOME"), os.path.join(os.path.expanduser("~"), ".insightface")]
    for root in [r for r in roots if r]:
        if glob.glob(os.path.join(root, "models", "buffalo_l", "*.onnx")):
            return root, ""
    return None, "insightface model buffalo_l not found under " + " / ".join(r for r in roots if r) + " (not downloaded)"


def identity_score(out_frames: np.ndarray, reference: str | None, root: str | None = None) -> dict:
    if not reference:
        return {"measured": False, "why": "no --reference"}
    try:
        import insightface  # noqa: F401
        from insightface.app import FaceAnalysis
    except Exception as e:  # noqa: BLE001
        return {"measured": False, "why": f"insightface not installed ({type(e).__name__})"}
    root, why = insightface_root(root)
    if root is None:
        return {"measured": False, "why": why}
    try:
        app = FaceAnalysis(name="buffalo_l", root=root, providers=["CPUExecutionProvider"],
                           allowed_modules=["detection", "recognition"])
        app.prepare(ctx_id=-1, det_size=(640, 640))
    except Exception as e:  # noqa: BLE001 - a missing model must not trigger a download path here
        return {"measured": False, "why": f"insightface model buffalo_l not available ({type(e).__name__})"}
    ref = read_video(reference)[0][..., ::-1]
    rf = app.get(ref)
    if not rf:
        return {"measured": False, "why": "no face in the reference"}
    r = rf[0].normed_embedding
    sims = []
    for f in out_frames[:: max(1, len(out_frames) // 24)]:
        faces = app.get(f[..., ::-1].copy())
        if faces:
            sims.append(float(np.dot(r, faces[0].normed_embedding)))
    if not sims:
        return {"measured": False, "why": "no face in the output"}
    return {"measured": True, "model": "insightface buffalo_l (ArcFace)", "cosine_mean": float(np.mean(sims)),
            "cosine_min": float(np.min(sims)), "frames": len(sims)}


def score(plate: str, output: str, mask: str, performance: str | None = None, reference: str | None = None,
          max_frames: int = 0, insightface_home: str | None = None) -> dict:
    t0 = time.perf_counter()
    P, O, M = read_video(plate, max_frames), read_video(output, max_frames), read_video(mask, max_frames)
    notes = []
    h, w = P.shape[1:3]
    if O.shape[1:3] != (h, w):
        notes.append(f"output {O.shape[2]}x{O.shape[1]} resized to the plate's {w}x{h}")
        O = _fit(O, h, w, nearest=False)
    M = _fit(M, h, w, nearest=True)[..., 0].astype(np.float32) / 255.0
    t = min(len(P), len(O), len(M))
    if len({len(P), len(O), len(M)}) > 1:
        notes.append(f"frame counts differ (plate {len(P)}, output {len(O)}, mask {len(M)}): scored the first {t}")
    P, O, M = P[:t], O[:t], M[:t]
    res: dict = {"plate": plate, "output": output, "mask": mask, "frames": t, "size": [w, h], "notes": notes}
    res["pixels"] = PM.pixel_scores(P, O, M)
    src_frames = P
    if performance:
        S = read_video(performance, max_frames)
        src_frames = _fit(S, h, w, nearest=False)[:t]
        res["performance_source"] = performance
    lm_src, why = FL.landmarks(src_frames)
    lm_out, why2 = FL.landmarks(O) if lm_src is not None else (None, why)
    if lm_src is None or lm_out is None:
        res["face"] = {"measured": False, "why": why or why2}
    elif not any(x is not None for x in lm_src) or not any(x is not None for x in lm_out):
        where = "the performance source" if not any(x is not None for x in lm_src) else "the output"
        res["face"] = {"measured": False, "why": f"no face found in {where} (mediapipe FaceMesh)"}
    else:
        res["face"] = {"measured": True, **PM.performance_scores(PM.face_signals(lm_src, w, h),
                                                                 PM.face_signals(lm_out, w, h))}
    res["identity"] = identity_score(O, reference, insightface_home)
    res["score_seconds"] = round(time.perf_counter() - t0, 2)
    return res


# ── running a matrix on a ComfyUI server ─────────────────────────────────────

class VramPoller:
    """Peak `nvidia-smi memory.used` (MiB, GPU 0) while running; the whole GPU, not one process."""

    def __init__(self, every: float = 0.5):
        self.every, self.peak, self.idle, self.ok = every, 0, None, bool(shutil.which("nvidia-smi"))
        self._stop = threading.Event()

    def _read(self) -> int | None:
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-i", "0"],
                                 capture_output=True, text=True, timeout=5)
            return int(out.stdout.strip().splitlines()[0])
        except Exception:  # noqa: BLE001
            return None

    def __enter__(self):
        if self.ok:
            self.idle = self._read()
            threading.Thread(target=self._loop, daemon=True).start()
        return self

    def _loop(self):
        while not self._stop.is_set():
            v = self._read()
            if v is not None:
                self.peak = max(self.peak, v)
            self._stop.wait(self.every)

    def __exit__(self, *a):
        self._stop.set()

    def result(self) -> dict:
        if not self.ok:
            return {"measured": False, "why": "nvidia-smi not found"}
        return {"measured": True, "peak_mib": self.peak, "idle_mib": self.idle,
                "peak_over_idle_mib": (self.peak - self.idle) if self.idle is not None else None,
                "scope": "whole GPU 0 (nvidia-smi), not model-only"}


def _http(url: str, body=None, timeout=60):
    req = urllib.request.Request(url, data=None if body is None else json.dumps(body).encode(),
                                 method="GET" if body is None else "POST", headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read())


def run_one(server: str, entry: dict, out_dir: str, timeout_s: float = 7200.0) -> dict:
    wf = json.loads(Path(entry["workflow"]).read_text(encoding="utf-8"))
    rec: dict = {"id": entry["id"], "label": entry.get("label", ""), "workflow": entry["workflow"]}
    started = time.time()
    with VramPoller() as vp:
        try:
            q = _http(f"{server}/prompt", {"prompt": wf, "client_id": "h3bench"})
        except Exception as e:  # noqa: BLE001
            rec["error"] = f"queue failed: {e}"
            return rec
        if q.get("node_errors"):
            rec["error"] = f"node errors: {json.dumps(q['node_errors'])[:800]}"
            return rec
        pid, hist = q["prompt_id"], {}
        while time.time() - started < timeout_s:
            hist = _http(f"{server}/history/{pid}").get(pid, {})
            if hist.get("status", {}).get("completed") is not None:
                break
            time.sleep(2)
    rec["vram"] = vp.result()
    msgs = hist.get("status", {}).get("messages", [])
    stamps = {m[0]: m[1].get("timestamp") for m in msgs if isinstance(m, list) and len(m) == 2}
    if stamps.get("execution_start") and (stamps.get("execution_success") or stamps.get("execution_error")):
        end = stamps.get("execution_success") or stamps.get("execution_error")
        rec["seconds"] = round((end - stamps["execution_start"]) / 1000.0, 2)
    rec["status"] = hist.get("status", {}).get("status_str")
    if rec["status"] != "success":
        rec["error"] = json.dumps([m for m in msgs if m[0] == "execution_error"])[:1500]
    return rec


def newest_output(comfy_output_dir: str, prefix: str, since: float) -> str | None:
    pat = os.path.join(comfy_output_dir, prefix + "*")
    files = [f for f in glob.glob(pat) if Path(f).suffix.lower() in VIDEO_EXT and os.path.getmtime(f) >= since - 1]
    return max(files, key=os.path.getmtime) if files else None


def cmd_run(a) -> int:
    man = json.loads(Path(a.manifest).read_text(encoding="utf-8"))
    os.makedirs(a.out, exist_ok=True)
    for entry in man["runs"]:
        print(f"[{entry['id']}] {entry.get('label', '')}", flush=True)
        t0 = time.time()
        rec = run_one(a.server.rstrip("/"), entry, a.out)
        out = newest_output(man.get("comfy_output_dir", ""), entry.get("output_prefix", ""), t0) if "error" not in rec else None
        if out:
            rec["output"] = out
            rec["scores"] = score(man["plate"], out, man["mask"], man.get("performance"), man.get("reference"))
        elif "error" not in rec:
            rec["error"] = f"no video under {entry.get('output_prefix')} in {man.get('comfy_output_dir')}"
        Path(a.out, f"{entry['id']}.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
        print("   ", rec.get("status"), rec.get("seconds"), "s", rec.get("vram", {}).get("peak_over_idle_mib"), "MiB",
              rec.get("error", "")[:200], flush=True)
    return 0


def _f(x, nd=3):
    if x is None or (isinstance(x, float) and (math.isnan(x))):
        return "-"
    if isinstance(x, float) and math.isinf(x):
        return "inf"
    return f"{x:.{nd}f}" if isinstance(x, float) else str(x)


def report_rows(recs: list[dict]) -> str:
    head = ("| run | time s | VRAM MiB | mouth r | mouth F1 | eyes r | gaze r | yaw r | frozen | bg PSNR | edge leak | "
            "excess flicker | colour excess | identity |")
    lines = [head, "|" + "|".join(["---"] * 14) + "|"]
    for r in recs:
        s = r.get("scores", r if "pixels" in r else {})
        face = s.get("face", {})
        sig = face.get("signals", {}) if face.get("measured") else {}
        px = s.get("pixels", {})

        def rr(k):
            return _f((sig.get(k) or {}).get("pearson_r"))
        eyes = [v for v in ((sig.get("eye_open_l") or {}).get("pearson_r"), (sig.get("eye_open_r") or {}).get("pearson_r"))
                if v is not None]
        lines.append("| " + " | ".join([
            str(r.get("id", Path(s.get("output", "?")).stem)), _f(r.get("seconds"), 1),
            _f((r.get("vram") or {}).get("peak_over_idle_mib")), rr("mouth_open"),
            _f((face.get("mouth_events") or {}).get("f1_tol1")), _f(float(np.mean(eyes)) if eyes else None),
            rr("gaze_x"), rr("yaw"), _f(face.get("frozen")),
            _f((px.get("background") or {}).get("psnr_db"), 1), _f((px.get("edge_leakage") or {}).get("band_2_12px")),
            _f((px.get("flicker") or {}).get("excess"), 4), _f((px.get("colour") or {}).get("inside_trajectory_excess"), 4),
            _f((s.get("identity") or {}).get("cosine_mean")) if (s.get("identity") or {}).get("measured") else "-",
        ]) + " |")
    return "\n".join(lines)


def cmd_report(a) -> int:
    recs = []
    for pat in a.results:
        for f in sorted(glob.glob(pat)):
            recs.append(json.loads(Path(f).read_text(encoding="utf-8")))
    table = report_rows(recs)
    print(table)
    if a.md:
        Path(a.md).write_text("# H3 replacement benchmark\n\n" + table + "\n\nface = vs the performance source; "
                              "identity only where insightface is installed; VRAM = whole GPU over idle.\n",
                              encoding="utf-8")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score")
    s.add_argument("--plate", required=True)
    s.add_argument("--output", required=True)
    s.add_argument("--mask", required=True)
    s.add_argument("--performance")
    s.add_argument("--reference")
    s.add_argument("--id", default="")
    s.add_argument("--max-frames", type=int, default=0)
    s.add_argument("--insightface-home", help="a folder that already holds models/buffalo_l (never downloaded)")
    s.add_argument("--json")
    r = sub.add_parser("run")
    r.add_argument("--manifest", required=True)
    r.add_argument("--server", default="http://127.0.0.1:8188")
    r.add_argument("--out", default="h3_bench_results")
    p = sub.add_parser("report")
    p.add_argument("results", nargs="+")
    p.add_argument("--md")
    a = ap.parse_args()
    if a.cmd == "score":
        res = score(a.plate, a.output, a.mask, a.performance, a.reference, a.max_frames, a.insightface_home)
        if a.id:
            res = {"id": a.id, "scores": res}
        text = json.dumps(res, indent=1)
        if a.json:
            Path(a.json).write_text(text, encoding="utf-8")
        print(report_rows([res]))
        return 0
    return cmd_run(a) if a.cmd == "run" else cmd_report(a)


if __name__ == "__main__":
    raise SystemExit(main())
