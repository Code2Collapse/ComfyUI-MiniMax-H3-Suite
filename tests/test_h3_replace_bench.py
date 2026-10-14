"""tools/h3_replace_bench.py - the H3 replacement benchmark harness (C2C L8.25). CPU-only, no network beyond a
stdlib fake ComfyUI server on 127.0.0.1."""

from __future__ import annotations

import importlib.util
import json
import math
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import numpy as np
import pytest

PACK = Path(__file__).resolve().parents[1]
if str(PACK) not in sys.path:
    sys.path.insert(0, str(PACK))


def _bench():
    spec = importlib.util.spec_from_file_location("h3_replace_bench", PACK / "tools" / "h3_replace_bench.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _clips(t=5, h=48, w=48):
    rng = np.random.default_rng(1)
    plate = (rng.random((t, h, w, 3)) * 255).astype(np.uint8)
    mask = np.zeros((t, h, w, 3), np.uint8)
    mask[:, 16:32, 16:32] = 255
    out = plate.copy()
    out[:, 16:32, 16:32] = 128
    return plate, out, mask


def test_score_reports_pixels_and_says_why_face_is_not_measured(monkeypatch):
    # INVARIANT: no face in the clip -> face scores "not measured" with a reason; pixel scores still computed.
    B = _bench()
    plate, out, mask = _clips()
    vids = {"p": plate, "o": out, "m": mask}
    monkeypatch.setattr(B, "read_video", lambda path, max_frames=0: vids[path])
    res = B.score("p", "o", "m")
    assert math.isinf(res["pixels"]["background"]["psnr_db"])
    assert res["face"]["measured"] is False and res["face"]["why"]
    assert res["identity"] == {"measured": False, "why": "no --reference"}


def test_frame_count_and_size_mismatch_are_reported_not_hidden(monkeypatch):
    # INVARIANT: differing lengths -> common prefix + a note; a differently sized output is resized and noted.
    B = _bench()
    plate, out, mask = _clips(t=6)
    big = np.repeat(np.repeat(out[:4], 2, axis=1), 2, axis=2)
    vids = {"p": plate, "o": big, "m": mask}
    monkeypatch.setattr(B, "read_video", lambda path, max_frames=0: vids[path])
    res = B.score("p", "o", "m")
    assert res["frames"] == 4
    assert any("resized" in n for n in res["notes"]) and any("frame counts differ" in n for n in res["notes"])


def test_identity_never_constructs_insightface_without_a_local_model(tmp_path, monkeypatch):
    # INVARIANT: FaceAnalysis would DOWNLOAD a missing model, so without models/buffalo_l/*.onnx on disk it is not
    # called at all - the score says where it looked.
    B = _bench()
    root, why = B.insightface_root(str(tmp_path))
    assert root is None and "not downloaded" in why
    (tmp_path / "models" / "buffalo_l").mkdir(parents=True)
    (tmp_path / "models" / "buffalo_l" / "w600k_r50.onnx").write_bytes(b"x")
    assert B.insightface_root(str(tmp_path))[0] == str(tmp_path)


class _FakeComfy:
    def __init__(self, out_dir: Path):
        self.out_dir, self.queued = out_dir, []
        fake = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _json(self, obj):
                data = json.dumps(obj).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                fake.queued.append(body)
                (fake.out_dir / "bench" / "T1_00001.mp4").write_bytes(b"video")     # the "render"
                self._json({"prompt_id": "abc", "node_errors": {}})

            def do_GET(self):
                now = int(time.time() * 1000)
                self._json({"abc": {"status": {"completed": True, "status_str": "success", "messages": [
                    ["execution_start", {"timestamp": now - 4200}], ["execution_success", {"timestamp": now}]]}}})

        self.server = HTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.server.server_port}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()


def test_run_queues_each_entry_times_it_and_scores_its_newest_output(tmp_path, monkeypatch):
    # INVARIANT: one entry -> queued as given, wall time from ComfyUI's own timestamps, the newest video under the
    # prefix is scored, one JSON per entry; the report table has a row for it.
    B = _bench()
    (tmp_path / "bench").mkdir()
    fake = _FakeComfy(tmp_path)
    wf = tmp_path / "t1.json"
    wf.write_text(json.dumps({"1": {"class_type": "X", "inputs": {}}}), encoding="utf-8")
    man = tmp_path / "bench.json"
    man.write_text(json.dumps({"plate": "p", "mask": "m", "comfy_output_dir": str(tmp_path),
                               "runs": [{"id": "T1", "label": "Ref2VA baseline", "workflow": str(wf),
                                         "output_prefix": "bench/T1"}]}), encoding="utf-8")
    scored = []
    monkeypatch.setattr(B, "score", lambda plate, out, mask, perf=None, ref=None: scored.append(out) or {"pixels": {}})

    class A:
        manifest, server, out = str(man), fake.url, str(tmp_path / "results")

    assert B.cmd_run(A) == 0
    rec = json.loads((tmp_path / "results" / "T1.json").read_text(encoding="utf-8"))
    assert fake.queued[0]["prompt"] == {"1": {"class_type": "X", "inputs": {}}}
    assert rec["status"] == "success" and rec["seconds"] == pytest.approx(4.2, abs=0.05)
    assert scored and scored[0].endswith("T1_00001.mp4")
    assert "| T1 |" in B.report_rows([rec])
    fake.server.shutdown()
