"""H3 Shot Boundary Analyzer (owner A20 Phase 1 slice 3, ledger L9.74; Revision 2 Part 8.5).

Finds the shots of a clip: hard cuts (and, if asked, dips to black / white), each with a confidence, a minimum shot
length, manual add / remove overrides, and a contact sheet of every shot's first frame. A hard cut starts a new
visual context: the manifest says latent context does not cross it (Part 8.5) - a planner reads that.

The detectors are H3 Seam Correction's own (mmx_utils/seam_correction.py, ported from
supElement/ComfyUI_MinimaxH3_AutoContext, Apache-2.0): PySceneDetect's ContentDetector when installed, else the
dependency-free colour-histogram correlation, each cut refined to the strongest frame step. Original node, GPL-3.0.
Design: docs/evidence/A20/phase1/DESIGN_slice3_shot_boundary.md in the workspace.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

from comfy_api.latest import io

_PKG = Path(__file__).resolve().parents[1]
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from mmx_utils import seam_correction as E  # noqa: E402

MANIFEST = "h3.shot_manifest"
METHODS = ("auto", "histogram", "pyscenedetect")


def _ints(text: str) -> list[int]:
    out = []
    for part in str(text or "").replace(";", ",").split(","):
        part = part.strip()
        if part:
            try:
                out.append(int(part))
            except ValueError:
                raise ValueError(f"H3 Shot Boundary Analyzer: '{part}' is not a frame number (use e.g. 12, 40).") from None
    return out


def _histogram_cuts(images, sensitivity, min_gap, device):
    thr = 0.30 + 0.60 * float(sensitivity)
    hists = E._frame_histograms(images, device)
    corr = [1.0] * int(images.shape[0])
    for t in range(1, int(images.shape[0])):
        corr[t] = float(E._hist_pearson(hists[t - 1], hists[t]))
    cuts = E._detect_shot_cuts_histogram(images, thr, device, min_gap=min_gap)
    return cuts, {c: round(max(0.0, min(1.0, 1.0 - corr[c])), 4) for c in cuts}


def _fades(images, min_gap):
    """Dips to black or white: >= 3 frames of mean luma falling monotonically to the extreme and rising back."""
    luma = (images[..., :3] * images.new_tensor(E._LUMA_W)).sum(-1).mean(dim=(1, 2)).tolist()
    n, out, t = len(luma), [], 1
    while t < n - 1:
        for extreme, near in ((min, lambda v: v < 0.08), (max, lambda v: v > 0.92)):
            if near(luma[t]):
                a = t
                while a > 0 and ((luma[a - 1] > luma[a]) if extreme is min else (luma[a - 1] < luma[a])):
                    a -= 1
                b = t
                while b < n - 1 and ((luma[b + 1] > luma[b]) if extreme is min else (luma[b + 1] < luma[b])):
                    b += 1
                if b - a >= 2 and (not out or a - out[-1][1] >= min_gap):
                    out.append((a, b, "black" if extreme is min else "white"))
                    t = b
                break
        t += 1
    return out


class MiniMaxH3_ShotBoundaries(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MiniMaxH3_ShotBoundaries",
            display_name="H3 Shot Boundary Analyzer",
            category="MiniMax H3/Media",
            description=(
                "Finds the shots of a clip: hard cuts with a confidence (and dips to black / white if asked), a minimum "
                "shot length, manual add / remove, and a contact sheet of every shot's first frame. Latent context should "
                "not cross a cut: the manifest says so for later stages."
            ),
            inputs=[
                io.Image.Input("images", tooltip="The frames to analyse (a window or a short clip)."),
                io.String.Input("window_json", optional=True, force_input=True,
                                tooltip="From H3 Frame Window Reader: cuts are also reported as source frames."),
                io.Combo.Input("method", options=list(METHODS), default="auto",
                               tooltip="auto = PySceneDetect when installed, else the colour-histogram detector."),
                io.Float.Input("sensitivity", default=0.5, min=0.0, max=1.0, step=0.05,
                               tooltip="Higher finds more cuts. Histogram: a cut where the colour correlation is below "
                                       "0.30 + 0.60 x sensitivity. PySceneDetect: content threshold 45 - 35 x sensitivity."),
                io.Int.Input("min_shot_frames", default=8, min=1, max=1000, tooltip="Shorter shots merge into the one before."),
                io.Boolean.Input("fades", default=False, tooltip="Also report dips to black or white as transitions."),
                io.String.Input("add_cuts", default="", tooltip="Rows to cut at anyway, e.g. 12, 40."),
                io.String.Input("remove_cuts", default="", tooltip="Rows NOT to cut at, e.g. 18."),
            ],
            outputs=[
                io.String.Output("shots_json", tooltip="Contract h3.shot_manifest v1: one entry per shot."),
                io.String.Output("cut_rows", tooltip="The rows where a new shot starts, comma separated."),
                io.Int.Output("shot_count"),
                io.Image.Output("contact_sheet", tooltip="The first frame of every shot, side by side."),
            ],
        )

    @classmethod
    def execute(cls, images, window_json=None, method="auto", sensitivity=0.5, min_shot_frames=8, fades=False,
                add_cuts="", remove_cuts="") -> io.NodeOutput:
        if images.ndim == 3:
            images = images.unsqueeze(0)
        n = int(images.shape[0])
        device = torch.device("cpu")
        min_gap = max(1, int(min_shot_frames))
        used, conf, notes = method, {}, []
        cuts: list[int] = []
        if n >= 2:
            if method in ("auto", "pyscenedetect"):
                got = E._detect_shot_cuts(images, 45.0 - 35.0 * float(sensitivity), device, min_gap=min_gap)
                if got is None:
                    if method == "pyscenedetect":
                        raise ValueError("H3 Shot Boundary Analyzer: method 'pyscenedetect' needs the scenedetect and "
                                         "opencv-python packages (pip install scenedetect opencv-python), or use "
                                         "'histogram'.")
                    notes.append("PySceneDetect not installed: used the colour-histogram detector")
                    used = "histogram"
                else:
                    used, cuts = "pyscenedetect", list(got)
                    conf = {c: 1.0 for c in cuts}
            if used == "histogram":
                cuts, conf = _histogram_cuts(images, sensitivity, min_gap, device)
        kinds = {c: "hard" for c in cuts}
        transitions = []
        if fades and n >= 3:
            for a, b, tone in _fades(images, min_gap):
                transitions.append({"rows": [a, b], "kind": f"dip to {tone}"})
                mid = (a + b + 1) // 2
                cuts = [c for c in cuts if not (a <= c <= b + 1)]      # the detector's cuts inside the dip ARE the fade
                cuts.append(mid)
                kinds[mid], conf[mid] = "fade", 1.0
        merged = []                                       # min shot length applies to DETECTED cuts only
        for c in sorted(set(cuts)):
            if c - (merged[-1] if merged else 0) < min_gap:
                continue
            merged.append(c)
        ignored = []                                      # manual cuts are overrides: never merged away, never merging
        for c in _ints(add_cuts):
            if 0 < c < n:
                kinds[c], conf[c] = "manual", 1.0
                merged.append(c)
            else:
                ignored.append(c)
        removed = set(_ints(remove_cuts))
        cuts = sorted(c for c in set(merged) if c not in removed and 0 < c < n)
        src = None
        if window_json:
            win = json.loads(window_json)
            src = win.get("source_frames")
        bounds = [0] + cuts + [n]
        shots = []
        for i in range(len(bounds) - 1):
            a, b = bounds[i], bounds[i + 1]
            shots.append({"shot": i, "rows": [a, b],
                          "source_frames": [src[a], src[b - 1] + 1] if src and len(src) == n else None,
                          "cut_in": "start" if i == 0 else kinds.get(a, "hard"),
                          "confidence": 1.0 if i == 0 else conf.get(a, 1.0)})
        manifest = {"c2c_contract": MANIFEST, "c2c_contract_version": "1.0", "frames": n, "method_used": used,
                    "sensitivity": float(sensitivity), "min_shot_frames": min_gap, "shots": shots,
                    "transitions": transitions, "latent_context_crosses_cuts": False, "ignored": ignored,
                    "notes": notes}
        sheet = torch.cat([images[s["rows"][0]:s["rows"][0] + 1] for s in shots], dim=2) if n else images
        return io.NodeOutput(json.dumps(manifest), ", ".join(str(c) for c in cuts), len(shots), sheet)


NODE_CLASS_MAPPINGS = {"MiniMaxH3_ShotBoundaries": MiniMaxH3_ShotBoundaries}
