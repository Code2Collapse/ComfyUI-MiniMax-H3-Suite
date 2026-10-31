"""H3 Review Package (owner A21 Revision 3 Part 9.8; ledger L9.74 slice 9).

Builds a review folder for a shot version, inside ComfyUI's output folder:

* ``review.html`` - self-contained (images are files next to it; no network): an A/B wipe per selected frame, the
  difference heatmaps, QC markers and annotations. Every text from the inputs is HTML-escaped.
* a contact sheet (also the node's IMAGE output): original | generated (| difference) per selected frame, with the
  frame number / timecode burned in when asked;
* an A | B side-by-side movie of the whole compared range (streamed, one chunk in memory);
* a JSON manifest (``h3.review_package`` v1) listing every file with its sha256.

Original node, GPL-3.0. Design: docs/evidence/A21/phase1/DESIGN_slices7_10.md in the workspace.
"""

from __future__ import annotations

import hashlib
import html
import json
import os
import sys
import time
from fractions import Fraction
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from comfy_api.latest import io

_PKG = Path(__file__).resolve().parents[1]
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from mmx_vendor.c2c_core.video.writer import StreamWriter  # noqa: E402

from .media_window import C2C_VIDEO  # noqa: E402
from .segment_io import job_folder  # noqa: E402
from .sequence_diff import _heat, _Source, compare_chunk  # noqa: E402

WHO = "H3 Review Package"
CONTRACT = "h3.review_package"
THUMB_W = 256
AB_SIDE_MAX = 960


def _sha256(path: Path) -> str:
    d = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            d.update(block)
    return d.hexdigest()


def _png(path: Path, rgb: torch.Tensor) -> None:
    from PIL import Image

    arr = (rgb.clamp(0, 1).numpy() * 255.0 + 0.5).astype(np.uint8)
    Image.fromarray(arr).save(path)


def _label(rgb: torch.Tensor, text: str) -> torch.Tensor:
    """Burn ``text`` into the top-left corner (white on a dark box)."""
    from PIL import Image, ImageDraw

    arr = (rgb.clamp(0, 1).numpy() * 255.0 + 0.5).astype(np.uint8)
    im = Image.fromarray(arr)
    d = ImageDraw.Draw(im)
    box = d.textbbox((4, 2), text)
    d.rectangle((box[0] - 3, box[1] - 2, box[2] + 3, box[3] + 2), fill=(0, 0, 0))
    d.text((4, 2), text, fill=(255, 255, 255))
    return torch.from_numpy(np.asarray(im).astype(np.float32) / 255.0)


def _fit(x: torch.Tensor, w: int, h: int) -> torch.Tensor:
    """[N, H, W, 3] -> [N, h, w, 3]"""
    if tuple(x.shape[1:3]) == (h, w):
        return x
    return F.interpolate(x.movedim(-1, 1), size=(h, w), mode="area" if w < x.shape[2] else "bilinear",
                         **({} if w < x.shape[2] else {"align_corners": False})).movedim(1, -1)


def _tc(frame: int, fps: Fraction) -> str:
    f = int(frame)
    r = max(1, round(float(fps)))
    s, ff = divmod(f, r)
    m, ss = divmod(s, 60)
    hh, mm = divmod(m, 60)
    return f"{hh:02d}:{mm:02d}:{ss:02d}:{ff:02d}"


def _json_or_none(raw: str):
    s = str(raw or "").strip()
    if not s:
        return None
    try:
        return json.loads(s)
    except ValueError as exc:
        raise ValueError(f"{WHO}: an input is not valid JSON ({exc})") from exc


def qc_summary(qc) -> dict:
    """{verdict: pass | fail | none, checks: [{name, ok, frames}], failed_frames: [...]} from several QC shapes."""
    if qc is None:
        return {"verdict": "none", "checks": [], "failed_frames": []}
    checks = []
    raw = qc.get("checks") if isinstance(qc, dict) else qc if isinstance(qc, list) else None
    for c in raw or []:
        if isinstance(c, dict):
            ok = bool(c.get("ok", c.get("passed", str(c.get("verdict", "")).lower() == "pass")))
            checks.append({"name": str(c.get("name", "check")), "ok": ok,
                           "frames": [int(f) for f in c.get("frames", []) if isinstance(f, (int, float))]})
    if isinstance(qc, dict) and any(k in qc for k in ("passed", "verdict", "status", "ok")):
        v = qc.get("passed", qc.get("ok"))
        if v is None:
            v = str(qc.get("verdict", qc.get("status", ""))).lower() in ("pass", "passed", "ok")
        verdict = "pass" if v else "fail"
    else:
        verdict = "pass" if checks and all(c["ok"] for c in checks) else ("fail" if checks else "none")
    failed = sorted({f for c in checks if not c["ok"] for f in c["frames"]})
    return {"verdict": verdict, "checks": checks, "failed_frames": failed}


def _positions(spec: str, every: int, n: int) -> list[int]:
    s = str(spec or "").strip()
    if s:
        out = []
        for part in s.replace(";", ",").split(","):
            part = part.strip()
            if part:
                try:
                    v = int(part)
                except ValueError as exc:
                    raise ValueError(f"{WHO}: frames {spec!r} - write positions like 0, 12, 40") from exc
                if 0 <= v < n and v not in out:
                    out.append(v)
        if not out:
            raise ValueError(f"{WHO}: none of the frames {spec!r} is inside 0..{n - 1}")
        return out
    return list(range(0, n, max(1, int(every))))


_PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport"
content="width=device-width,initial-scale=1"><title>{title}</title><style>
:root {{ color-scheme: dark; --bg:#141518; --fg:#e8e6e1; --dim:#9a978f; --bad:#e0564f; --ok:#5fb37a; }}
body {{ margin:0; padding:24px 16px; background:var(--bg); color:var(--fg);
  font:14px/1.5 system-ui,-apple-system,Segoe UI,sans-serif; }}
main {{ max-width:1100px; margin:0 auto; display:flex; flex-direction:column; gap:24px; }}
h1 {{ font-size:20px; margin:0; }} h2 {{ font-size:15px; margin:0 0 8px; }}
.meta {{ color:var(--dim); font-variant-numeric:tabular-nums; }}
.pass {{ color:var(--ok); }} .fail {{ color:var(--bad); }}
.frame {{ display:flex; flex-direction:column; gap:6px; }}
.frame.bad h2 {{ color:var(--bad); }}
.wipe {{ position:relative; width:100%; max-width:960px; }}
.wipe img {{ display:block; width:100%; height:auto; }}
.wipe .top {{ position:absolute; inset:0; overflow:hidden; clip-path:inset(0 50% 0 0); }}
.wipe input {{ width:100%; max-width:960px; }}
.diff img {{ width:100%; max-width:960px; display:block; }}
ul {{ margin:0; padding-left:18px; }}
</style></head><body><main>
<header><h1>{title}</h1><p class="meta">{meta}</p></header>
{qc}
{notes}
{frames}
</main><script>
document.querySelectorAll('.wipe').forEach(function (w) {{
  var top = w.querySelector('.top'); var r = w.parentNode.querySelector('input');
  r.addEventListener('input', function () {{ top.style.clipPath = 'inset(0 ' + (100 - r.value) + '% 0 0)'; }});
}});
</script></body></html>
"""


class MiniMaxH3_ReviewPackage(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MiniMaxH3_ReviewPackage",
            display_name="H3 Review Package",
            category="MiniMax H3/Media",
            description=(
                "Writes a review folder: review.html (A/B wipe per frame, difference heatmaps, QC markers, notes), a "
                "contact sheet, an A | B movie and a JSON manifest with checksums. Self-contained, no network."
            ),
            inputs=[
                C2C_VIDEO.Input("original", optional=True),
                C2C_VIDEO.Input("generated", optional=True),
                io.Image.Input("original_images", optional=True),
                io.Image.Input("generated_images", optional=True),
                io.String.Input("version_manifest", default="", optional=True, tooltip="JSON: which versions these are."),
                io.String.Input("qc", default="", optional=True,
                                tooltip='JSON: {"passed": true} / {"verdict": "fail", "checks": [{"name", "ok", "frames"}]}'),
                io.String.Input("annotations", default="", optional=True,
                                tooltip='JSON list: [{"frame": 12, "text": "fix the cuff"}] (frame = position).'),
                io.String.Input("job_dir", default="h3_jobs/review"),
                io.String.Input("title", default="H3 review"),
                io.Int.Input("contact_every", default=12, min=1, max=10000, tooltip="A contact cell every N frames."),
                io.String.Input("frames", default="", tooltip="Positions to show instead, e.g. 0, 12, 40."),
                io.Combo.Input("difference", options=["ssim", "pixel", "none"], default="ssim"),
                io.Boolean.Input("timecode", default=True, tooltip="Burn the frame position and timecode into the cells."),
                io.Boolean.Input("include_qc", default=True),
                io.Boolean.Input("include_annotations", default=True),
                io.Boolean.Input("ab_movie", default=True, tooltip="Write the A | B side-by-side movie (h264)."),
                io.Float.Input("fps", default=24.0, min=1.0, max=240.0, tooltip="For IMAGE inputs and timecodes."),
            ],
            outputs=[
                io.String.Output("html_path"),
                io.Image.Output("contact_sheet"),
                io.String.Output("ab_path"),
                io.String.Output("manifest", tooltip="h3.review_package v1: files with sha256, counts, QC verdict."),
            ],
        )

    @classmethod
    def execute(cls, original=None, generated=None, original_images=None, generated_images=None, version_manifest="",
                qc="", annotations="", job_dir="h3_jobs/review", title="H3 review", contact_every=12, frames="",
                difference="ssim", timecode=True, include_qc=True, include_annotations=True, ab_movie=True,
                fps=24.0) -> io.NodeOutput:
        A = _Source("original", original, original_images)
        B = _Source("generated", generated, generated_images)
        n = min(A.count, B.count)
        if n == 0:
            raise ValueError(f"{WHO}: nothing to review - an input has no frames.")
        rate = Fraction(original.fps) if original is not None else Fraction(fps).limit_denominator(1001000)
        versions = _json_or_none(version_manifest)
        qcs = qc_summary(_json_or_none(qc))
        notes = _json_or_none(annotations) or []
        if not isinstance(notes, list):
            raise ValueError(f"{WHO}: annotations must be a JSON list of {{frame, text}}.")
        stamp = time.strftime("%Y%m%d_%H%M%S")
        folder = job_folder(job_dir, create=True, who=WHO) / f"review_{stamp}"
        k = 1
        while folder.exists():
            k += 1
            folder = folder.with_name(f"review_{stamp}_{k}")
        folder.mkdir(parents=True)
        pos = _positions(frames, contact_every, n)
        cells, sections, files = [], [], []
        for p in pos:
            a, b = A.read([p]), B.read([p])
            b = _fit(b, a.shape[2], a.shape[1])
            label = f"#{p}  frame {A.numbers[p]}  {_tc(A.numbers[p] if A.numbers[p] >= 0 else p, rate)}"
            ai, bi = (_label(a[0], label), _label(b[0], label)) if timecode else (a[0], b[0])
            row = [ai, bi]
            names = {"a": f"a_{p:06d}.png", "b": f"b_{p:06d}.png"}
            _png(folder / names["a"], a[0])
            _png(folder / names["b"], b[0])
            if difference != "none":
                score, heat = compare_chunk(a, b, difference, None)
                hm = _heat(heat[0])
                names["d"] = f"d_{p:06d}.png"
                _png(folder / names["d"], hm)
                row.append(_label(hm, f"{difference} {float(score[0]):.4f}") if timecode else hm)
            files += list(names.values())
            th = max(8, round(a.shape[1] * THUMB_W / a.shape[2]))
            cells.append(torch.cat([_fit(r[None], THUMB_W, th)[0] for r in row], dim=1))
            bad = p in qcs["failed_frames"]
            mine = [x for x in notes if isinstance(x, dict) and int(x.get("frame", -1)) == p] if include_annotations else []
            sections.append(
                f'<section class="frame{" bad" if bad else ""}"><h2>{html.escape(label)}'
                f'{" - QC failed here" if bad and include_qc else ""}</h2>'
                f'<div><div class="wipe"><img src="{names["b"]}" alt="generated"><div class="top">'
                f'<img src="{names["a"]}" alt="original"></div></div>'
                f'<input type="range" min="0" max="100" value="50" aria-label="wipe original / generated"></div>'
                + (f'<div class="diff"><img src="{names["d"]}" alt="difference"></div>' if "d" in names else "")
                + ("<ul>" + "".join(f"<li>{html.escape(str(x.get('text', '')))}</li>" for x in mine) + "</ul>"
                   if mine else "") + "</section>")
        width = max(c.shape[1] for c in cells)
        sheet = torch.cat([F.pad(c, (0, 0, 0, width - c.shape[1])) for c in cells], dim=0)[None]
        _png(folder / "contact_sheet.png", sheet[0])
        files.append("contact_sheet.png")
        ab_path = ""
        if ab_movie:
            side_w = min(AB_SIDE_MAX, A.read([0]).shape[2])
            side_w -= side_w % 2
            a0 = A.read([0])
            side_h = max(2, round(a0.shape[1] * side_w / a0.shape[2]))
            side_h -= side_h % 2
            ab_path = str(folder / "ab.mp4")
            with StreamWriter(ab_path, rate, side_w * 2, side_h, codec="h264", quality=85) as w:
                for s in range(0, n, 8):
                    idx = list(range(s, min(n, s + 8)))
                    w.write(torch.cat([_fit(A.read(idx), side_w, side_h), _fit(B.read(idx), side_w, side_h)], dim=2))
            files.append("ab.mp4")
        qc_html = ""
        if include_qc and qcs["verdict"] != "none":
            rows = "".join(f'<li class="{"pass" if c["ok"] else "fail"}">{html.escape(c["name"])}: '
                           f'{"pass" if c["ok"] else "FAIL"}{" - frames " + ", ".join(map(str, c["frames"])) if c["frames"] else ""}'
                           f"</li>" for c in qcs["checks"])
            qc_html = (f'<section><h2>QC: <span class="{qcs["verdict"]}">{qcs["verdict"].upper()}</span></h2>'
                       f"<ul>{rows}</ul></section>")
        notes_html = ""
        if include_annotations and notes:
            notes_html = "<section><h2>Notes</h2><ul>" + "".join(
                f"<li>frame {html.escape(str(x.get('frame', '?')))}: {html.escape(str(x.get('text', '')))}</li>"
                for x in notes if isinstance(x, dict)) + "</ul></section>"
        meta = (f"{n} frames compared · {len(pos)} shown · {html.escape(str(rate))} fps · "
                f"difference {html.escape(difference)} · {html.escape(time.strftime('%Y-%m-%d %H:%M'))}"
                + (f" · versions {html.escape(json.dumps(versions))}" if versions else ""))
        page = _PAGE.format(title=html.escape(str(title)), meta=meta, qc=qc_html, notes=notes_html,
                            frames="\n".join(sections))
        (folder / "review.html").write_text(page, encoding="utf-8")
        files.append("review.html")
        manifest = {"c2c_contract": CONTRACT, "c2c_contract_version": "1.0", "folder": str(folder),
                    "title": str(title), "frames_compared": n, "frames_shown": pos, "difference": difference,
                    "qc": qcs, "versions": versions, "annotations": len(notes),
                    "files": [{"name": f, "sha256": _sha256(folder / f), "bytes": (folder / f).stat().st_size}
                              for f in files]}
        (folder / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
        return io.NodeOutput(str(folder / "review.html"), sheet, ab_path, json.dumps(manifest))


NODE_CLASS_MAPPINGS = {"MiniMaxH3_ReviewPackage": MiniMaxH3_ReviewPackage}
