#!/usr/bin/env python3
"""H3 zero-offset bench: does the V2V path give the plate back where it was?

Original implementation, GPL-3.0 (part of ComfyUI-MiniMax-H3-Suite). Owner order 2026-11-23: "we don't want any
pixel offset when doing V2V". Design note and findings: docs/PLAN_h3_zero_offset.md.

Every chain runs the Suite's real node classes (``execute``) with the diffusion model and VAE replaced by a stand-in
whose geometry is known exactly:

* ``exact``   - the edit returns its input unchanged (an identity edit).
* ``vae16``   - core's H3 VAE geometry: each side centre-cropped to a multiple of 16 before encoding
                (comfy/sd.py ``vae_encode_crop_pixels``), so the decode comes back that size, plus a shift-free 2x
                area-down / bilinear-up softening (a decoder is not sharp). Any offset it shows is geometry, never
                the stand-in.

The plate carries independent signals per channel so one measurement cannot hide another:

* red   - a band-limited checkerboard (phase correlation),
* green - smooth gradients (colour / level shifts),
* blue  - Gaussian fiducial dots at known SUB-PIXEL positions on a flat background (centroids, the primary
          registration number).

Measured per chain:

* ``ext_box_255``  - largest |out - plate| x 255 per channel OUTSIDE every crop box (must be 0: nothing touched).
* ``ext_mask_255`` - the same outside the edit mask, for the modes that promise it (exact outside mask, plate
                     restore with an edit mask).
* ``reg_px``       - largest dot-centroid displacement inside the fully pasted region (target < 0.25 px).
* ``pc_px``        - phase-correlation shift of the red channel over the pasted region (cross-check).

Run with the ComfyUI python (torch + comfy importable)::

    comfy_env/python.exe tools/h3_zero_offset_bench.py [--json out.json] [--sizes 1922x1081,1366x770,1081x1921]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
_env = os.environ.get("COMFYUI_PATH", "")
for _cand in ((Path(_env),) if _env else ()) + (ROOT.parent.parent / "ComfyUI_windows_portable" / "ComfyUI",
                                               ROOT.parent / "third_party" / "ComfyUI"):
    if (_cand / "comfy_api").is_dir() and str(_cand) not in sys.path:
        sys.path.insert(1, str(_cand))
        break

DOT_SIGMA = 2.2
DOT_SPACING = 48
DEFAULT_SIZES = ((1922, 1081), (1366, 770), (1081, 1921))


# ── synthetic plate ──────────────────────────────────────────────────────────────────────────────────────────────

def _blur1d(x: torch.Tensor, sigma: float, dim: int) -> torch.Tensor:
    r = max(1, int(math.ceil(3 * sigma)))
    k = torch.exp(-(torch.arange(-r, r + 1, dtype=torch.float32) ** 2) / (2 * sigma * sigma))
    k = k / k.sum()
    x = x.movedim(dim, -1)
    shp = x.shape
    y = F.conv1d(F.pad(x.reshape(-1, 1, shp[-1]), (r, r), mode="replicate"), k.view(1, 1, -1))
    return y.reshape(shp).movedim(-1, dim)


def dot_positions(w: int, h: int, t: int) -> list[torch.Tensor]:
    """Per frame [N, 2] (x, y) dot centres in continuous pixel coordinates (pixel i covers [i, i+1))."""
    out = []
    for f in range(t):
        xs = torch.arange(DOT_SPACING / 2, w - DOT_SPACING / 4, DOT_SPACING, dtype=torch.float64)
        ys = torch.arange(DOT_SPACING / 2, h - DOT_SPACING / 4, DOT_SPACING, dtype=torch.float64)
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        # irrational sub-pixel phases, different per dot and per frame, so no grid phase is favoured
        jx = torch.frac(gx * 0.6180339 + gy * 0.4142135 + f * 0.37) - 0.5
        jy = torch.frac(gx * 0.7320508 + gy * 0.2360679 + f * 0.29) - 0.5
        out.append(torch.stack([(gx + jx * 6).flatten(), (gy + jy * 6).flatten()], 1))
    return out


def make_plate(w: int, h: int, t: int = 5) -> tuple[torch.Tensor, list[torch.Tensor]]:
    yy = torch.arange(h, dtype=torch.float32).view(h, 1) + 0.5
    xx = torch.arange(w, dtype=torch.float32).view(1, w) + 0.5
    frames = []
    dots = dot_positions(w, h, t)
    for f in range(t):
        chk = (((torch.floor((xx + 3 * f) / 20) + torch.floor(yy / 20)) % 2) * 0.3 + 0.1).expand(h, w)
        # aperiodic band-limited texture so phase correlation has one true peak (a pure checker aliases by its period)
        g = torch.Generator().manual_seed(1234 + f)
        tex = _blur1d(_blur1d(torch.rand(h, w, generator=g), 1.6, 0), 1.6, 1)
        tex = (tex - tex.mean()) / tex.std().clamp_min(1e-6) * 0.12
        chk = (_blur1d(_blur1d(chk.clone(), 0.9, 0), 0.9, 1) + 0.3 + tex).clamp(0.0, 1.0)
        grad = 0.15 + 0.7 * (0.6 * xx / w + 0.4 * yy / h) + 0.04 * torch.sin(xx / 37.0 + f) * torch.cos(yy / 53.0)
        blue = torch.full((h, w), 0.15)
        p = dots[f].float()
        for x0, y0 in p.tolist():
            xa, xb = max(0, int(x0 - 4 * DOT_SIGMA)), min(w, int(x0 + 4 * DOT_SIGMA) + 1)
            ya, yb = max(0, int(y0 - 4 * DOT_SIGMA)), min(h, int(y0 + 4 * DOT_SIGMA) + 1)
            gx = xx[0, xa:xb] - x0
            gy = yy[ya:yb, 0] - y0
            blue[ya:yb, xa:xb] += 0.75 * torch.exp(-(gy.view(-1, 1) ** 2 + gx.view(1, -1) ** 2) / (2 * DOT_SIGMA ** 2))
        frames.append(torch.stack([chk, grad.expand(h, w), blue.clamp(0, 1)], -1))
    return torch.stack(frames).contiguous(), dots


def subject_masks(w: int, h: int, t: int = 5) -> torch.Tensor:
    """An ellipse that drifts across the frame (the 'subject')."""
    yy = torch.arange(h, dtype=torch.float32).view(h, 1) + 0.5
    xx = torch.arange(w, dtype=torch.float32).view(1, w) + 0.5
    out = []
    for f in range(t):
        cx = w * (0.42 + 0.03 * f) + 0.37
        cy = h * (0.5 + 0.01 * f) + 0.61
        rx, ry = 0.11 * min(w, h), 0.16 * min(w, h)
        out.append((((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 <= 1.0).float())
    return torch.stack(out)


# ── edit stand-ins (VAE + model replaced by known geometry) ──────────────────────────────────────────────────────

def standin_exact(x: torch.Tensor) -> torch.Tensor:
    return x.clone()


def standin_vae16(x: torch.Tensor) -> torch.Tensor:
    """Core's H3 VAE geometry (centre crop of each side to a multiple of 16) + shift-free softening."""
    _t, h, w, _c = x.shape
    kh, kw = h // 16 * 16, w // 16 * 16
    y0, x0 = (h % 16) // 2, (w % 16) // 2
    x = x[:, y0:y0 + kh, x0:x0 + kw, :]
    n = x.movedim(-1, 1)
    lo = F.avg_pool2d(n, 2)
    up = F.interpolate(lo, size=(kh, kw), mode="bilinear", align_corners=False)
    return up.movedim(1, -1).contiguous()


STANDINS: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {"exact": standin_exact, "vae16": standin_vae16}


# ── measurement ──────────────────────────────────────────────────────────────────────────────────────────────────

def _centroid(img: torch.Tensor, x0: float, y0: float, r: int) -> tuple[float, float] | None:
    h, w = img.shape
    xa, xb, ya, yb = int(round(x0)) - r, int(round(x0)) + r + 1, int(round(y0)) - r, int(round(y0)) + r + 1
    if xa < 0 or ya < 0 or xb > w or yb > h:
        return None
    patch = img[ya:yb, xa:xb].double()
    patch = (patch - patch.min()).clamp_min(0)
    s = patch.sum()
    if s <= 1e-9:
        return None
    ys = torch.arange(ya, yb, dtype=torch.float64).view(-1, 1) + 0.5
    xs = torch.arange(xa, xb, dtype=torch.float64).view(1, -1) + 0.5
    return float((patch * xs).sum() / s), float((patch * ys).sum() / s)


def dot_registration(out: torch.Tensor, plate: torch.Tensor, region: torch.Tensor,
                     dots: list[torch.Tensor]) -> tuple[float, float, int]:
    """(max |displacement| px, mean |displacement| px, dots measured) of the blue fiducials inside ``region``."""
    r = int(math.ceil(5 * DOT_SIGMA))
    worst, acc, n = 0.0, 0.0, 0
    for f in range(min(out.shape[0], len(dots))):
        reg = region[f]
        for x0, y0 in dots[f].tolist():
            xi, yi = int(x0), int(y0)
            ya, yb, xa, xb = yi - r - 1, yi + r + 2, xi - r - 1, xi + r + 2
            if ya < 0 or xa < 0 or yb > reg.shape[0] or xb > reg.shape[1] or not bool(reg[ya:yb, xa:xb].all()):
                continue
            a = _centroid(plate[f, ..., 2], x0, y0, r)
            b = _centroid(out[f, ..., 2], x0, y0, r)
            if a is None or b is None:
                continue
            d = math.hypot(b[0] - a[0], b[1] - a[1])
            worst, acc, n = max(worst, d), acc + d, n + 1
    return worst, (acc / n if n else 0.0), n


def phase_shift(a: torch.Tensor, b: torch.Tensor) -> tuple[float, float]:
    """Sub-pixel (dx, dy) such that b ~ a shifted by (dx, dy); parabolic peak refinement on the cross-power."""
    a = a.double() - a.double().mean()
    b = b.double() - b.double().mean()
    h, w = a.shape
    win = torch.hann_window(h, periodic=False, dtype=torch.float64).view(-1, 1) * \
        torch.hann_window(w, periodic=False, dtype=torch.float64).view(1, -1)
    fa, fb = torch.fft.fft2(a * win), torch.fft.fft2(b * win)
    r = fb * fa.conj()
    r = r / r.abs().clamp_min(1e-12)
    c = torch.fft.ifft2(r).real
    iy, ix = divmod(int(torch.argmax(c)), w)

    def para(m1, c0, p1):
        den = m1 - 2 * c0 + p1
        return 0.0 if abs(den) < 1e-15 else 0.5 * (m1 - p1) / den

    dy = iy + para(float(c[(iy - 1) % h, ix]), float(c[iy, ix]), float(c[(iy + 1) % h, ix]))
    dx = ix + para(float(c[iy, (ix - 1) % w]), float(c[iy, ix]), float(c[iy, (ix + 1) % w]))
    if dy > h / 2:
        dy -= h
    if dx > w / 2:
        dx -= w
    return dx, dy


def region_phase(out: torch.Tensor, plate: torch.Tensor, region: torch.Tensor) -> float:
    worst = 0.0
    for f in range(out.shape[0]):
        ys, xs = torch.nonzero(region[f], as_tuple=True)
        if ys.numel() < 64 * 64:
            continue
        ya, yb, xa, xb = int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1
        dx, dy = phase_shift(plate[f, ya:yb, xa:xb, 0], out[f, ya:yb, xa:xb, 0])
        worst = max(worst, math.hypot(dx, dy))
    return worst


def ext_255(out: torch.Tensor, plate: torch.Tensor, keep: torch.Tensor) -> list[float]:
    """Largest |out - plate| x 255 per channel where ``keep`` (bool [T,H,W]) is set."""
    n = min(out.shape[0], plate.shape[0])
    if out.shape[1:3] != plate.shape[1:3]:
        return [float("inf")] * 3
    d = (out[:n, ..., :3].float() - plate[:n, ..., :3].float()).abs()
    k = keep[:n]
    if not bool(k.any()):
        return [0.0, 0.0, 0.0]
    return [round(float(d[..., c][k].max()) * 255.0, 4) for c in range(3)]


def erode(mask: torch.Tensor, px: int) -> torch.Tensor:
    if px <= 0:
        return mask
    m = mask.float().unsqueeze(1)
    m = -F.max_pool2d(-m, 2 * px + 1, stride=1, padding=px)
    return m.squeeze(1) > 0.5


def dilate(mask: torch.Tensor, px: int) -> torch.Tensor:
    if px <= 0:
        return mask
    m = mask.float().unsqueeze(1)
    return F.max_pool2d(m, 2 * px + 1, stride=1, padding=px).squeeze(1) > 0.5


def _vals(o):
    return o.result if hasattr(o, "result") else o


# ── chains ───────────────────────────────────────────────────────────────────────────────────────────────────────

def _box_mask(t: int, h: int, w: int, boxes: list[tuple[float, float, float, float]]) -> torch.Tensor:
    m = torch.zeros(t, h, w, dtype=torch.bool)
    for f, (x, y, bw, bh) in enumerate(boxes[:t]):
        xa, ya = max(0, int(math.floor(x)) - 1), max(0, int(math.floor(y)) - 1)
        xb, yb = min(w, int(math.ceil(x + bw)) + 1), min(h, int(math.ceil(y + bh)) + 1)
        m[f, ya:yb, xa:xb] = True
    return m


def chain_track_stitch(plate, masks, standin, *, canvas_mode="manual", paste_region="full_crop",
                       colour_match=1.0, crop_factor=2.5):
    from mmx_nodes.stitch_back import MiniMaxH3_StitchBack
    from mmx_nodes.track_crop import MiniMaxH3_TrackCrop
    t, h, w, _ = plate.shape
    r = _vals(MiniMaxH3_TrackCrop.execute(plate, subject_mask=masks, canvas_mode=canvas_mode,
                                          crop_factor=crop_factor))
    crops, tr = r[0], r[1]
    edited = standin(crops)
    s = _vals(MiniMaxH3_StitchBack.execute(plate, edited, tr, paste_region=paste_region,
                                           colour_match=colour_match))
    out, alpha = s[0], s[1]
    inside = _box_mask(t, h, w, list(tr.boxes))
    return {"out": out, "keep_box": ~inside, "full": erode(alpha >= 0.999, 3),
            "info": f"canvas {tr.canvas[0]}x{tr.canvas[1]} box {tr.boxes[0][2]:.1f}x{tr.boxes[0][3]:.1f}"}


def chain_subject_uncrop(plate, masks, standin, *, protect="feathered box", div=32, mp=0.0, mode="combined",
                         crop_scale=1.5):
    from mmx_nodes.subject_crop import MiniMaxH3_SubjectCrop, MiniMaxH3_SubjectUncrop
    t, h, w, _ = plate.shape
    m = {"mode": mode, "crop_scale": crop_scale, "aspect_ratio": 0.0}
    if mode != "combined":
        m.update({"padding": "firm", "prefer": "stillness", "pad_surplus_tol": 16, "zoom_step": 1.0,
                  "seamless_loop": False})
    r = _vals(MiniMaxH3_SubjectCrop.execute(plate, masks, m, div, mp))
    crops, cmasks, bboxes = r[0], r[1], r[2]
    edited = standin(crops)
    u = _vals(MiniMaxH3_SubjectUncrop.execute(edited, plate, bboxes, 16, cmasks, protect))
    out = u[0]
    ledger = json.loads(u[2])
    boxes = [tuple(e["box"]) for e in ledger]
    inside = _box_mask(t, h, w, boxes)
    full = torch.zeros(t, h, w, dtype=torch.bool)
    for f, (x, y, bw, bh) in enumerate(boxes):
        full[f, y + 20:y + bh - 20, x + 20:x + bw - 20] = True
    if crop_scale <= 0:
        inside[:] = True
    if protect in ("exact outside mask", "soft mask edge", "feathered box"):
        # cropped_masks are wired: the blend follows the mask, so a dot straddling its edge mixes edited and plate
        # pixels - a blend, not a geometric shift. Registration is measured where alpha is 1.
        full &= erode(masks > 0.5, 18)
    keep_mask = ~dilate(masks > 0.5, 1) if protect in ("exact outside mask", "soft mask edge") else None
    return {"out": out, "keep_box": ~inside, "keep_mask": keep_mask, "full": full,
            "info": f"crop {crops.shape[2]}x{crops.shape[1]} box {boxes[0][2]}x{boxes[0][3]}"}


def chain_plate_restore(plate, masks, standin, *, register="correct", colour="affine", div=32):
    from mmx_nodes.plate_restore import MiniMaxH3_PlateRestore
    from mmx_nodes.subject_crop import MiniMaxH3_SubjectCrop
    t, h, w, _ = plate.shape
    r = _vals(MiniMaxH3_SubjectCrop.execute(plate, masks, {"mode": "combined", "crop_scale": 1.5,
                                                            "aspect_ratio": 0.0}, div, 0.0))
    crops, cmasks, bboxes = r[0], r[1], r[2]
    edited = standin(crops)
    p = _vals(MiniMaxH3_PlateRestore.execute(edited, plate, bboxes, edit_mask=cmasks, register=register,
                                             colour=colour))
    out, alpha = p[0], p[1]
    boxes = [(b[0]["x"], b[0]["y"], b[0]["width"], b[0]["height"]) for b in bboxes]
    inside = _box_mask(t, h, w, boxes)
    return {"out": out, "keep_box": ~inside, "keep_mask": alpha <= 0.0, "keep_mask_strict": ~dilate(masks > 0.5, 40),
            "full": erode(alpha >= 0.999, 3), "info": f"crop {crops.shape[2]}x{crops.shape[1]}"}


def chain_detail_reinject(plate, masks, standin):
    """Identity edit composited exactly, then Detail Reinject: identity in must give identity out."""
    from mmx_nodes.detail_reinject import MiniMaxH3_DetailReinject
    t, h, w, _ = plate.shape
    out = _vals(MiniMaxH3_DetailReinject.execute(plate, plate.clone(), masks))[0]
    return {"out": out, "keep_box": masks <= 0.0, "full": erode(masks > 0.5, 3), "info": "generated == plate"}


def _standin_floor(plate, masks):
    t, h, w, _ = plate.shape
    kw, kh = w // 16 * 16, h // 16 * 16
    x0, y0 = (w % 16) // 2, (h % 16) // 2
    out = plate.clone()
    out[:, y0:y0 + kh, x0:x0 + kw] = standin_vae16(plate)
    full = torch.zeros(t, h, w, dtype=torch.bool)
    full[:, y0 + 4:y0 + kh - 4, x0 + 4:x0 + kw - 4] = True
    return {"out": out, "keep_box": ~dilate(full, 5), "full": full, "info": "reference, not a chain"}


def run_all(sizes=DEFAULT_SIZES, frames: int = 5, quick: bool = False) -> list[dict[str, Any]]:
    rows = []
    for (w, h) in sizes:
        plate, dots = make_plate(w, h, frames)
        masks = subject_masks(w, h, frames)
        cases: list[tuple[str, Callable[[], dict]]] = []
        for sname, sfn in STANDINS.items():
            cases += [
                (f"TrackCrop->StitchBack full_crop cm=0 [{sname}]",
                 lambda sfn=sfn: chain_track_stitch(plate, masks, sfn, paste_region="full_crop", colour_match=0.0)),
                (f"TrackCrop->StitchBack full_crop cm=1 [{sname}]",
                 lambda sfn=sfn: chain_track_stitch(plate, masks, sfn, paste_region="full_crop", colour_match=1.0)),
                (f"TrackCrop(auto)->StitchBack full_crop [{sname}]",
                 lambda sfn=sfn: chain_track_stitch(plate, masks, sfn, canvas_mode="auto_no_downscale",
                                                    paste_region="full_crop", colour_match=0.0)),
                (f"SubjectCrop->Uncrop feathered box [{sname}]",
                 lambda sfn=sfn: chain_subject_uncrop(plate, masks, sfn, protect="feathered box")),
                (f"SubjectCrop->Uncrop exact outside mask [{sname}]",
                 lambda sfn=sfn: chain_subject_uncrop(plate, masks, sfn, protect="exact outside mask")),
                (f"SubjectCrop(1MP)->Uncrop whole crop [{sname}]",
                 lambda sfn=sfn: chain_subject_uncrop(plate, masks, sfn, protect="whole crop", mp=1.0)),
                (f"SubjectCrop(zoomed)->Uncrop whole crop [{sname}]",
                 lambda sfn=sfn: chain_subject_uncrop(plate, masks, sfn, protect="whole crop", mode="zoomed")),
                (f"SubjectCrop(div 8)->Uncrop whole crop [{sname}]",
                 lambda sfn=sfn: chain_subject_uncrop(plate, masks, sfn, protect="whole crop", div=8)),
                (f"SubjectCrop->PlateRestore correct/affine [{sname}]",
                 lambda sfn=sfn: chain_plate_restore(plate, masks, sfn)),
                (f"SubjectCrop->PlateRestore off/off [{sname}]",
                 lambda sfn=sfn: chain_plate_restore(plate, masks, sfn, register="off", colour="off")),
            ]
            if quick:
                break
        for div in (8, 16, 32):
            cases.append((f"SubjectCrop(full frame, div {div})->Uncrop whole crop [vae16]",
                          lambda div=div: chain_subject_uncrop(plate, masks, standin_vae16, protect="whole crop",
                                                               div=div, crop_scale=0.0)))
        hdr = plate * 1.5
        cases.append(("HDR x1.5 SubjectCrop(zoomed)->Uncrop whole crop [exact]",
                      lambda: dict(chain_subject_uncrop(hdr, masks, standin_exact, protect="whole crop", mode="zoomed"),
                                   ref=hdr)))
        cases.append(("HDR x1.5 SubjectCrop(1MP)->Uncrop whole crop [exact]",
                      lambda: dict(chain_subject_uncrop(hdr, masks, standin_exact, protect="whole crop", mp=1.0),
                                   ref=hdr)))
        cases.append(("DetailReinject identity", lambda: chain_detail_reinject(plate, masks, standin_exact)))
        # the stand-in's own floor: vae16 applied to the whole (on-grid part of the) plate with no chain at all. Its
        # 2x box-filter aliasing moves dot centroids ~0.1 px by itself; chain rows at that level are the stand-in.
        cases.append(("stand-in floor: vae16 on the plate, no chain", lambda: _standin_floor(plate, masks)))
        for name, fn in cases:
            try:
                res = fn()
            except Exception as exc:  # noqa: BLE001 - a bench row, not a crash
                rows.append({"size": f"{w}x{h}", "chain": name, "error": f"{type(exc).__name__}: {exc}"})
                continue
            out = res["out"]
            ref = res.get("ref", plate)
            row: dict[str, Any] = {"size": f"{w}x{h}", "chain": name, "info": res.get("info", ""),
                                   "out_size": f"{out.shape[2]}x{out.shape[1]}"}
            row["ext_box_255"] = ext_255(out, ref, res["keep_box"])
            if res.get("keep_mask") is not None:
                row["ext_mask_255"] = ext_255(out, ref, res["keep_mask"])
            if res.get("keep_mask_strict") is not None:
                row["ext_mask40_255"] = ext_255(out, ref, res["keep_mask_strict"])
            if out.shape[1:3] == ref.shape[1:3]:
                reg, mean, nd = dot_registration(out, ref, res["full"], dots)
                row["reg_px"], row["reg_mean_px"], row["dots"] = round(reg, 4), round(mean, 4), nd
                row["pc_px"] = round(region_phase(out, ref, res["full"]), 4)
                full = res["full"][: out.shape[0]]
                if bool(full.any()):
                    d = (out[..., :3].float() - ref[: out.shape[0], ..., :3].float()).abs()
                    row["int_mean_255"] = round(float(d[full].mean()) * 255.0, 4)
                    row["int_max_255"] = round(float(d[full].max()) * 255.0, 4)
            rows.append(row)
    return rows


def format_rows(rows: list[dict[str, Any]]) -> str:
    lines = [f"{'size':>10} | {'chain':<52} | {'ext box /255':>18} | {'ext mask /255':>18} | {'reg px':>7} | "
             f"{'pc px':>7} | {'int /255':>8} | info"]
    for r in rows:
        if "error" in r:
            lines.append(f"{r['size']:>10} | {r['chain']:<52} | ERROR {r['error']}")
            continue
        eb = "/".join(f"{v:g}" for v in r["ext_box_255"])
        em = "/".join(f"{v:g}" for v in r.get("ext_mask_255", [])) or "-"
        lines.append(f"{r['size']:>10} | {r['chain']:<52} | {eb:>18} | {em:>18} | {r.get('reg_px', '-'):>7} | "
                     f"{r.get('pc_px', '-'):>7} | {r.get('int_mean_255', '-'):>8} | max {r.get('int_max_255', '-')} {r.get('info', '')} "
                     f"[{r.get('dots', 0)} dots, out {r['out_size']}]")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--json", default="")
    ap.add_argument("--sizes", default=",".join(f"{w}x{h}" for w, h in DEFAULT_SIZES))
    ap.add_argument("--frames", type=int, default=3)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args(argv)
    sizes = tuple(tuple(int(v) for v in s.split("x")) for s in a.sizes.split(",") if s)
    rows = run_all(sizes, a.frames, a.quick)
    print(format_rows(rows))
    if a.json:
        Path(a.json).write_text(json.dumps(rows, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
