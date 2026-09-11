# MIT License — ComfyUI-H3-FaceRefine
# Author: Carasibana
# PORTED FROM: ComfyUI-H3-FaceRefine :: nodes.py @ HEAD (helpers only; not H3PerFrameDenoise)

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from .affine_transform import affine_crop
from .feather_composite import gaussian_blur_mask, subject_region_mask
from .transform_types import H3Transform

_DETECTOR_CACHE: dict[str, object] = {}


def detector_list() -> list[str]:
    """Face detectors from Impact subpack ultralytics registration if present."""
    import folder_paths

    names: list[str] = []
    for key in ("ultralytics_bbox", "ultralytics"):
        try:
            names.extend(folder_paths.get_filename_list(key))
        except Exception:
            pass
    seen, out = set(), []
    for n in names:
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out or ["face_yolov8m.pt"]


def load_detector(name: str):
    """Load a YOLO detector; raise FileNotFoundError with a clear message if missing."""
    if name in _DETECTOR_CACHE:
        return _DETECTOR_CACHE[name]

    import folder_paths

    path = None
    for key in ("ultralytics_bbox", "ultralytics"):
        try:
            path = folder_paths.get_full_path(key, name)
        except Exception:
            path = None
        if path:
            break
    if path is None:
        base = getattr(folder_paths, "models_dir", "models")
        for sub in ("ultralytics/bbox", "ultralytics", "ultralytics/segm"):
            cand = os.path.join(base, *sub.split("/"), name)
            if os.path.exists(cand):
                path = cand
                break
    if path is None:
        raise FileNotFoundError(
            f"Face detector '{name}' not found. Install a YOLO face/person model under "
            f"models/ultralytics_bbox/ or models/ultralytics/ (e.g. face_yolov8m.pt)."
        )
    from ultralytics import YOLO

    model = YOLO(path)
    _DETECTOR_CACHE[name] = model
    return model


_REC_CACHE: dict = {}


def face_recogniser(pack: str = "buffalo_l"):
    """InsightFace recognition model for identity matching. Cached."""
    if pack in _REC_CACHE:
        return _REC_CACHE[pack]
    import folder_paths
    import insightface

    root = os.path.join(getattr(folder_paths, "models_dir", "models"), "insightface")
    app = insightface.app.FaceAnalysis(
        name=pack,
        root=root,
        allowed_modules=["detection", "recognition"],
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    app.prepare(ctx_id=0, det_size=(640, 640))
    _REC_CACHE[pack] = app
    return app


def embed_faces(app, bgr: np.ndarray) -> list:
    """[(bbox, normed_embedding), ...] for every face insightface finds."""
    out = []
    for f in app.get(bgr):
        e = getattr(f, "normed_embedding", None)
        if e is None:
            continue
        out.append((f.bbox.tolist(), np.asarray(e, dtype=np.float32)))
    return out


def best_match(cands: list, ref_emb: np.ndarray):
    """Index of the candidate closest to the reference by cosine similarity, and the score."""
    if not cands or ref_emb is None:
        return None, -1.0
    sims = [float(np.dot(e, ref_emb)) for _, e in cands]
    i = int(np.argmax(sims))
    return i, sims[i]


def iou(a, b) -> float:
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def continuity_cost(box, last):
    """Distance from the predicted position, with a size-change penalty."""
    cx, cy, sz = (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0, box[3] - box[1]
    d = ((cx - last[0]) ** 2 + (cy - last[1]) ** 2) ** 0.5
    return d + abs(sz - last[2]) * 2.0


def build_clip_anchor(app, images, model, confidence, max_samples=24):
    """Average embedding of the subject taken from unambiguous clip frames."""
    b = images.shape[0]
    step = max(1, b // max_samples)
    embs = []
    for i in range(0, b, step):
        bgr = to_bgr_u8(images[i])
        det = model.predict(bgr, conf=confidence, verbose=False)[0]
        boxes = det.boxes.xyxy.tolist() if len(det.boxes) else []
        if not boxes:
            continue
        heights = sorted((box[3] - box[1] for box in boxes), reverse=True)
        if len(heights) > 1 and heights[0] < heights[1] * 1.6:
            continue
        cands = embed_faces(app, bgr)
        if not cands:
            continue
        j = max(range(len(cands)), key=lambda k: cands[k][0][3] - cands[k][0][1])
        embs.append(cands[j][1])
    if not embs:
        return None, 0
    a = np.mean(np.stack(embs), axis=0)
    n = np.linalg.norm(a)
    return (a / n if n > 0 else a), len(embs)


def to_bgr_u8(img: torch.Tensor) -> np.ndarray:
    """ComfyUI IMAGE frame [H,W,C] float 0..1 -> BGR uint8 for ultralytics."""
    a = (img[..., :3].clamp(0, 1).cpu().numpy() * 255.0).astype(np.uint8)
    return a[..., ::-1].copy()


def interp_gaps(vals: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Fill non-detected frames by linear interpolation; hold at the ends."""
    n = len(vals)
    idx = np.arange(n)
    if not valid.any():
        return np.zeros(n, dtype=np.float64)
    return np.interp(idx, idx[valid], vals[valid])


def smooth(vals: np.ndarray, window: int, method: str = "gaussian") -> np.ndarray:
    """Smooth a trajectory with reflected edges. window<=1 is a no-op."""
    if window <= 1 or len(vals) < 3:
        return vals
    window = min(int(window), len(vals))
    if window % 2 == 0:
        window += 1
    if window < 3:
        return vals
    pad = window // 2
    padded = np.pad(vals, pad, mode="reflect")

    if method == "savgol":
        try:
            from scipy.signal import savgol_filter

            polyorder = 2 if window > 3 else 1
            return np.asarray(savgol_filter(padded, window, polyorder))[pad : pad + len(vals)]
        except Exception:
            method = "gaussian"

    if method == "gaussian":
        x = np.arange(window, dtype=np.float64) - pad
        sigma = max(window / 6.0, 0.5)
        kernel = np.exp(-(x**2) / (2.0 * sigma**2))
        kernel /= kernel.sum()
    else:
        kernel = np.ones(window, dtype=np.float64) / window

    return np.convolve(padded, kernel, mode="valid")[: len(vals)]


def feather_mask(h: int, w: int, feather: int, device, dtype) -> torch.Tensor:
    """[h,w] mask: 1 in the core, cosine ramp to 0 over feather px at every edge."""
    m = torch.ones((h, w), device=device, dtype=dtype)
    f = int(max(0, min(feather, min(h, w) // 2 - 1)))
    if f <= 0:
        return m
    ramp = 0.5 - 0.5 * torch.cos(
        torch.linspace(0, np.pi, f + 2, device=device, dtype=dtype)[1:-1]
    )
    m[:f, :] *= ramp.view(-1, 1)
    m[h - f :, :] *= ramp.flip(0).view(-1, 1)
    m[:, :f] *= ramp.view(1, -1)
    m[:, w - f :] *= ramp.flip(0).view(1, -1)
    return m


def _jit(a: np.ndarray) -> float:
    return float(np.abs(np.diff(a)).mean()) if len(a) > 1 else 0.0


def track_face_crop(
    images: torch.Tensor,
    *,
    detector: str,
    confidence: float,
    crop_factor: float,
    canvas_width: int,
    canvas_height: int,
    canvas_mode: str,
    smooth_window: int,
    size_smooth_window: int,
    smooth_method: str,
    size_mode: str,
    select: str = "largest",
    fallback_detector: str = "none",
    fallback_head_frac: float = 0.5,
    identity_reference: Optional[torch.Tensor] = None,
    identity_threshold: float = 0.28,
    identity_track: bool = True,
    interrupt_check=None,
) -> tuple[torch.Tensor, H3Transform, torch.Tensor, str, int, int]:
    """Detect, track, smooth, and crop faces. Returns crops, transform, preview, report, cw, ch."""
    model = load_detector(detector)
    b, h, w, _ = images.shape

    cx = np.zeros(b)
    cy = np.zeros(b)
    sz = np.zeros(b)
    fw = np.zeros(b)
    valid = np.zeros(b, dtype=bool)
    via_body = np.zeros(b, dtype=bool)

    ref_emb, app = None, None
    n_ident, n_cont, n_conflict = 0, 0, 0
    multi = False
    try:
        probe = model.predict(to_bgr_u8(images[0]), conf=confidence, verbose=False)[0]
        multi = len(probe.boxes) > 1
    except Exception:
        pass

    if identity_track and (multi or identity_reference is not None):
        try:
            app = face_recogniser()
            if identity_reference is not None:
                cands = embed_faces(app, to_bgr_u8(identity_reference[0]))
                if cands:
                    j = max(range(len(cands)), key=lambda k: cands[k][0][3] - cands[k][0][1])
                    ref_emb = cands[j][1]
                    print("[MiniMaxH3_FaceTrackCrop] identity anchor from the supplied reference")
            if ref_emb is None:
                ref_emb, used = build_clip_anchor(app, images, model, confidence)
                if ref_emb is not None:
                    print(
                        f"[MiniMaxH3_FaceTrackCrop] identity anchor built from the clip itself "
                        f"({used} unambiguous frames)"
                    )
        except Exception as exc:
            print(f"[MiniMaxH3_FaceTrackCrop] identity matching unavailable ({exc})")

    last = None
    for i in range(b):
        if interrupt_check:
            interrupt_check()
        frame_bgr = to_bgr_u8(images[i])
        res = model.predict(frame_bgr, conf=confidence, verbose=False)[0]
        boxes = res.boxes.xyxy.tolist() if len(res.boxes) else []
        if not boxes:
            continue

        box = None
        if len(boxes) == 1:
            box = boxes[0]
            n_cont += 1
        elif last is None:
            if ref_emb is not None:
                cands = embed_faces(app, frame_bgr)
                k, _ = best_match(cands, ref_emb)
                if k is not None:
                    box = cands[k][0]
                    n_ident += 1
            if box is None:
                if select == "most_central":
                    fc = (w / 2.0, h / 2.0)
                    box = min(
                        boxes,
                        key=lambda q: ((q[0] + q[2]) / 2 - fc[0]) ** 2
                        + ((q[1] + q[3]) / 2 - fc[1]) ** 2,
                    )
                else:
                    box = max(boxes, key=lambda q: (q[3] - q[1]))
        else:
            ranked = sorted(boxes, key=lambda q: continuity_cost(q, last))
            best, second = ranked[0], ranked[1]
            c0, c1 = continuity_cost(best, last), continuity_cost(second, last)
            conflict = (c1 < c0 * 2.0) or (iou(best, second) > 0.2)
            if conflict and ref_emb is not None:
                n_conflict += 1
                near = [q for q in boxes if continuity_cost(q, last) < c0 * 3.0] or boxes
                cands = [
                    c
                    for c in embed_faces(app, frame_bgr)
                    if any(iou(c[0], q) > 0.3 for q in near)
                ]
                k, score = best_match(cands, ref_emb)
                if k is not None and score >= identity_threshold:
                    box = cands[k][0]
                    n_ident += 1
            if box is None:
                box = best
                n_cont += 1

        last = ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0, box[3] - box[1])
        cx[i] = (box[0] + box[2]) / 2.0
        cy[i] = (box[1] + box[3]) / 2.0
        sz[i] = box[3] - box[1]
        fw[i] = box[2] - box[0]
        valid[i] = True

    found = int(valid.sum())
    if found == 0:
        raise ValueError(
            "No face detected in any frame. Lower `confidence`, or this clip has no "
            "usable face and should be skipped."
        )

    sz_seed = interp_gaps(sz, valid)
    if fallback_detector != "none" and (~valid).any():
        try:
            bmodel = load_detector(fallback_detector)
            for i in np.nonzero(~valid)[0]:
                res = bmodel.predict(to_bgr_u8(images[i]), conf=confidence, verbose=False)[0]
                if not len(res.boxes):
                    continue
                bb = res.boxes.xyxy.tolist()
                cls = (
                    res.boxes.cls.tolist()
                    if getattr(res.boxes, "cls", None) is not None
                    else [0] * len(bb)
                )
                people = [q for q, cc in zip(bb, cls) if int(cc) == 0] or bb
                p = max(people, key=lambda q: (q[2] - q[0]) * (q[3] - q[1]))
                cx[i] = (p[0] + p[2]) / 2.0
                cy[i] = p[1] + fallback_head_frac * max(sz_seed[i], 8.0)
                sz[i] = sz_seed[i]
                via_body[i] = True
        except Exception as exc:
            print(f"[MiniMaxH3_FaceTrackCrop] body fallback '{fallback_detector}' failed: {exc}")

    known = valid | via_body
    raw_cx = interp_gaps(cx, known)
    raw_cy = interp_gaps(cy, known)
    raw_sz = interp_gaps(sz, valid)
    raw_fw = interp_gaps(fw, valid)
    sm_fw = smooth(raw_fw, size_smooth_window, smooth_method)
    cx = smooth(raw_cx, smooth_window, smooth_method)
    cy = smooth(raw_cy, smooth_window, smooth_method)
    sz = smooth(raw_sz, size_smooth_window, smooth_method)
    if size_mode == "max_of_clip":
        sz[:] = sz.max()

    jit_before = (_jit(raw_cx) + _jit(raw_cy)) / 2.0
    jit_after = (_jit(cx) + _jit(cy)) / 2.0
    sz_before, sz_after = _jit(raw_sz), _jit(sz)

    if canvas_mode != "manual":
        need = float(min(sz.max() * crop_factor, h))
        snapped = int(np.ceil(need / 32.0) * 32)
        if canvas_mode == "auto_capped_768":
            snapped = min(snapped, 768)
        snapped = max(128, min(snapped, 1344))
        if snapped != canvas_height:
            print(
                f"[MiniMaxH3_FaceTrackCrop] canvas_mode={canvas_mode}: "
                f"{canvas_width}x{canvas_height} -> {snapped}x{snapped} "
                f"(largest crop {need:.0f}px)"
            )
        canvas_width = canvas_height = snapped

    aspect = canvas_width / float(canvas_height)
    boxes: list[tuple[float, float, float, float]] = []
    crops = torch.zeros((b, canvas_height, canvas_width, 3), dtype=images.dtype)
    preview = images[..., :3].clone()

    for i in range(b):
        bh = sz[i] * crop_factor
        bw = bh * aspect
        if bw > w:
            bw, bh = float(w), float(w) / aspect
        if bh > h:
            bh, bw = float(h), float(h) * aspect
        x = min(max(cx[i] - bw / 2.0, 0.0), max(0.0, w - bw))
        y = min(max(cy[i] - bh / 2.0, 0.0), max(0.0, h - bh))
        box = (float(x), float(y), float(bw), float(bh))
        boxes.append(box)

        crops[i : i + 1] = affine_crop(
            images[i : i + 1], box, canvas_width, canvas_height
        ).to(crops.dtype)

        xi, yi = int(round(x)), int(round(y))
        wi, hi = max(4, int(round(bw))), max(4, int(round(bh)))
        xi = min(xi, w - wi)
        yi = min(yi, h - hi)
        if valid[i]:
            r, g = 0.0, 1.0
        elif via_body[i]:
            r, g = 1.0, 1.0
        else:
            r, g = 1.0, 0.0
        for yy0, yy1, xx0, xx1 in (
            (yi, yi + 2, xi, xi + wi),
            (yi + hi - 2, yi + hi, xi, xi + wi),
            (yi, yi + hi, xi, xi + 2),
            (yi, yi + hi, xi + wi - 2, xi + wi),
        ):
            preview[i, yy0:yy1, xx0:xx1, 0] = r
            preview[i, yy0:yy1, xx0:xx1, 1] = g
            preview[i, yy0:yy1, xx0:xx1, 2] = 0.0

    weights = smooth(valid.astype(np.float64), max(9, smooth_window // 2), "gaussian")
    weights = np.clip(weights, 0.0, 1.0)

    runs, cur = [], 0
    for v in known:
        if v:
            if cur:
                runs.append(cur)
            cur = 0
        else:
            cur += 1
    if cur:
        runs.append(cur)
    longest_gap = max(runs) if runs else 0

    mags = [canvas_height / float(box[3]) for box in boxes]
    subject_rect = [
        (
            float(canvas_width) * 0.5 - 0.5 * float(sm_fw[i]) / max(box[2], 1e-6) * canvas_width,
            float(canvas_height) * 0.5 - 0.5 * float(sz[i]) / max(box[3], 1e-6) * canvas_height,
            float(sm_fw[i]) / max(box[2], 1e-6) * canvas_width,
            float(sz[i]) / max(box[3], 1e-6) * canvas_height,
        )
        for i, box in enumerate(boxes)
    ]

    transform = H3Transform(
        boxes=tuple(boxes),
        canvas=(int(canvas_width), int(canvas_height)),
        src_size=(int(w), int(h)),
        frames=int(b),
        weights=tuple(float(v) for v in weights),
        detected=tuple(bool(v) for v in valid),
        subject_rect=tuple(subject_rect),
        crop_factor=float(crop_factor),
        planner_mode="face_track",
    )

    gapwarn = ""
    if longest_gap >= 12:
        gapwarn = (
            f"\n!! longest dropout is {longest_gap} frames ({longest_gap / 24.0:.1f}s). The crop "
            f"box is linearly interpolated across it, so it may drift if the subject moved "
            f"while turned away. Detection weighting fades the composite out there, so those "
            f"frames keep their original pixels - check the preview over that stretch."
        )

    n_down = sum(1 for m in mags if m < 1.0)
    warn = ""
    if n_down:
        need = max(box[3] for box in boxes)
        warn = (
            f"\n!! {n_down}/{b} frames ({n_down / b * 100:.0f}%) have magnification < 1.0x - "
            f"their crops are DOWNSCALED into the canvas, losing real detail.\n"
            f"   Fix: raise canvas to >= {need}px (rounded up to a multiple of 32), or lower "
            f"crop_factor, or skip this clip if it is close-up throughout."
        )

    box_jit = (
        float(
            np.mean(
                [
                    abs(boxes[i][0] - boxes[i - 1][0]) + abs(boxes[i][1] - boxes[i - 1][1])
                    for i in range(1, len(boxes))
                ]
            )
        )
        if len(boxes) > 1
        else 0.0
    )
    report = (
        f"tracking: {n_cont} by continuity, {n_conflict} ambiguous "
        f"({n_ident} resolved by face identity)\n"
        f"frames={b}  face={found} ({found / b * 100:.0f}%)  "
        f"body-fallback={int(via_body.sum())}  interpolated={b - int(known.sum())}\n"
        f"face height  min={sz.min():.0f}px  mean={sz.mean():.0f}px  max={sz.max():.0f}px\n"
        f"face fills   ~{100.0 / crop_factor:.0f}% of every crop (crop_factor={crop_factor})\n"
        f"crop box     min={min(box[3] for box in boxes)}px  max={max(box[3] for box in boxes)}px\n"
        f"magnification into {canvas_width}x{canvas_height}: "
        f"min={min(mags):.2f}x  mean={sum(mags) / len(mags):.2f}x  max={max(mags):.2f}x\n"
        f"jitter ({smooth_method}) centre {jit_before:.2f} -> {jit_after:.2f} px/frame"
        f"   size {sz_before:.2f} -> {sz_after:.2f} px/frame\n"
        f"box movement {box_jit:.2f} px/frame (sub-pixel float boxes - no integer rounding)\n"
        f"dropout runs: {len(runs)}  longest={longest_gap} frames ({longest_gap / 24.0:.1f}s "
        f"at 24fps)  -> composite fades out across these"
        f"{gapwarn}{warn}"
    )
    print("[MiniMaxH3_FaceTrackCrop] " + report.replace("\n", "\n[MiniMaxH3_FaceTrackCrop] "))
    return crops, transform, preview, report, int(canvas_width), int(canvas_height)


def stitch_refined_faces(
    base_images: torch.Tensor,
    refined_crops: torch.Tensor,
    transform: H3Transform,
    *,
    paste_region: str,
    mask_dilation: int,
    feather: int,
    colour_match: float,
    blend: float,
    undetected_frames: str = "fade_out",
    masks: Optional[torch.Tensor] = None,
    feather_scales_with_crop: bool = False,
    interrupt_check=None,
    device=None,
) -> torch.Tensor:
    """Paste refined crops back using per-frame transform with feather + colour match."""
    boxes = list(transform.boxes)
    if undetected_frames == "composite_anyway":
        weights = None
    elif undetected_frames == "skip":
        weights = [1.0 if d else 0.0 for d in transform.detected] or None
    else:
        weights = list(transform.weights)

    b = min(len(boxes), base_images.shape[0], refined_crops.shape[0])
    if base_images.shape[0] != refined_crops.shape[0]:
        print(
            f"[MiniMaxH3_FaceStitch] frame count mismatch: base={base_images.shape[0]} "
            f"refined={refined_crops.shape[0]} transform={len(boxes)} -> using {b}"
        )

    cw, ch = transform.canvas
    w, h = transform.src_size
    face_rects = transform.subject_rect

    if device is None:
        device = base_images.device
    dt = base_images.dtype
    out = base_images[..., :3].clone()

    per_frame_mb = (h * w * 3 * 4) / 2**20
    chunk = max(1, min(32, int(1024 / max(per_frame_mb, 1e-6))))

    for c0 in range(0, b, chunk):
        if interrupt_check:
            interrupt_check()
        c1 = min(c0 + chunk, b)
        n = c1 - c0

        if feather_scales_with_crop:
            f_can = int(feather)
        else:
            bh_mid = float(boxes[(c0 + c1 - 1) // 2][3])
            f_can = int(round(feather * (ch / max(bh_mid, 1.0))))
            f_can = max(1, min(f_can, ch // 3))

        if masks is not None:
            mk = masks[c0:c1].to(device).float()
            if mk.shape[-2:] != (ch, cw):
                mk = F.interpolate(mk.unsqueeze(1), size=(ch, cw), mode="bilinear", align_corners=False)
            else:
                mk = mk.unsqueeze(1)
            if mask_dilation > 0:
                k = 2 * int(mask_dilation) + 1
                mk = F.max_pool2d(mk, k, stride=1, padding=k // 2)
            mask_can = gaussian_blur_mask(mk, f_can).clamp(0, 1)
        elif paste_region == "full_crop":
            one = feather_mask(ch, cw, f_can, device, torch.float32)
            mask_can = one.view(1, 1, ch, cw).expand(n, 1, ch, cw)
        else:
            mask_can = torch.cat(
                [
                    subject_region_mask(
                        ch,
                        cw,
                        face_rects[i]
                        if face_rects and i < len(face_rects)
                        else (cw * 0.25, ch * 0.25, cw * 0.5, ch * 0.5),
                        int(mask_dilation),
                        f_can,
                        paste_region == "face_ellipse",
                        device,
                    )
                    for i in range(c0, c1)
                ],
                dim=0,
            )

        th = torch.empty((n, 2, 3), dtype=torch.float32, device=device)
        for j, i in enumerate(range(c0, c1)):
            x, y, bw, bh = (float(v) for v in boxes[i])
            th[j, 0, 0] = w / bw
            th[j, 0, 1] = 0.0
            th[j, 0, 2] = (w - 2.0 * x) / bw - 1.0
            th[j, 1, 0] = 0.0
            th[j, 1, 1] = h / bh
            th[j, 1, 2] = (h - 2.0 * y) / bh - 1.0
        grid = F.affine_grid(th, (n, 3, int(h), int(w)), align_corners=False)

        patch_can = refined_crops[c0:c1, ..., :3].to(device).movedim(-1, 1).float()
        patch = F.grid_sample(patch_can, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
        m = F.grid_sample(
            mask_can.to(device), grid, mode="bilinear", padding_mode="zeros", align_corners=False
        ).clamp(0, 1)

        patch = patch.movedim(1, -1)
        m = m.movedim(1, -1)
        base = out[c0:c1].to(device).float()

        if colour_match > 0.0:
            wsum = m.sum(dim=(1, 2), keepdim=True).clamp_min(1e-6)
            bmu = (base * m).sum(dim=(1, 2), keepdim=True) / wsum
            pmu = (patch * m).sum(dim=(1, 2), keepdim=True) / wsum
            bsd = (((base - bmu) ** 2 * m).sum(dim=(1, 2), keepdim=True) / wsum).sqrt().clamp_min(1e-6)
            psd = (((patch - pmu) ** 2 * m).sum(dim=(1, 2), keepdim=True) / wsum).sqrt().clamp_min(1e-6)
            adj = (patch - pmu) * (bsd / psd) + bmu
            patch = patch + (adj - patch) * float(colour_match)
            patch = patch.clamp(0, 1)

        wv = torch.full((n, 1, 1, 1), float(blend), device=device, dtype=torch.float32)
        if weights is not None:
            for j, i in enumerate(range(c0, c1)):
                if i < len(weights):
                    wv[j] *= float(weights[i])
        mm_ = m * wv

        out[c0:c1] = ((1.0 - mm_) * base + mm_ * patch).to(out.device, dt)

    return out


def run_sam_face_masks(
    crops: torch.Tensor,
    sam_model,
    transform: H3Transform,
    *,
    threshold: float,
    dilation: int,
    temporal_smooth: int,
    interrupt_check=None,
    progress_cb=None,
) -> tuple[torch.Tensor, str]:
    """Per-frame SAM face masks on stabilised crops, temporally smoothed."""
    sam_obj = sam_model if not hasattr(sam_model, "sam_wrapper") else sam_model.sam_wrapper
    face_rects = transform.subject_rect or []
    b, ch, cw, _ = crops.shape
    masks = torch.zeros((b, ch, cw), dtype=torch.float32)
    ok = 0

    if hasattr(sam_obj, "prepare_device"):
        sam_obj.prepare_device()

    try:
        for i in range(b):
            if interrupt_check:
                interrupt_check()
            if progress_cb:
                progress_cb(i)
            if i % 25 == 0:
                print(f"[MiniMaxH3_FaceMaskSAM] SAM mask {i}/{b}")
            fr = (
                face_rects[i]
                if i < len(face_rects)
                else (cw * 0.25, ch * 0.25, cw * 0.5, ch * 0.5)
            )
            fx, fy, fwd, fhd = fr
            bbox = [
                max(0, int(fx)),
                max(0, int(fy)),
                min(cw, int(fx + fwd)),
                min(ch, int(fy + fhd)),
            ]
            pts = [(int(fx + fwd / 2), int(fy + fhd / 2))]
            img = (crops[i, ..., :3].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
            try:
                det = sam_obj.predict(img, pts, [1], bbox, threshold)
            except Exception:
                det = None
            if det:
                m = det[0] if not isinstance(det, torch.Tensor) else det
                m = torch.as_tensor(np.asarray(m), dtype=torch.float32).squeeze()
                if m.shape[-2:] == (ch, cw):
                    masks[i] = (m > 0.5).float()
                    ok += 1
    finally:
        if hasattr(sam_obj, "release_device"):
            try:
                sam_obj.release_device()
            except Exception:
                pass

    for i in range(b):
        if masks[i].max() <= 0:
            fr = (
                face_rects[i]
                if i < len(face_rects)
                else (cw * 0.25, ch * 0.25, cw * 0.5, ch * 0.5)
            )
            fx, fy, fwd, fhd = fr
            x0, y0 = max(0, int(fx)), max(0, int(fy))
            x1, y1 = min(cw, int(fx + fwd)), min(ch, int(fy + fhd))
            if x1 > x0 and y1 > y0:
                masks[i, y0:y1, x0:x1] = 1.0

    if dilation > 0:
        k = 2 * int(dilation) + 1
        masks = F.max_pool2d(masks.unsqueeze(1), k, stride=1, padding=k // 2).squeeze(1)

    if temporal_smooth > 1 and b > 2:
        w = min(int(temporal_smooth) | 1, b if b % 2 else b - 1)
        if w >= 3:
            pad = w // 2
            t = masks.permute(1, 2, 0).reshape(-1, 1, b).contiguous()
            t = F.pad(t, (pad, pad), mode="replicate")
            kern = torch.ones(1, 1, w, dtype=t.dtype, device=t.device) / w
            sm = F.conv1d(t, kern)
            masks = sm.reshape(ch, cw, b).permute(2, 0, 1).contiguous()

    report = (
        f"SAM masks: {ok}/{b} frames segmented ({b - ok} fell back to the face rect)\n"
        f"dilation={dilation}  temporal_smooth={temporal_smooth}\n"
        f"mean coverage {float(masks.mean()) * 100:.1f}% of canvas"
    )
    print("[MiniMaxH3_FaceMaskSAM] " + report)
    return masks, report


def format_transform_info(transform: H3Transform, max_rows: int) -> str:
    """Format per-frame transform table for sanity-checking tracking."""
    boxes = transform.boxes
    cw, ch = transform.canvas
    lines = [
        f"frames={transform.frames}  canvas={cw}x{ch}  src={transform.src_size}",
        f"{'frame':>6} {'x':>6} {'y':>6} {'w':>6} {'h':>6} {'mag':>6}",
    ]
    step = max(1, len(boxes) // max_rows)
    for i in range(0, len(boxes), step):
        x, y, bw, bh = boxes[i]
        lines.append(f"{i:>6} {x:>6.1f} {y:>6.1f} {bw:>6.1f} {bh:>6.1f} {ch / bh:>5.2f}x")
    return "\n".join(lines)
