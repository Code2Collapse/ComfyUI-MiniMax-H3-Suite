# SPDX-License-Identifier: GPL-3.0-or-later
# PORTED FROM: open-source-higgsfield (lucataco, Apache-2.0) :: bodydouble/runner.py detect / track / _reacquire @ 864c8de
#   (third_party/open-source-higgsfield; licence + NOTICE in THIRD_PARTY_LICENSES/open-source-higgsfield-*)
"""Subject selection, track-loss detection and re-identification for the character-swap track job (owner A21 Rev 3
Part 10; ledger L9.78 slice 3).

Ported from the reference:
- the pick on frame 0 (the largest person, or a chosen index);
- the loss rule: no mask, an area under 35% of the median of the last 5 frames, or a centre jump over 15% of the
  frame width;
- the re-acquire score: colour-histogram similarity to the performer's mean histogram minus 0.5 x the centre distance
  (a fraction of the width).

Changed here (Part 10: "do not silently change the selected subject ID when confidence is insufficient"): the
reference always takes the best candidate; here a choice whose score is under a threshold, or too close to the second
candidate, goes to REVIEW. Selection also weighs size, a reference's appearance and a position preference, and a
target lock keeps the subject of an earlier run.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np

from . import body_double as BD

SUBJECT_CONTRACT = "h3.subject"
SUBJECT_VERSION = "1.0"
POSITIONS = ("none", "centre", "left", "right", "top", "bottom")


# ── detections in, candidates out ─────────────────────────────────────────────────────────────────────────────────

def frames_of(bboxes) -> list[list[dict]]:
    """A BOUNDING_BOX value as one list of box dicts per frame: core SAM3_Detect gives a list per frame; other nodes
    give one dict or a flat list."""
    if bboxes is None:
        return []
    if isinstance(bboxes, dict):
        return [[bboxes]]
    seq = list(bboxes)
    if seq and all(isinstance(b, dict) for b in seq):
        return [seq]
    return [list(f) if isinstance(f, (list, tuple)) else [f] for f in seq]


def _xyxy(d: dict) -> list[float]:
    x, y = float(d.get("x", 0)), float(d.get("y", 0))
    return [x, y, x + float(d.get("width", 0)), y + float(d.get("height", 0))]


def box_mask(box, h: int, w: int) -> np.ndarray:
    m = np.zeros((h, w), np.float32)
    x0, y0, x1, y1 = (int(round(v)) for v in box)
    m[max(0, y0):max(0, min(h, y1)), max(0, x0):max(0, min(w, x1))] = 1.0
    return m


def candidates(frame, boxes: list[dict] | None, masks: list | None = None, bins: int = 8) -> list[dict]:
    """One frame's detections as candidates: box [x0, y0, x1, y1], detector score, mask (its own, or the box), area and
    appearance (colour histogram under the mask). Masks without boxes (a mask-only source) get the mask's box."""
    f = BD._np(frame)
    h, w = int(f.shape[0]), int(f.shape[1])
    out: list[dict] = []
    boxes = list(boxes or [])
    masks = [BD._np(m) for m in (masks or [])]
    if masks and boxes and len(masks) != len(boxes):
        masks = []                                         # a union mask cannot be split per person: boxes only
    n = max(len(boxes), len(masks))
    for k in range(n):
        m = masks[k] if k < len(masks) else None
        if k < len(boxes):
            box, score = _xyxy(boxes[k]), float(boxes[k].get("score", 1.0))
        else:
            b = BD.bbox(m) if m is not None else None
            if b is None:
                continue
            box, score = list(b), 1.0
        if m is None or m.shape != (h, w):
            m = box_mask(box, h, w)
        area = float((m > 0.5).sum())
        if area <= 0:
            continue
        out.append({"index": k, "box": [round(v, 2) for v in box], "score": round(score, 4), "mask": m,
                    "area": area, "hist": BD.color_hist(f, m, bins)})
    return out


def appearance(frames, masks, bins: int = 8) -> np.ndarray | None:
    """Mean colour histogram of a subject over frames (reference images: their masks, or the whole image)."""
    hs = []
    for i in range(len(frames)):
        m = masks[min(i, len(masks) - 1)] if masks is not None and len(masks) else None
        f = BD._np(frames[i])
        hs.append(BD.color_hist(f, BD._np(m) if m is not None else np.ones(f.shape[:2], np.float32), bins))
    hs = [h for h in hs if h.sum() > 0]
    return np.mean(hs, axis=0) if hs else None


# ── selection ────────────────────────────────────────────────────────────────────────────────────────────────────

def _position_score(box, w: int, h: int, prefer: str) -> float:
    cx, cy = (box[0] + box[2]) / 2 / max(w, 1), (box[1] + box[3]) / 2 / max(h, 1)
    return {"centre": 1.0 - min(1.0, float(np.hypot(cx - 0.5, cy - 0.5)) / 0.7071),
            "left": 1.0 - cx, "right": cx, "top": 1.0 - cy, "bottom": cy}.get(prefer, 0.0)


def rank_selection(cands: list[dict], w: int, h: int, *, size_weight: float = 1.0, ref_hist=None,
                   reference_weight: float = 1.0, position: str = "none", position_weight: float = 0.0) -> list[dict]:
    """Every candidate scored 0..1: a weighted mean of its size (area / the largest), its appearance similarity to the
    reference, and the position preference. Best first."""
    if not cands:
        return []
    big = max(c["area"] for c in cands)
    parts = [("size", float(size_weight))]
    if ref_hist is not None:
        parts.append(("reference", float(reference_weight)))
    if position in POSITIONS and position != "none":
        parts.append(("position", float(position_weight)))
    total = sum(wt for _, wt in parts if wt > 0) or 1.0
    ranked = []
    for c in cands:
        terms = {"size": c["area"] / big}
        if ref_hist is not None:
            terms["reference"] = BD.hist_similarity(c["hist"], ref_hist)
        if position != "none":
            terms["position"] = _position_score(c["box"], w, h, position)
        s = sum(terms[k] * wt for k, wt in parts if wt > 0) / total
        ranked.append({"index": c["index"], "score": round(float(s), 4), "terms": {k: round(float(v), 4)
                                                                                   for k, v in terms.items()},
                       "box": c["box"], "area": c["area"]})
    ranked.sort(key=lambda r: -r["score"])
    return ranked


def rank_reid(frame_w: int, cands: list[dict], ref_hist, last_box, *, appearance_weight: float = 1.0,
              motion_weight: float = 0.5, search_distance: float = 1.0) -> list[dict]:
    """The reference's re-acquire score: appearance x similarity - motion_weight x centre distance (a fraction of the
    frame width). Candidates farther than ``search_distance`` (a fraction of the width) are left out. Best first."""
    ranked = []
    for c in cands:
        b = c["box"]
        dist = abs((b[0] + b[2]) / 2 - (last_box[0] + last_box[2]) / 2) / max(frame_w, 1) if last_box else 0.0
        if dist > float(search_distance):
            continue
        sim = BD.hist_similarity(c["hist"], ref_hist) if ref_hist is not None else 0.0
        s = float(appearance_weight) * sim - float(motion_weight) * dist
        ranked.append({"index": c["index"], "score": round(float(s), 4), "similarity": round(float(sim), 4),
                       "distance": round(float(dist), 4), "box": b, "area": c["area"]})
    ranked.sort(key=lambda r: -r["score"])
    return ranked


def decide(ranked: list[dict], *, min_score: float | None = None, margin: float = 0.0) -> tuple[str, str]:
    """('accept', why) or ('review', why): review when there is no candidate, the best is under ``min_score``, or the
    second is within ``margin`` of it."""
    if not ranked:
        return "review", "no candidate"
    best = ranked[0]
    if min_score is not None and best["score"] < float(min_score):
        return "review", f"the best candidate (#{best['index']}) scores {best['score']:.3f}, under {float(min_score):.3f}"
    if len(ranked) > 1 and best["score"] - ranked[1]["score"] < float(margin):
        return "review", (f"candidates #{best['index']} ({best['score']:.3f}) and #{ranked[1]['index']} "
                          f"({ranked[1]['score']:.3f}) are within {float(margin):.3f} of each other")
    return "accept", f"#{best['index']} scores {best['score']:.3f}"


# ── the subject ──────────────────────────────────────────────────────────────────────────────────────────────────

def subject_id(frame: int, box, w: int, h: int) -> str:
    """Stable id: the same pick on the same frame always gives the same id (the box quantised to 1/64 of the frame,
    so detector noise of a pixel does not change it)."""
    q = [int(round(box[0] / max(w, 1) * 64)), int(round(box[1] / max(h, 1) * 64)),
         int(round(box[2] / max(w, 1) * 64)), int(round(box[3] / max(h, 1) * 64))]
    return "S" + hashlib.sha1(f"{int(frame)}:{q}:{w}x{h}".encode()).hexdigest()[:8]


def subject_json(sid: str, frame: int, box, w: int, h: int, hist, area: float, **extra: Any) -> str:
    d = {"c2c_contract": SUBJECT_CONTRACT, "c2c_contract_version": SUBJECT_VERSION, "subject_id": sid,
         "frame": int(frame), "box": [round(float(v), 2) for v in box], "frame_size": [int(w), int(h)],
         "area": round(float(area), 1), "hist": [round(float(v), 6) for v in np.asarray(hist).ravel()]}
    d.update(extra)
    return json.dumps(d)


def parse_subject(text: str | None, who: str) -> dict | None:
    if text is None or not str(text).strip():
        return None
    try:
        d = json.loads(text)
    except ValueError as exc:
        raise ValueError(f"{who}: 'subject' is not JSON ({exc}) - connect H3 Multi-Subject Selector's subject.") from exc
    if not isinstance(d, dict) or d.get("c2c_contract") != SUBJECT_CONTRACT or not d.get("subject_id"):
        raise ValueError(f"{who}: 'subject' is not an H3 subject (contract {SUBJECT_CONTRACT}) - connect H3 "
                         "Multi-Subject Selector's or H3 Temporal Subject Tracker's subject output.")
    return d


def scaled_box(box, from_size, to_size) -> list[float]:
    sx, sy = to_size[0] / max(from_size[0], 1), to_size[1] / max(from_size[1], 1)
    return [box[0] * sx, box[1] * sy, box[2] * sx, box[3] * sy]


# ── the loss rule ────────────────────────────────────────────────────────────────────────────────────────────────

def loss_reason(mask, areas: list[float], last_box, width: int, *, area_ratio: float = 0.35,
                jump: float = 0.15) -> str | None:
    """Why the tracked mask is no longer the subject, or None. ``areas``: the accepted areas so far (the median of the
    last 5 is the reference)."""
    b = BD.bbox(mask)
    if b is None:
        return "no mask"
    area = float((BD._np(mask) > 0.5).sum())
    recent = float(np.median(areas[-5:])) if areas else area
    if area < float(area_ratio) * recent:
        return f"area {area:.0f} px is under {float(area_ratio):.0%} of the recent {recent:.0f} px"
    if last_box is not None:
        d = abs((b[0] + b[2]) / 2 - (last_box[0] + last_box[2]) / 2)
        if d > float(jump) * width:
            return f"the centre jumped {d:.0f} px (over {float(jump):.0%} of the width)"
    return None


def iou(a, b) -> float:
    a, b = BD._np(a) > 0.5, BD._np(b) > 0.5
    u = float((a | b).sum())
    return float((a & b).sum()) / u if u else 1.0
