"""Head Size Mode for character replacement (owner decision 2026-10-13, C2C D8.21).

Two modes, chosen per job:

  match character proportions (default)  the new character's own head-to-body ratio from the reference, placed
                                          at the original's chin / neck
  keep original composition               the original performer's head size and position

The mode decides how much room the crop and the regeneration mask leave for the new head. Without it every
geometric signal of a crop -> mask -> regenerate -> uncrop replacement is the ORIGINAL performer's, and a head
larger than ~1.15x the original cannot be drawn at its own size (measured: C2C docs/research/
l820_minimax_character_replacement.md). Scaling is always uniform about the chin point - never a stretch.

Pure CPU geometry; poses are COCO-WholeBody [T, 135, 3] pixels from ``pose_interop.pose_keypoint_to_wholebody``.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from .swap_regions import FACE

MODES = ("match character proportions", "keep original composition")
SCALE_MIN, SCALE_MAX = 0.5, 2.0
GATE = 0.3                                  # keypoint confidence below this is "not seen"

# COCO-17 body slots in the WholeBody array
_L_SHOULDER, _R_SHOULDER = 5, 6
# face block (dlib-68 order inside COCO-WholeBody)
_JAW_L, _JAW_R, _CHIN = FACE[0] + 0, FACE[0] + 16, FACE[0] + 8
_BROWS = range(FACE[0] + 17, FACE[0] + 27)

HEAD_W_PER_FACE_W = 1.1                     # head (with hair / ears) vs jaw-to-jaw width
HEAD_H_PER_BROW_CHIN = 1.5                  # chin-to-crown vs chin-to-brow


def _pt(kp: np.ndarray, i: int) -> Optional[np.ndarray]:
    return kp[i, :2] if kp[i, 2] >= GATE else None


def face_metrics(kp: np.ndarray) -> dict:
    """One frame: face width / height, shoulder width, chin point, face centre x (None where not seen)."""
    out = {"face_w": None, "face_h": None, "shoulder_w": None, "chin": None, "cx": None}
    jl, jr, chin = _pt(kp, _JAW_L), _pt(kp, _JAW_R), _pt(kp, _CHIN)
    brows = [kp[i, :2] for i in _BROWS if kp[i, 2] >= GATE]
    if jl is not None and jr is not None:
        out["face_w"] = float(np.linalg.norm(jr - jl))
        out["cx"] = float((jl[0] + jr[0]) / 2)
    if chin is not None:
        out["chin"] = (float(chin[0]), float(chin[1]))
        if brows:
            out["face_h"] = float(chin[1] - np.mean([b[1] for b in brows]))
    ls, rs = _pt(kp, _L_SHOULDER), _pt(kp, _R_SHOULDER)
    if ls is not None and rs is not None:
        out["shoulder_w"] = float(np.linalg.norm(ls - rs))
    return out


def _ratio(m: dict) -> Optional[float]:
    if m["face_w"] and m["shoulder_w"] and m["shoulder_w"] > 1e-3:
        return m["face_w"] / m["shoulder_w"]
    return None


def head_scale_from_poses(original: Optional[np.ndarray], reference: Optional[np.ndarray]) -> tuple[float, str]:
    """(scale, why). Ratio of ratios: (ref face w / ref shoulder w) / median (orig face w / orig shoulder w), so the
    images' resolution and framing cancel. 1.0 with the reason when a pose or its points are missing."""
    if original is None or reference is None:
        return 1.0, "no poses connected (original_pose and reference_pose): head scale 1.0"
    orig = [r for r in (_ratio(face_metrics(f)) for f in original) if r]
    if not orig:
        return 1.0, "the original pose shows no face width and shoulders together: head scale 1.0"
    ref = [r for r in (_ratio(face_metrics(f)) for f in reference) if r]
    if not ref:
        return 1.0, ("the reference pose shows no face width and shoulders together (a head-only photo?): head "
                     "scale 1.0 - set head_scale by hand")
    s = float(np.median(ref)) / float(np.median(orig))
    clamped = min(SCALE_MAX, max(SCALE_MIN, s))
    why = f"head scale {clamped:.3f} from the poses (reference face/shoulder {np.median(ref):.3f}, original {np.median(orig):.3f})"
    if clamped != s:
        why += f"; {s:.3f} clamped to [{SCALE_MIN}, {SCALE_MAX}]"
    return clamped, why


def resolve_scale(mode: str, head_scale: float, original, reference) -> tuple[float, str]:
    if mode == MODES[1]:
        return 1.0, "keep original composition: head scale 1.0"
    if head_scale and head_scale > 0:
        s = min(SCALE_MAX, max(SCALE_MIN, float(head_scale)))
        return s, f"head scale {s:.3f} (set by hand)"
    return head_scale_from_poses(original, reference)


def _ellipse(h: int, w: int, cx: float, cy: float, rx: float, ry: float) -> torch.Tensor:
    yy = torch.arange(h, dtype=torch.float32).view(-1, 1)
    xx = torch.arange(w, dtype=torch.float32).view(1, -1)
    return (((xx - cx) / max(rx, 0.5)) ** 2 + ((yy - cy) / max(ry, 0.5)) ** 2 <= 1.0).float()


def _scale_about(mask_hw: torch.Tensor, ax: float, ay: float, s: float) -> torch.Tensor:
    """Uniform scale of a mask about (ax, ay) - nearest sampling, so it stays binary."""
    if abs(s - 1.0) < 1e-6:
        return mask_hw
    h, w = mask_hw.shape
    yy = torch.arange(h, dtype=torch.float32).view(-1, 1).expand(h, w)
    xx = torch.arange(w, dtype=torch.float32).view(1, -1).expand(h, w)
    sx = ax + (xx - ax) / s                                   # output pixel <- source pixel
    sy = ay + (yy - ay) / s
    grid = torch.stack([(sx + 0.5) / w * 2 - 1, (sy + 0.5) / h * 2 - 1], -1)[None]
    return F.grid_sample(mask_hw[None, None].float(), grid, mode="nearest", padding_mode="zeros",
                         align_corners=False)[0, 0]


def target_heads(masks: torch.Tensor, original: Optional[np.ndarray], scale: float,
                 threshold: float = 0.1) -> tuple[torch.Tensor, dict]:
    """Per frame, where the new head will be: the original head (an ellipse from the face points, or the mask
    itself without them) scaled uniformly by ``scale`` about the chin. [T, H, W] binary, plus facts for the report."""
    T, H, W = masks.shape
    out = torch.zeros(T, H, W)
    used_face = 0
    heights = []
    for t in range(T):
        m = face_metrics(original[min(t, len(original) - 1)]) if original is not None else None
        if m and m["face_w"] and m["face_h"] and m["chin"] and m["face_h"] > 0:
            hw, hh = HEAD_W_PER_FACE_W * m["face_w"], HEAD_H_PER_BROW_CHIN * m["face_h"]
            cx, chin_y = (m["cx"] if m["cx"] is not None else m["chin"][0]), m["chin"][1]
            head = _ellipse(H, W, cx, chin_y - hh / 2, hw / 2, hh / 2)
            out[t] = _scale_about(head, cx, chin_y, scale)
            heights.append(hh * scale)
            used_face += 1
        else:
            mt = masks[t] > threshold
            if not bool(mt.any()):
                continue
            ys, xs = torch.nonzero(mt, as_tuple=True)
            cx, chin_y = float(xs.float().mean()), float(ys.max())  # the mask's bottom centre stands in for the chin
            out[t] = _scale_about(mt.float(), cx, chin_y, scale)
            heights.append(float(ys.max() - ys.min() + 1) * scale)
    facts = {"frames_with_face_points": used_face, "frames": T,
             "target_head_height_px": float(np.median(heights)) if heights else 0.0}
    return out, facts


def _dilate(m: torch.Tensor, px: int) -> torch.Tensor:
    if px <= 0:
        return m
    k = 2 * px + 1
    x = F.max_pool2d(m[:, None].float(), kernel_size=(k, 1), stride=1, padding=(px, 0))
    return F.max_pool2d(x, kernel_size=(1, k), stride=1, padding=(0, px))[:, 0]


def head_room(masks: torch.Tensor, mode: str, head_scale: float, original, reference,
              threshold: float = 0.1) -> tuple[torch.Tensor, torch.Tensor, str]:
    """(planning masks, head-room masks, report line).

    planning masks = mask OR target head: what the crop planner must keep inside its window.
    head-room masks = the planning masks dilated by max(4, 10 % of the target head height) px: the region to
    regenerate and to composite back, so a soft edge still leaves the head itself fully regenerated."""
    scale, why = resolve_scale(mode, head_scale, original, reference)
    heads, facts = target_heads(masks, original, scale, threshold)
    if not facts["frames_with_face_points"] and abs(scale - 1.0) > 1e-6:
        why += "; no face points, so the whole mask is scaled as the head (connect original_pose for a body mask)"
    union = torch.maximum((masks > threshold).float().cpu(), heads)
    pad = max(4, int(round(0.10 * facts["target_head_height_px"])))
    room = _dilate(union, pad).clamp(0, 1)
    line = (f"head size: {mode}; {why}; target head ~{facts['target_head_height_px']:.0f} px high; "
            f"face points on {facts['frames_with_face_points']}/{facts['frames']} frames; mask padding {pad} px")
    return union, room, line

