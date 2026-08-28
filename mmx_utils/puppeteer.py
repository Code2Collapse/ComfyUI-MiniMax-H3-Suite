"""Pose puppeteer — package driving performance as conditioning hints (catalog #61)."""

from __future__ import annotations

import json

import numpy as np
import torch
import torch.nn.functional as F

from .keypoint_spine import render_pose_hints

RIG_TYPES = ("human_to_creature", "stick_figure", "mocap_data")


def _resize_batch(img: torch.Tensor, h: int, w: int) -> torch.Tensor:
    x = img.float()
    if x.ndim == 3:
        x = x.unsqueeze(0)
    if x.shape[1:3] == (h, w):
        return x
    nchw = x.permute(0, 3, 1, 2)
    out = F.interpolate(nchw, size=(h, w), mode="bilinear", align_corners=False)
    return out.permute(0, 2, 3, 1).contiguous()


def _stick_overlay(base: torch.Tensor, pose: torch.Tensor, alpha: float = 0.65) -> torch.Tensor:
    """Deterministic blend — pose edges emphasized on character plate."""
    edges = pose[..., :3].mean(dim=-1, keepdim=True)
    edge_mask = (edges > 0.15).float()
    mixed = base * (1.0 - alpha * edge_mask) + pose[..., :3] * (alpha * edge_mask)
    return mixed.clamp(0, 1)


def _keypoints_json_to_array(keypoints_json: str) -> np.ndarray | None:
    try:
        data = json.loads(keypoints_json)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict) or "frames" not in data:
        return None
    frames = data["frames"]
    if not isinstance(frames, list) or not frames:
        return None
    rows: list[np.ndarray] = []
    max_k = 0
    for frame in frames:
        if not isinstance(frame, dict):
            continue
        kps = frame.get("keypoints")
        if not isinstance(kps, list) or not kps:
            continue
        pts = []
        for kp in kps:
            if isinstance(kp, dict):
                pts.append([float(kp.get("x", 0)), float(kp.get("y", 0)), float(kp.get("c", 1))])
            elif isinstance(kp, (list, tuple)) and len(kp) >= 2:
                conf = float(kp[2]) if len(kp) > 2 else 1.0
                pts.append([float(kp[0]), float(kp[1]), conf])
        if pts:
            rows.append(np.asarray(pts, dtype=np.float32))
            max_k = max(max_k, len(pts))
    if not rows:
        return None
    out = np.zeros((len(rows), max_k, 3), dtype=np.float32)
    for i, row in enumerate(rows):
        out[i, : row.shape[0], :] = row
    return out


def build_motion_text(
    *,
    rig_type: str,
    frame_count: int,
    keypoints_json: str | None = None,
) -> str:
    """Structured camera/motion description — catalog #61."""
    rig = (rig_type or "human_to_creature").lower()
    kp_note = "keypoints attached" if keypoints_json and keypoints_json.strip() not in ("", "{}") else "keypoints from pose image only"
    return (
        f"Motion package (catalog #61)\n"
        f"Rig: {rig}\n"
        f"Camera: static plate; motion transferred from driving pose\n"
        f"Frames: {int(frame_count)}\n"
        f"Keypoints: {kp_note}\n"
        f"Union: blocked (D1) — conditioning hints only, no Union apply path\n"
        f"Use: wire motion_hints as control conditioning; do not expect Union injection."
    )


def render_motion_hints(
    driving_pose: torch.Tensor,
    character_ref: torch.Tensor,
    *,
    rig_type: str = "human_to_creature",
    keypoints_json: str | None = None,
) -> tuple[torch.Tensor, str]:
    """Deterministic motion_hints [T,H,W,3] + motion_text."""
    pose = driving_pose.float()
    ref = character_ref.float()
    if pose.ndim == 3:
        pose = pose.unsqueeze(0)
    if ref.ndim == 3:
        ref = ref.unsqueeze(0)
    t = max(pose.shape[0], ref.shape[0])
    if pose.shape[0] == 1 and t > 1:
        pose = pose.expand(t, -1, -1, -1)
    if ref.shape[0] == 1 and t > 1:
        ref = ref.expand(t, -1, -1, -1)
    t = min(t, pose.shape[0], ref.shape[0])
    h, w = ref.shape[1], ref.shape[2]
    pose = _resize_batch(pose[:t], h, w)
    ref = _resize_batch(ref[:t], h, w)

    rig = (rig_type or "human_to_creature").lower()
    if rig == "stick_figure":
        hints = _stick_overlay(ref, pose, alpha=0.75)
    elif rig == "mocap_data":
        hints = (0.35 * ref + 0.65 * pose).clamp(0, 1)
    else:
        hints = (0.55 * ref + 0.45 * pose).clamp(0, 1)

    kps = _keypoints_json_to_array(keypoints_json) if keypoints_json else None
    if kps is not None:
        kps_t = min(kps.shape[0], t)
        skeleton = render_pose_hints(kps[:kps_t], (w, h)).to(hints.device, dtype=hints.dtype)
        if skeleton.shape[0] < hints.shape[0]:
            pad = hints.shape[0] - skeleton.shape[0]
            skeleton = torch.cat([skeleton, skeleton[-1:].expand(pad, -1, -1, -1)], dim=0)
        skel_rgb = skeleton[..., :3]
        skel_mask = (skel_rgb.max(dim=-1, keepdim=True).values > 0.05).float()
        hints = hints * (1.0 - 0.7 * skel_mask) + skel_rgb * (0.7 * skel_mask)
        hints = hints.clamp(0, 1)

    text = build_motion_text(rig_type=rig, frame_count=t, keypoints_json=keypoints_json)
    return hints.to(driving_pose.dtype), text
