# Apache-2.0 — ComfyUI-WanNodeExperiments + ComfyUI-CustomNodePacks
# PORTED FROM: ComfyUI-WanNodeExperiments :: flow_core.py + CustomNodePacks :: _control_backends.py @ HEAD

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from .optical_flow import compute_flow


@dataclass(frozen=True)
class DriftQCResult:
    passed: bool
    drift_per_frame: list[float]
    drift_heatmap: torch.Tensor
    report: str


def exterior_mask_from_edit(edit_mask: torch.Tensor) -> torch.Tensor:
    """1 outside edit region, 0 inside — [B,H,W] float."""
    m = edit_mask.float()
    if m.ndim == 4:
        m = m[:, :, :, 0] if m.shape[-1] == 1 else m.mean(dim=-1)
    if m.ndim == 3 and m.shape[0] == 1:
        m = m[0]
    if m.ndim == 2:
        m = m.unsqueeze(0)
    return (1.0 - m.clamp(0, 1)).contiguous()


def inject_pixel_shift(image: torch.Tensor, dx: float, dy: float) -> torch.Tensor:
    """Shift IMAGE [1,H,W,3] by (dx,dy) pixels via grid_sample."""
    if image.ndim == 3:
        image = image.unsqueeze(0)
    _b, h, w, _c = image.shape
    x = image.permute(0, 3, 1, 2)
    theta = torch.tensor(
        [[[1.0, 0.0, 2.0 * float(dx) / max(w - 1, 1)],
          [0.0, 1.0, 2.0 * float(dy) / max(h - 1, 1)]]],
        dtype=torch.float32,
        device=x.device,
    )
    grid = F.affine_grid(theta, x.shape, align_corners=True)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=True).permute(
        0, 2, 3, 1
    )


def measure_exterior_drift(
    original: torch.Tensor,
    edited: torch.Tensor,
    edit_mask: torch.Tensor,
    *,
    threshold_px: float = 2.0,
    raft_size: str = "small",
    supersample: int = 2,
) -> DriftQCResult:
    """
  Compare original vs edited using RAFT on exterior pixels only.
  original/edited: [T,H,W,3]. edit_mask: [T,H,W] or [H,W], 1=inside edit.

  `supersample` exists because RAFT's sub-pixel accuracy scales with pixels-per-unit-
  displacement, and this node is asked to resolve drifts BELOW one pixel.

  Measured on a 192x192 synthetic pair with supersample=1 (i.e. RAFT at native size):
      injected 0.0 -> 0.0655   (err 0.066)
      injected 0.5 -> 0.2800   (err 0.220)
      injected 1.0 -> 0.7285   (err 0.272)  <-- misses the +/-0.25 gate
      injected 2.0 -> 1.6848   (err 0.315)  <-- misses the +/-0.25 gate
  RAFT-small systematically UNDER-reports small displacements at native resolution.

  This was hidden for a while because the unit test used 64x64 frames: `_prep_for_raft`
  upscales anything under RAFT_MIN_SIDE to 128, so the test was accidentally getting a 2x
  supersample and passing, while real crops (128-512px) got none and failed. Making the
  supersample explicit means the calibration gate covers the ACTUAL operating regime
  instead of an artefact of the test's frame size.
    """
    if original.ndim == 3:
        original = original.unsqueeze(0)
    if edited.ndim == 3:
        edited = edited.unsqueeze(0)
    t = min(original.shape[0], edited.shape[0])
    original = original[:t]
    edited = edited[:t]

    ext = exterior_mask_from_edit(edit_mask)
    if ext.ndim == 2:
        ext = ext.unsqueeze(0).expand(t, -1, -1)
    elif ext.shape[0] == 1 and t > 1:
        ext = ext.expand(t, -1, -1)
    ext = ext[:t]

    # Follow the CALLER's device — do not hardcode CPU here.
    # Hardcoding cpu had two bugs: (1) `pair` went to cpu while `ext` (derived from the
    # caller's mask) stayed on cuda, so `mag * m` below raised
    #   RuntimeError: Expected all tensors to be on the same device ... cuda:0 and cpu
    # and (2) it made run_with_cpu_fallback a no-op — the work was never on the accelerator,
    # so there was nothing to fall back FROM. The node picks the device; this honours it.
    dev = original.device
    ext = ext.to(dev)
    per_frame: list[float] = []
    heatmaps: list[torch.Tensor] = []

    ss = max(1, int(supersample))
    for i in range(t):
        pair = torch.stack([original[i], edited[i]], dim=0).to(dev)
        if ss > 1:
            # Upscale BOTH frames identically, so a sub-pixel shift becomes a
            # multi-pixel one that RAFT can actually resolve. compute_flow returns
            # flow in the pixels of what it was given, so divide by ss to get back
            # to source pixels. Bilinear is fine here: we are measuring displacement,
            # not reconstructing detail.
            up = pair.permute(0, 3, 1, 2)
            up = F.interpolate(up, scale_factor=ss, mode="bilinear", align_corners=False)
            pair = up.permute(0, 2, 3, 1).contiguous()
        flow = compute_flow(pair, size=raft_size, device=dev)
        if flow.shape[0] and ss > 1:
            flow = F.interpolate(flow, size=ext.shape[-2:], mode="bilinear", align_corners=False) / float(ss)
        mag = torch.sqrt(flow[0, 0] ** 2 + flow[0, 1] ** 2) if flow.shape[0] else torch.zeros_like(ext[i])
        m = ext[i]
        denom = float(m.sum().clamp(min=1.0))
        drift = float((mag * m).sum() / denom)
        per_frame.append(drift)
        hm = (mag * m).unsqueeze(-1).repeat(1, 1, 3)
        heatmaps.append(hm)

    heat = torch.stack(heatmaps, dim=0).to(original.dtype)
    peak = max(per_frame) if per_frame else 0.0
    passed = peak <= float(threshold_px)
    report = (
        f"exterior drift peak={peak:.4f}px threshold={threshold_px:g}px "
        f"frames={t} raft={raft_size} — {'PASS' if passed else 'FAIL'}"
    )
    return DriftQCResult(
        passed=passed,
        drift_per_frame=per_frame,
        drift_heatmap=heat,
        report=report,
    )


def drift_per_frame_json(values: list[float]) -> str:
    return json.dumps({"drift_px": values}, separators=(",", ":"))
