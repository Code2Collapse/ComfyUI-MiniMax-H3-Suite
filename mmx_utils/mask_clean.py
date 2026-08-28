# MIT License — ComfyUI-NKD-Basic-Tools (mask_core.py)
# PORTED FROM: ComfyUI-NKD-Basic-Tools :: mask_core.py @ HEAD

from __future__ import annotations

import torch
import torch.nn.functional as F


def _temporal(x: torch.Tensor, frames: int, op: str) -> torch.Tensor:
    if frames <= 0 or x.shape[0] < 2:
        return x
    k = 2 * frames + 1
    t = x.reshape(1, 1, x.shape[0], -1)
    t = F.pad(t, (0, 0, frames, frames), mode="replicate")
    if op == "max":
        t = F.max_pool2d(t, (k, 1), stride=1)
    else:
        t = F.avg_pool2d(t, (k, 1), stride=1)
    return t.reshape(x.shape)


def temporal_despeckle(mask: torch.Tensor, radius: int) -> torch.Tensor:
    """Temporal max-pool despeckle on [B,H,W] mask batch."""
    if radius <= 0:
        return mask
    x = mask
    if x.ndim == 2:
        x = x.unsqueeze(0)
    return _temporal(x, int(radius), "max")


def spatial_dilate(mask: torch.Tensor, radius: int) -> torch.Tensor:
    if radius <= 0:
        return mask
    x = mask.unsqueeze(1) if mask.ndim == 3 else mask
    for _ in range(int(radius)):
        x = torch.maximum(
            F.max_pool2d(x, (3, 1), stride=1, padding=(1, 0)),
            F.max_pool2d(x, (1, 3), stride=1, padding=(0, 1)),
        )
    return x.squeeze(1) if mask.ndim == 3 else x[:, 0]
