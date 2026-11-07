# MIT License — ComfyUI-NKD-Basic-Tools (mask_core.py)
# PORTED FROM: ComfyUI-NKD-Basic-Tools :: mask_core.py @ HEAD

from __future__ import annotations

import torch
import torch.nn.functional as F


def _temporal(x: torch.Tensor, frames: int, op: str) -> torch.Tensor:
    if frames <= 0 or x.shape[0] < 2:
        return x
    if op == "max":
        # A24 L9.165 (measured): the max over a +-frames window with the clip's end frames repeated (= the replicate
        # pad + max-pool below), as a running elementwise max of shifted frames: identical values, without torch's
        # slow CPU max-pool over a (B, H*W) image (4.2 s for 24 x 1080p at radius 2).
        b = x.shape[0]
        idx = torch.arange(b, device=x.device)
        out = x.clone()
        for d in range(1, int(frames) + 1):
            torch.maximum(out, x[(idx + d).clamp(max=b - 1)], out=out)
            torch.maximum(out, x[(idx - d).clamp(min=0)], out=out)
        return out
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


def _dilate_cv2(mask: torch.Tensor, radius: int):
    """CPU fast path (A24 L9.165, measured): OpenCV's float dilate with the same 3x3 cross, `radius` iterations and
    a -inf border equals the torch loop below bit for bit, 9-15x faster at 24 x 1080p (docs/RUST_ACCELERATION_AUDIT.md).
    None when it does not apply (GPU tensor, no OpenCV, not [B,H,W] float)."""
    if mask.device.type != "cpu" or mask.ndim != 3 or not mask.is_floating_point():
        return None
    try:
        import cv2
        import numpy as np
    except Exception:  # noqa: BLE001 - OpenCV is optional here: the torch path below is the reference
        return None
    k = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    arr = mask.detach().to(torch.float32).contiguous().numpy()
    out = np.empty_like(arr)
    for i in range(arr.shape[0]):
        out[i] = cv2.dilate(arr[i], k, iterations=int(radius), borderType=cv2.BORDER_CONSTANT,
                            borderValue=-float("inf"))
    return torch.from_numpy(out).to(mask.dtype)


def spatial_dilate(mask: torch.Tensor, radius: int) -> torch.Tensor:
    if radius <= 0:
        return mask
    fast = _dilate_cv2(mask, radius)
    if fast is not None:
        return fast
    x = mask.unsqueeze(1) if mask.ndim == 3 else mask
    for _ in range(int(radius)):
        x = torch.maximum(
            F.max_pool2d(x, (3, 1), stride=1, padding=(1, 0)),
            F.max_pool2d(x, (1, 3), stride=1, padding=(0, 1)),
        )
    return x.squeeze(1) if mask.ndim == 3 else x[:, 0]
