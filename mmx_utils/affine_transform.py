# MIT License — ComfyUI-H3-FaceRefine
# PORTED FROM: ComfyUI-H3-FaceRefine :: nodes.py @ HEAD

from __future__ import annotations

import torch
import torch.nn.functional as F


def affine_crop(img: torch.Tensor, box: tuple[float, float, float, float], cw: int, ch: int) -> torch.Tensor:
    """Sub-pixel crop+resize. img [1,H,W,C] -> [1,ch,cw,C]."""
    x, y, bw, bh = box
    _, H, W, _C = img.shape
    src = img[..., :3].movedim(-1, 1).float()
    theta = torch.tensor(
        [[[bw / W, 0.0, (2.0 * x + bw) / W - 1.0],
          [0.0, bh / H, (2.0 * y + bh) / H - 1.0]]],
        dtype=torch.float32,
        device=src.device,
    )
    grid = F.affine_grid(theta, (1, 3, int(ch), int(cw)), align_corners=False)
    out = F.grid_sample(src, grid, mode="bilinear", padding_mode="border", align_corners=False)
    return out.movedim(1, -1).to(img.dtype)


def affine_crop_batch(
    images: torch.Tensor,
    boxes: list[tuple[float, float, float, float]],
    cw: int,
    ch: int,
) -> torch.Tensor:
    """Batched affine crop. images [B,H,W,C]."""
    out = []
    for i in range(images.shape[0]):
        box = boxes[i] if i < len(boxes) else boxes[-1]
        out.append(affine_crop(images[i : i + 1], box, cw, ch))
    return torch.cat(out, dim=0)


def inverse_affine_theta(
    boxes: list[tuple[float, float, float, float]],
    src_w: int,
    src_h: int,
    device: torch.device,
) -> torch.Tensor:
    """Forward theta mapping crop canvas -> source frame [B,2,3]."""
    n = len(boxes)
    th = torch.empty((n, 2, 3), dtype=torch.float32, device=device)
    for j, (x, y, bw, bh) in enumerate(boxes):
        th[j, 0, 0] = src_w / bw
        th[j, 0, 1] = 0.0
        th[j, 0, 2] = (src_w - 2.0 * x) / bw - 1.0
        th[j, 1, 0] = 0.0
        th[j, 1, 1] = src_h / bh
        th[j, 1, 2] = (src_h - 2.0 * y) / bh - 1.0
    return th


def inverse_affine_warp_batch(
    crops: torch.Tensor,
    boxes: list[tuple[float, float, float, float]],
    src_w: int,
    src_h: int,
) -> torch.Tensor:
    """Warp crops back to source resolution [B,H,W,C]."""
    dev = crops.device
    n = crops.shape[0]
    th = inverse_affine_theta(boxes[:n], src_w, src_h, dev)
    grid = F.affine_grid(th, (n, 3, int(src_h), int(src_w)), align_corners=False)
    patch = crops[..., :3].to(dev).movedim(-1, 1).float()
    warped = F.grid_sample(patch, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    return warped.movedim(1, -1)
