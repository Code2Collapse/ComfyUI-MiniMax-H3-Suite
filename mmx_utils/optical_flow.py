# Apache-2.0 — ComfyUI-WanNodeExperiments
# PORTED FROM: ComfyUI-WanNodeExperiments :: nodes/flow/flow_core.py @ HEAD

"""RAFT optical flow — CPU-default, lazy weight load."""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as F

_RAFT_CACHE: dict[str, torch.nn.Module] = {}
RaftSize = Literal["large", "small"]


def _round_to_multiple(x: int, m: int = 8) -> int:
    return max(m, int(round(x / m)) * m)


def load_raft(size: RaftSize = "small", device: torch.device | str = "cpu") -> torch.nn.Module:
    key = f"{size}:{str(device)}"
    cached = _RAFT_CACHE.get(key)
    if cached is not None:
        return cached
    from torchvision.models.optical_flow import (
        Raft_Large_Weights,
        Raft_Small_Weights,
        raft_large,
        raft_small,
    )

    if size == "small":
        model = raft_small(weights=Raft_Small_Weights.DEFAULT, progress=False)
    else:
        model = raft_large(weights=Raft_Large_Weights.DEFAULT, progress=False)
    model = model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    _RAFT_CACHE[key] = model
    return model


# torchvision's RAFT downsamples by 8 and its correlation pyramid needs feature maps of at
# least 16x16, so ANY input below 128x128 dies inside the model with:
#   ValueError: Feature maps are too small to be down-sampled by the correlation pyramid.
#   H and W of feature maps should be at least 16; got: torch.Size([4, 4]).
# (torchvision/models/optical_flow/raft.py:378)
# Crops in this pack are routinely small, so upscaling to the floor here is the difference
# between DriftQC working on a 96px crop and handing the artist raw torchvision internals.
RAFT_MIN_SIDE = 128


def _prep_for_raft(frames_bchw: torch.Tensor) -> tuple[torch.Tensor, int, int]:
    n, _c, h, w = frames_bchw.shape
    th, tw = _round_to_multiple(h), _round_to_multiple(w)
    # enforce RAFT's hard floor AFTER the multiple-of-8 rounding, so the result stays legal on both
    th, tw = max(th, RAFT_MIN_SIDE), max(tw, RAFT_MIN_SIDE)
    x = frames_bchw
    if (th, tw) != (h, w):
        x = F.interpolate(x, size=(th, tw), mode="bilinear", align_corners=False)
    x = (x * 2.0 - 1.0).clamp(-1.0, 1.0)
    return x.contiguous(), h, w


@torch.no_grad()
def compute_flow(
    frames: torch.Tensor,
    *,
    size: RaftSize = "small",
    direction: Literal["forward", "backward"] = "forward",
    iters: int = 12,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """frames [T,H,W,3] → flow [T-1,2,H,W] in pixels."""
    assert frames.ndim == 4 and frames.shape[-1] == 3
    t, h, w, _ = frames.shape
    dev = torch.device(device) if device is not None else frames.device
    if t <= 1:
        return frames.new_zeros((0, 2, h, w))

    fr = frames.permute(0, 3, 1, 2).to(dev)
    img1, img2 = (fr[:-1], fr[1:]) if direction == "forward" else (fr[1:], fr[:-1])
    x1, _, _ = _prep_for_raft(img1)
    x2, _, _ = _prep_for_raft(img2)
    model = load_raft(size, dev)
    flow = model(x1, x2, num_flow_updates=int(iters))[-1]
    _, _, th, tw = flow.shape
    if (th, tw) != (h, w):
        flow = F.interpolate(flow, size=(h, w), mode="bilinear", align_corners=False)
        flow[:, 0] *= w / tw
        flow[:, 1] *= h / th
    return flow.contiguous()
