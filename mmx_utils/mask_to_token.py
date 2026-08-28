"""Pixel mask -> H3 video-latent mask (video side only)."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .h3_grid import h3_pixel_frame_groups, snap_frame_count


def _spatial_downsample_max(mask: torch.Tensor, factor: int) -> torch.Tensor:
    """[T,H,W] -> [T, H//factor, W//factor] via max pool."""
    if factor <= 1:
        return mask
    x = mask.unsqueeze(1)
    h, w = x.shape[-2], x.shape[-1]
    pad_h = (-h) % factor
    pad_w = (-w) % factor
    if pad_h or pad_w:
        x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
    x = F.max_pool2d(x, factor, stride=factor)
    return x[:, 0, : (h + pad_h) // factor, : (w + pad_w) // factor]


def _token_snap_2x2(mask: torch.Tensor) -> torch.Tensor:
    """Max-unify each 2x2 latent patch block (matches core DiT patch)."""
    t, h, w = mask.shape
    x = mask.unsqueeze(1)
    pad_h = (-h) % 2
    pad_w = (-w) % 2
    if pad_h or pad_w:
        x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
    x = F.max_pool2d(x, 2, stride=2)
    x = x.repeat_interleave(2, dim=-2).repeat_interleave(2, dim=-1)
    return x[:, 0, :h, :w]


def _temporal_max_groups(mask: torch.Tensor, frame_count: int) -> torch.Tensor:
    """Reduce pixel-frame masks to video-latent temporal dimension."""
    groups = h3_pixel_frame_groups(frame_count)
    pooled = []
    for start, end in groups:
        seg = mask[start:end]
        pooled.append(seg.amax(0) if seg.shape[0] > 0 else mask[0] * 0.0)
    return torch.stack(pooled, dim=0)


def quantize_soft_mask(mask: torch.Tensor) -> torch.Tensor:
    return torch.ceil(mask.clamp(0, 1) * 256.0) / 256.0


def pixel_mask_to_h3_latent(
    mask: torch.Tensor,
    *,
    width: int,
    height: int,
    frame_count: int,
    spatial_dilate: int = 0,
    temporal_clean: int = 0,
    denoise_strength: float = 1.0,
    quantize: bool = True,
) -> torch.Tensor:
    """Return video-side mask [T_lat, H//16, W//16].

    Pre-applying 2x2 max-pool and ceil(m*256)/256 is idempotent with the sampler's
    ``model_base._pool_masks_to_token_grid`` and ``_token_grid_masks`` — no double-shrink.
    """
    fc = snap_frame_count(frame_count)
    x = mask.float()
    if x.ndim == 2:
        x = x.unsqueeze(0)
    if x.shape[0] < fc:
        pad = x[-1:].expand(fc - x.shape[0], -1, -1)
        x = torch.cat([x, pad], dim=0)
    elif x.shape[0] > fc:
        x = x[:fc]

    if temporal_clean > 0:
        from .mask_clean import temporal_despeckle
        x = temporal_despeckle(x, temporal_clean)

    if spatial_dilate > 0:
        from .mask_clean import spatial_dilate as _dilate
        x = _dilate(x, spatial_dilate)

    vae_factor = 16
    lat_h = max(1, height // vae_factor)
    lat_w = max(1, width // vae_factor)
    x = _spatial_downsample_max(x, vae_factor)
    if x.shape[-2] != lat_h or x.shape[-1] != lat_w:
        x = F.interpolate(x.unsqueeze(1), size=(lat_h, lat_w), mode="nearest-exact").squeeze(1)

    x = _temporal_max_groups(x, fc)
    x = _token_snap_2x2(x)

    if denoise_strength != 1.0:
        x = (x * float(denoise_strength)).clamp(0, 1)

    if quantize:
        x = quantize_soft_mask(x)

    return x


def token_preview_upsample(latent_mask: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Upsample latent mask to pixel size for visualization."""
    t = latent_mask.shape[0]
    fc = snap_frame_count(max(t * 4, 5))
    groups = h3_pixel_frame_groups(fc)
    out = torch.zeros((fc, height, width), dtype=latent_mask.dtype, device=latent_mask.device)
    for li, (start, end) in enumerate(groups):
        if li >= t:
            break
        val = latent_mask[li]
        up = F.interpolate(
            val.unsqueeze(0).unsqueeze(0),
            size=(height // 16 * 16, width // 16 * 16),
            mode="nearest-exact",
        ).squeeze()
        out[start:end] = up[:height, :width]
    return out
