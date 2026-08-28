"""Pure AV-latent logic for N4 — testable without VAE weights or GPU."""

from __future__ import annotations

from typing import Any, Protocol, Sequence

import torch
import torch.nn.functional as F

from .h3_constants import CANVAS_MULTIPLE, FPS, AUDIO_LATENT_FPS
from .h3_grid import is_h3_compatible, snap_frame_count


class NestedLike(Protocol):
    def unbind(self) -> Sequence[torch.Tensor]: ...


def assert_nested_samples(samples: Any) -> list[torch.Tensor]:
    if samples is None:
        raise ValueError('LATENT is missing "samples". Feed a MiniMax H3 AV latent.')
    if not hasattr(samples, "unbind"):
        raise ValueError(
            "Expected a MiniMax H3 joint AV latent (NestedTensor). Feed the LATENT output "
            "of MiniMaxH3ReferenceToVideo / EmptyMiniMaxH3LatentAV."
        )
    members = list(samples.unbind())
    if len(members) < 2:
        raise ValueError(
            f"AV latent NestedTensor has {len(members)} stream(s); expected video+audio pair."
        )
    return members


def validate_h3_pixel_frame_count(frame_count: int) -> int:
    n = int(frame_count)
    if not is_h3_compatible(n):
        snapped = snap_frame_count(n)
        raise ValueError(
            f"Frame count {n} is off H3's 17n+5 grid (remainder {n % 17}, expected 5). "
            f"Snap to {snapped} with H3 Frame Handles or N1 before sampling."
        )
    return n


def validate_canvas_multiple(width: int, height: int) -> None:
    for label, val in (("width", width), ("height", height)):
        if int(val) % CANVAS_MULTIPLE != 0:
            raise ValueError(
                f"Canvas {label}={val} is not a multiple of {CANVAS_MULTIPLE}. "
                "Wire canvas size from the crop planner — do not type arbitrary values."
            )


def normalize_encoded_video(encoded: torch.Tensor) -> torch.Tensor:
    """[B,C,H,W] or [B,C,T,H,W] -> [B,C,T,H,W]."""
    if encoded.ndim == 4:
        return encoded.unsqueeze(0).movedim(1, 2)
    if encoded.ndim == 5:
        return encoded
    raise ValueError(f"encoded video latent must be 4D or 5D, got shape {tuple(encoded.shape)}")


def validate_spatial_latent(
    encoded: torch.Tensor,
    video_tmpl: torch.Tensor,
) -> None:
    _, _, _, got_h, got_w = normalize_encoded_video(encoded).shape
    tgt_h, tgt_w = int(video_tmpl.shape[-2]), int(video_tmpl.shape[-1])
    if (got_h, got_w) != (tgt_h, tgt_w):
        raise ValueError(
            f"Spatial latent mismatch: encoded {got_h}x{got_w} but the AV latent expects "
            f"{tgt_h}x{tgt_w}. The crop canvas and the H3 node's width/height must match "
            f"(both are pixels/16)."
        )


def fit_temporal_dim(
    encoded: torch.Tensor,
    video_tmpl: torch.Tensor,
    *,
    allow_trim_pad: bool = True,
) -> tuple[torch.Tensor, str]:
    enc = normalize_encoded_video(encoded)
    tgt_t = int(video_tmpl.shape[-3])
    got_t = int(enc.shape[-3])
    if got_t == tgt_t:
        return enc, ""
    if not allow_trim_pad:
        raise ValueError(f"Temporal latent mismatch: encoded t={got_t} vs latent t={tgt_t}.")
    if got_t > tgt_t:
        enc = enc[..., :tgt_t, :, :]
        action = "trimmed"
    else:
        pad = video_tmpl[..., : tgt_t - got_t, :, :].to(enc.device, enc.dtype)
        enc = torch.cat([enc, pad], dim=-3)
        action = "padded"
    note = (
        f"WARNING temporal mismatch: encoded t={got_t} vs latent t={tgt_t} -> {action}. "
        f"Frame count is probably off H3's 17k+5 grid."
    )
    return enc, note


def inject_video_stream(
    members: list[torch.Tensor],
    encoded: torch.Tensor,
    video_tmpl: torch.Tensor,
) -> tuple[list[torch.Tensor], str]:
    enc, note = fit_temporal_dim(encoded, video_tmpl)
    validate_spatial_latent(enc, video_tmpl)
    out = list(members)
    out[0] = enc.to(video_tmpl.device, video_tmpl.dtype)
    return out, note


def fit_audio_latent_length(encoded_audio: torch.Tensor, target_t: int) -> torch.Tensor:
    """Trim or pad audio latent along last dim to *target_t*."""
    z = encoded_audio
    got = int(z.shape[-1])
    if got == target_t:
        return z
    if got > target_t:
        return z[..., :target_t]
    pad = torch.zeros(
        *z.shape[:-1],
        target_t - got,
        device=z.device,
        dtype=z.dtype,
    )
    return torch.cat([z, pad], dim=-1)


def inject_audio_stream(
    members: list[torch.Tensor],
    encoded_audio: torch.Tensor,
) -> list[torch.Tensor]:
    audio_tmpl = members[1]
    z = fit_audio_latent_length(encoded_audio, int(audio_tmpl.shape[-1]))
    out = list(members)
    out[1] = z.to(audio_tmpl.device, audio_tmpl.dtype)
    return out


def expand_video_latent_mask(
    latent_mask: torch.Tensor,
    video: torch.Tensor,
) -> torch.Tensor:
    """Expand N2 mask [T_lat,H,W] to video noise_mask [B,C,T,H,W]."""
    m = latent_mask.float()
    if m.ndim == 2:
        m = m.unsqueeze(0)
    if m.ndim == 4:
        m = m[:, 0]
    t_lat = int(video.shape[-3])
    if m.shape[0] != t_lat:
        m = F.interpolate(
            m.unsqueeze(0).unsqueeze(0),
            size=(t_lat, m.shape[-2], m.shape[-1]),
            mode="nearest",
        ).squeeze(0).squeeze(0)
    m = m.view(1, 1, t_lat, m.shape[-2], m.shape[-1]).to(video.device, torch.float32)
    m = m.expand(video.shape[0], 1, t_lat, m.shape[-2], m.shape[-1])
    m = m.expand(-1, video.shape[1], -1, -1, -1).contiguous()
    return m.to(video.dtype)


def build_av_noise_mask(
    video_mask: torch.Tensor,
    video: torch.Tensor,
    audio: torch.Tensor,
    *,
    audio_locked: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    vmask = expand_video_latent_mask(video_mask, video)
    if audio_locked:
        amask = torch.zeros_like(audio)
    else:
        amask = torch.ones_like(audio)
    return vmask, amask


def audio_latent_length_for_frames(frame_count: int) -> int:
    fc = validate_h3_pixel_frame_count(frame_count)
    return round(fc / FPS * AUDIO_LATENT_FPS)


def pixel_frame_count_from_video_latent(video: torch.Tensor) -> int:
    """Best-effort inverse of video_latent_t for validation messages."""
    t_lat = int(video.shape[-3])
    if t_lat <= 2:
        return 5
    return (t_lat - 2) * 17 + 5


def validate_crop_frame_count(crop_frames: int, video: torch.Tensor) -> None:
    validate_h3_pixel_frame_count(int(crop_frames))


def pack_latent_dict(
    av_latent: dict,
    members: list[torch.Tensor],
    noise_mask_members: tuple[torch.Tensor, torch.Tensor] | None,
    nested_ctor: Any,
) -> dict:
    out = dict(av_latent)
    out["samples"] = nested_ctor(tuple(members))
    if noise_mask_members is not None:
        out["noise_mask"] = nested_ctor(noise_mask_members)
    return out
