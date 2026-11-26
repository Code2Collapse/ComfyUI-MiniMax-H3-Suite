# MIT License — ComfyUI-H3-FaceRefine
# PORTED FROM: ComfyUI-H3-FaceRefine :: nodes.py H3PerFrameDenoise @ HEAD

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .transform_types import H3Transform


def _smooth(vals: np.ndarray, window: int, method: str = "gaussian") -> np.ndarray:
    if window <= 1 or len(vals) < 3:
        return vals
    window = min(int(window), len(vals))
    if window % 2 == 0:
        window += 1
    if window < 3:
        return vals
    pad = window // 2
    padded = np.pad(vals, pad, mode="reflect")
    if method == "gaussian":
        x = np.arange(window, dtype=np.float64) - pad
        sigma = max(window / 6.0, 0.5)
        kernel = np.exp(-(x**2) / (2.0 * sigma**2))
        kernel /= kernel.sum()
    else:
        kernel = np.ones(window, dtype=np.float64) / window
    return np.convolve(padded, kernel, mode="valid")


def face_heights_from_transform(transform: H3Transform) -> np.ndarray:
    cf = float(transform.crop_factor) or 3.0
    return np.array([b[3] / cf for b in transform.boxes], dtype=np.float64)


def strength_curve_from_faces(
    face: np.ndarray,
    *,
    strength_small_face: float,
    strength_large_face: float,
    face_px_small: float,
    face_px_large: float,
    gamma: float,
    smooth_frames: int,
    scale_mode: str,
) -> np.ndarray:
    if face.size == 0:
        raise ValueError("transform has no boxes")
    if scale_mode == "relative_to_clip":
        lo, hi = float(face.min()), float(face.max())
    else:
        lo, hi = float(face_px_small), float(face_px_large)
    if hi - lo < 1e-6:
        t = np.zeros_like(face)
    else:
        t = np.clip((face - lo) / (hi - lo), 0.0, 1.0)
    t = np.clip(t, 0.0, 1.0) ** float(gamma)
    strength = strength_small_face + (strength_large_face - strength_small_face) * t
    strength = _smooth(strength, int(smooth_frames), "gaussian")
    return np.clip(strength, 0.0, 1.0)



def _combine(base: torch.Tensor, vmask: torch.Tensor) -> torch.Tensor:
    """The incoming spatial noise mask times the per-frame strength. Masks from Set AV Noise Mask / Hard Keyframes
    are [1,1,T,h,w] and broadcast; they used to be dropped unless the shape matched exactly (L8.28), which
    regenerated the whole frame, background included."""
    try:
        shape = torch.broadcast_shapes(base.shape, vmask.shape)
    except RuntimeError:
        shape = None
    if shape is None or tuple(shape) != tuple(vmask.shape):
        raise ValueError(f"Per-Frame Denoise: the incoming noise mask {tuple(base.shape)} does not fit the video "
                         f"latent {tuple(vmask.shape)}. Make the mask for this latent (same length and size).")
    return (base * vmask).clamp(0, 1)


def _strength_per_step(strength_px, latent_t: int) -> torch.Tensor:
    """Per-pixel-frame strength -> one value per latent step: the mean over the frames that step covers on H3's
    grid (1,4,4,4,4 from frame 0). A curve of another length is resampled linearly as before."""
    from .h3_grid import frames_for_tokens, h3_pixel_frame_groups
    s = torch.as_tensor(np.asarray(strength_px), dtype=torch.float32).reshape(-1)
    if s.numel() == frames_for_tokens(latent_t):
        return torch.stack([s[a:b].mean() for a, b in h3_pixel_frame_groups(s.numel())])
    return F.interpolate(s.view(1, 1, -1), size=latent_t, mode="linear", align_corners=True).reshape(-1)


def merge_per_frame_video_mask(
    av_latent: dict,
    vmask: torch.Tensor,
    video: torch.Tensor,
    audio: torch.Tensor,
) -> tuple[dict, str]:
    """Multiply per-frame strength into existing noise_mask video side; preserve audio."""
    prev = av_latent.get("noise_mask")
    if prev is not None and hasattr(prev, "unbind"):
        pm = list(prev.unbind())
        base = pm[0].to(vmask.device, torch.float32)
        vmask = _combine(base, vmask)
        pm[0] = vmask.to(pm[0].dtype)
        new_mask = type(prev)(tuple(pm))
    else:
        audio_zero = torch.zeros_like(audio)
        try:
            import comfy.nested_tensor as nested_tensor

            new_mask = nested_tensor.NestedTensor((vmask.to(video.dtype), audio_zero))
        except Exception:  # not just ImportError: the comfy_kitchen skew raises AttributeError
            new_mask = (vmask.to(video.dtype), audio_zero)
    out = dict(av_latent)
    out["noise_mask"] = new_mask
    return out, ""


def scale_video_noise_mask(
    video: torch.Tensor,
    prev_video_mask: torch.Tensor | None,
    strength_px: np.ndarray,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build 5-D video noise mask; preserve audio side via caller."""
    latent_t = int(video.shape[-3])
    s = _strength_per_step(strength_px, latent_t).view(1, 1, latent_t, 1, 1).to(video.device, torch.float32)
    vmask = s.expand(video.shape[0], 1, latent_t, video.shape[-2], video.shape[-1])
    vmask = vmask.expand(-1, video.shape[1], -1, -1, -1).contiguous()
    if prev_video_mask is not None:
        vmask = _combine(prev_video_mask.to(vmask.device, torch.float32), vmask)
    return vmask, s
