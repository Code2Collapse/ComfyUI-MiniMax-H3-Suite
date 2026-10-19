"""Where MiniMax H3 puts every conditioning stream in space and time (owner directive A16, Part II).

Original implementation, GPL-3.0. The rules are core's, read from source (ComfyUI 0.39, e8665ca8):

* A 2x2 patch of the video latent (16 px VAE stride -> 32 px) is one token. Its spatial RoPE coordinate is
  area-normalised and centred, and marks the patch's LEFT/TOP EDGE:
  ``linspace((1-r)/2, (1+r)/2, n, endpoint=False) * 32`` (comfy/ldm/minimax/model.py ``_axis_from_sqrt_area``,
  ``_frame_grid``).
* Keyframes (fl2va / guides) are packed on the TARGET's frame grid; reference blocks (``minimax_refs``) on their own
  grid ``_frame_grid(latent_h, latent_w)`` (PackedLayout).
* So a stream that shows the same scene as the target (the source video, its pose / depth, a guide) coincides with
  the target exactly when it has the target's latent size. With bigger patches (lower resolution) its content sits
  half a coarse patch right/down of the coordinate the model is given: offset = (P_target - P_stream) / 2 target px.
* The H3 video VAE centre-crops each spatial side to a multiple of 16 before encoding (comfy/sd.py
  ``vae_encode_crop_pixels``, ``crop_input`` left True for H3).
* Audio runs at 40 latent ticks per second against 24 fps video; core sizes it round(frames / 24 * 40).

Core's own functions are imported, never re-implemented, so a core change is picked up (or fails loudly) rather than
drifting. Nothing here touches weights.
"""

from __future__ import annotations

from typing import Any

LATENT_STRIDE = 16           # H3 video VAE: 16 px per latent cell
PATCH = 2                    # DiT patch on the latent: 2 x 2 cells -> 32 px token
TOKEN_PX = LATENT_STRIDE * PATCH
FPS = 24
AUDIO_TICKS_PER_SECOND = 40


class H3GeometryError(ValueError):
    """A geometry that H3 would misread, said in plain English."""


def _core_model():
    try:
        from comfy.ldm.minimax import model as m  # noqa: WPS433 - core is optional at import time
    except Exception as exc:  # noqa: BLE001 - AttributeError from comfy_kitchen skew is real (see memory notes)
        raise H3GeometryError(
            "This needs ComfyUI's MiniMax H3 model code (comfy/ldm/minimax/model.py), which could not be imported "
            f"({type(exc).__name__}: {exc}). Update ComfyUI to a build with MiniMax H3.") from exc
    return m


def frame_count_for_latent_t(latent_t: int) -> int:
    """Pixel frames covered by ``latent_t`` video latent rows (core FRAME_PER_TOKEN, row 0 = 1 frame)."""
    m = _core_model()
    return int(sum(m.FRAME_PER_TOKEN[k % len(m.FRAME_PER_TOKEN)] for k in range(int(latent_t))))


def expected_audio_ticks(frames: int) -> int:
    """Core's audio length for a clip of ``frames`` pixel frames: round(frames / 24 * 40)."""
    return int(round(int(frames) / FPS * AUDIO_TICKS_PER_SECOND))


def is_legal_frame_count(n: int) -> bool:
    return int(n) >= 5 and int(n) % 17 == 5


def nearest_legal_frame_counts(n: int) -> tuple[int, int]:
    """(largest legal count <= n, smallest legal count >= n) - legal = 17k + 5, at least 5."""
    n = int(n)
    down = n
    while down >= 5 and down % 17 != 5:
        down -= 1
    up = max(5, n)
    while up % 17 != 5:
        up += 1
    return (down if down >= 5 else 5), up


def vae_crop_margins(width: int, height: int, stride: int = LATENT_STRIDE) -> dict[str, int]:
    """What core's VAE encode silently removes from a frame of this size (centre crop to a multiple of stride)."""
    w, h, s = int(width), int(height), int(stride)
    left = (w % s) // 2
    top = (h % s) // 2
    return {"left": left, "right": (w % s) - left, "top": top, "bottom": (h % s) - top,
            "kept_width": w // s * s, "kept_height": h // s * s,
            "cropped": bool(w % s or h % s)}


def rope_grid(latent_h: int, latent_w: int):
    """Core's spatial RoPE coordinates for one latent frame: tensor [rows, 2] (h, w) per 2x2 patch."""
    grid, _ = _core_model()._frame_grid(int(latent_h), int(latent_w))
    return grid


def stream_offset_px(target_px: tuple[int, int], stream_px: tuple[int, int]) -> dict[str, Any]:
    """Offset (target pixels) at which a stream showing the SAME scene as the target is fed to the model.

    ``target_px`` / ``stream_px`` are (width, height) in pixels as encoded (multiples of 16). For every patch of the
    stream: its content is centred at the patch centre, in relative frame coordinates; the model is told the patch's
    RoPE coordinate; that coordinate is mapped back to target pixels through the target's own grid. The difference is
    what the model is fed. 0 everywhere when the stream has the target's latent size.
    """
    tw, th = int(target_px[0]), int(target_px[1])
    sw, sh = int(stream_px[0]), int(stream_px[1])
    t_lh, t_lw = th // LATENT_STRIDE, tw // LATENT_STRIDE
    s_lh, s_lw = sh // LATENT_STRIDE, sw // LATENT_STRIDE
    if min(t_lh, t_lw, s_lh, s_lw) < PATCH:
        raise H3GeometryError(f"A frame of {sw}x{sh} or {tw}x{th} is smaller than one H3 token ({TOKEN_PX} px).")
    tg = rope_grid(t_lh, t_lw).reshape(t_lh // PATCH, t_lw // PATCH, 2)
    sg = rope_grid(s_lh, s_lw).reshape(s_lh // PATCH, s_lw // PATCH, 2)
    t_rows, t_cols = tg[:, 0, 0], tg[0, :, 1]
    s_rows, s_cols = sg[:, 0, 0], sg[0, :, 1]
    nth, ntw = t_rows.shape[0], t_cols.shape[0]
    nsh, nsw = s_rows.shape[0], s_cols.shape[0]
    dh = float(t_rows[1] - t_rows[0]) if nth > 1 else 1.0
    dw = float(t_cols[1] - t_cols[0]) if ntw > 1 else 1.0

    def to_target_x(w_pos):
        return ((float(w_pos) - float(t_cols[0])) / dw + 0.5) * (tw / ntw)

    def to_target_y(h_pos):
        return ((float(h_pos) - float(t_rows[0])) / dh + 0.5) * (th / nth)

    xs = [to_target_x(s_cols[j]) - (j + 0.5) / nsw * tw for j in range(nsw)]
    ys = [to_target_y(s_rows[i]) - (i + 0.5) / nsh * th for i in range(nsh)]
    return {
        "target_px": [tw, th], "stream_px": [sw, sh],
        "target_token_px": [tw / ntw, th / nth], "stream_token_px_in_target": [tw / nsw, th / nsh],
        "same_grid": (t_lh, t_lw) == (s_lh, s_lw),
        "dx_px": [round(min(xs), 2), round(max(xs), 2)], "dy_px": [round(min(ys), 2), round(max(ys), 2)],
        "dx_mean_px": round(sum(xs) / len(xs), 2), "dy_mean_px": round(sum(ys) / len(ys), 2),
        "max_abs_px": round(max(max(abs(v) for v in xs), max(abs(v) for v in ys)), 2),
        "aspect_target": round(tw / th, 5), "aspect_stream": round(sw / sh, 5),
    }


def rope_extent(latent_h: int, latent_w: int) -> dict[str, list[float]]:
    g = rope_grid(latent_h, latent_w)
    return {"h": [round(float(g[:, 0].min()), 3), round(float(g[:, 0].max()), 3)],
            "w": [round(float(g[:, 1].min()), 3), round(float(g[:, 1].max()), 3)]}


def pool_video_mask_like_core(mask):
    """Core's own video noise-mask pooling (model_base.MiniMaxH3._pool_masks_to_token_grid, video branch):
    replicate-pad to the 2x2 patch, amax per patch, broadcast back, then quantise ceil(m*256)/256.
    ``mask`` [..., h, w] at latent resolution."""
    import torch
    h, w = mask.shape[-2:]
    lead = mask.shape[:-2]
    m = torch.nn.functional.pad(mask.reshape((-1,) + mask.shape[-3:]) if mask.ndim >= 3 else mask[None],
                                (0, -w % PATCH, 0, -h % PATCH), mode="replicate")
    m = m.reshape(lead + m.shape[-2:])
    m = m.reshape(m.shape[:-2] + (m.shape[-2] // PATCH, PATCH, m.shape[-1] // PATCH, PATCH)).amax(dim=(-3, -1))
    m = m.repeat_interleave(PATCH, dim=-2).repeat_interleave(PATCH, dim=-1)[..., :h, :w]
    return torch.ceil(m * 256.0) / 256.0


def mask_token_coverage(video_mask) -> dict[str, Any]:
    """What a video noise mask means once core has pooled it: share of tokens regenerated / kept / partial."""
    pooled = pool_video_mask_like_core(video_mask.float())
    tok = pooled[..., ::PATCH, ::PATCH]
    total = tok.numel()
    regen = int((tok >= 1.0 - 1e-3).sum())
    keep = int((tok <= 1e-6).sum())
    return {"tokens": int(total), "regenerated": regen, "kept": keep, "partial": int(total - regen - keep),
            "regenerated_pct": round(100.0 * regen / max(1, total), 2),
            "kept_pct": round(100.0 * keep / max(1, total), 2)}


def describe_offset(o: dict[str, Any]) -> str:
    if o["same_grid"]:
        return "same grid as the target - offset 0"
    return (f"fed {o['dx_mean_px']:+.1f} px horizontally, {o['dy_mean_px']:+.1f} px vertically on average "
            f"(range x {o['dx_px'][0]:+.1f}..{o['dx_px'][1]:+.1f}, y {o['dy_px'][0]:+.1f}..{o['dy_px'][1]:+.1f}) "
            f"from where its content is, because its tokens are "
            f"{o['stream_token_px_in_target'][0]:.0f} px wide in target pixels against the target's "
            f"{o['target_token_px'][0]:.0f} px")


def plate_scale(bbox: dict[str, Any] | None, target_px: tuple[int, int]) -> float | None:
    """Target-pixel -> plate-pixel factor for a crop box (Subject Crop bboxes); None without a box."""
    if not bbox:
        return None
    try:
        return float(bbox["width"]) / float(target_px[0])
    except (KeyError, TypeError, ZeroDivisionError):
        return None


__all__ = [
    "AUDIO_TICKS_PER_SECOND", "FPS", "H3GeometryError", "LATENT_STRIDE", "PATCH", "TOKEN_PX",
    "describe_offset", "expected_audio_ticks", "frame_count_for_latent_t", "is_legal_frame_count",
    "mask_token_coverage", "nearest_legal_frame_counts", "plate_scale", "pool_video_mask_like_core",
    "rope_extent", "rope_grid", "stream_offset_px", "vae_crop_margins",
]

