# PORTED FROM: https://github.com/seitanism/ComfyUI-H3-Motion-Context-MultiRef@361624f existing_video_extension.py
# (MiniMaxH3SetAVNoiseMask, MiniMaxH3ClearAVNoiseMask) and nodes.py (MiniMaxH3CustomKeyframesMasked, contributed by
# Reithan), GPL-3.0. Changed here: the frame mask is reduced on H3's own grid (1,4,4,4,4 frames, 32 px tokens) with the
# suite's Mask To Latent Space reducer instead of trilinear resampling (which smears the mask across the 1/4-frame
# latent steps - the temporal misregistration found in A16); keyframes come as one IMAGE batch plus a positions list
# (no custom front end); the pre-#15375 compatibility shim is dropped (native nested AV masks required).
"""H3 Set / Clear AV Noise Mask and H3 Hard Keyframes.

An H3 AV latent carries one denoise mask per stream (video [1,1,T,h,w], audio [1,1,2,ticks]); 0 = keep, 1 = generate,
fractions = partial denoise. Core's Set Latent Noise Mask writes ONE plain tensor, which H3 reads as the video mask
only: the audio mask is lost and protected audio is regenerated. These nodes keep both streams.
"""

from __future__ import annotations

import logging
import re
import sys
from pathlib import Path

import torch
from comfy_api.latest import io

_PKG = Path(__file__).resolve().parents[1]
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from mmx_utils import h3_geometry as geo  # noqa: E402
from mmx_utils.h3_constants import FRAME_PER_TOKEN  # noqa: E402

_LOG = logging.getLogger("C2C.mmx.av_noise_mask")
CATEGORY = "MiniMax H3/Masking"
H3_SPATIAL, H3_TOKEN = 16, 2


def _nested(parts):
    import comfy.nested_tensor  # noqa: WPS433
    return comfy.nested_tensor.NestedTensor(tuple(parts))


def _streams(latent, what):
    s = latent.get("samples") if isinstance(latent, dict) else None
    if s is None:
        raise ValueError(f"{what}: expected a LATENT")
    if isinstance(s, torch.Tensor):          # a plain tensor has unbind() too: it would split the batch axis
        raise ValueError(f"{what}: this is a single-stream latent, not an H3 video + audio latent.")
    if hasattr(s, "unbind"):
        parts = list(s.unbind())
    elif isinstance(s, (list, tuple)):
        parts = list(s)
    else:
        raise ValueError(f"{what}: this is a single-stream latent, not an H3 video + audio latent.")
    if len(parts) < 2:
        raise ValueError(f"{what}: the latent has no audio stream (an H3 AV latent has video + audio).")
    v, a = parts[0], parts[1]
    if v.ndim == 4:
        v = v.unsqueeze(0)
    if a.ndim == 3:
        a = a.unsqueeze(0)
    if v.ndim != 5 or a.ndim != 4:
        raise ValueError(f"{what}: expected video [B,C,T,h,w] and audio [B,C,2,ticks], got {tuple(v.shape)} / "
                         f"{tuple(a.shape)}.")
    return v, a


def existing_masks(latent):
    m = latent.get("noise_mask")
    if m is None or isinstance(m, torch.Tensor):
        return None, None            # none, or a plain single-tensor mask (core Set Latent Noise Mask): not an AV mask
    parts = list(m.unbind()) if hasattr(m, "unbind") else list(m) if isinstance(m, (list, tuple)) else None
    if parts is None:
        return None, None
    return (parts[0] if parts else None), (parts[1] if len(parts) > 1 else None)


def pixel_frames(latent_t: int) -> int:
    return sum(FRAME_PER_TOKEN[k % 5] for k in range(int(latent_t)))


def step_offsets(latent_t: int) -> list[int]:
    out, acc = [], 0
    for k in range(int(latent_t)):
        out.append(acc)
        acc += FRAME_PER_TOKEN[k % 5]
    return out


def _norm_mask(mask) -> torch.Tensor:
    m = mask if torch.is_tensor(mask) else torch.as_tensor(mask)
    m = m.float()
    if m.ndim == 2:
        m = m.unsqueeze(0)
    elif m.ndim == 4:
        m = m.mean(dim=1)
    if m.ndim != 3:
        raise ValueError(f"a MASK must be [H,W] or [frames,H,W], got {tuple(m.shape)}")
    return m


def video_mask_for(mask, t: int, h: int, w: int) -> tuple[torch.Tensor, str]:
    """Frame-space mask -> [1,1,T,h,w] on H3's grid (max over each latent step's frames and each 32 px token);
    a mask already at latent size passes through."""
    from mmx_nodes.mask_to_latent import MiniMaxH3_MaskToLatentSpace, _pattern_geometry  # noqa: WPS433

    m = _norm_mask(mask)
    if tuple(m.shape) == (t, h, w):
        return m.clamp(0, 1)[None, None], "video mask already at latent size"
    frames = pixel_frames(t)
    if m.shape[0] == 1:
        m = m.expand(frames, -1, -1)
    if m.shape[0] != frames:
        raise ValueError(f"the video mask has {m.shape[0]} frames; this latent is {frames} frames ({t} latent steps).")
    if (m.shape[1], m.shape[2]) != (h * H3_SPATIAL, w * H3_SPATIAL):
        raise ValueError(f"the video mask is {m.shape[2]}x{m.shape[1]} px; this latent is {w * H3_SPATIAL}x"
                         f"{h * H3_SPATIAL} px. Resize the mask to the canvas first.")
    spatial, groups, _ = _pattern_geometry(H3_SPATIAL, list(FRAME_PER_TOKEN))
    red = MiniMaxH3_MaskToLatentSpace._reduce(m, spatial, groups, H3_TOKEN, "max", "max", 0, 0)
    return red.clamp(0, 1)[None, None], f"video mask reduced on H3's grid ({frames} frames -> {t} steps, 32 px tokens)"


def audio_mask_for(mask, ticks: int) -> tuple[torch.Tensor, str]:
    """Frame-space mask -> [1,1,2,ticks]: each 1/40 s tick takes the largest value among the frames it covers (a
    fraction stays a fraction), so a regenerated stretch of picture never leaves its sound preserved."""
    m = _norm_mask(mask)
    per_frame = m.flatten(1).max(dim=1).values
    if per_frame.numel() == ticks:
        return per_frame.clamp(0, 1).reshape(1, 1, 1, ticks).expand(1, 1, 2, ticks).contiguous(), \
            "audio mask already at tick resolution"
    frames = per_frame.numel()
    out = torch.zeros(ticks)
    for k in range(ticks):
        a = int((k * geo.FPS) // geo.AUDIO_TICKS_PER_SECOND)
        b = int(-(-((k + 1) * geo.FPS) // geo.AUDIO_TICKS_PER_SECOND))
        a, b = min(a, frames - 1), min(max(b, a + 1), frames)
        out[k] = per_frame[a:b].max()
    return out.clamp(0, 1).reshape(1, 1, 1, ticks).expand(1, 1, 2, ticks).contiguous(), \
        f"audio mask from {frames} frames -> {ticks} ticks (max per tick)"


def set_av_noise_mask(latent, video_mask=None, audio_mask=None, video_lock=False, audio_lock=False) -> tuple[dict, str]:
    what = "H3 Set AV Noise Mask"
    if video_mask is None and audio_mask is None and not video_lock and not audio_lock:
        raise ValueError(f"{what}: connect video_mask and/or audio_mask, or lock a stream. To remove the mask use H3 "
                         "Clear AV Noise Mask; to fully generate one stream, give that stream an all-white mask.")
    v, a = _streams(latent, what)
    ex_v, ex_a = existing_masks(latent)
    notes = []
    if video_lock:
        out_v = torch.zeros((1, 1) + tuple(v.shape[2:]))
        notes.append("video locked: every frame kept (only the sound is generated)" if not audio_lock
                     else "video locked")
    elif video_mask is not None:
        out_v, note = video_mask_for(video_mask, int(v.shape[2]), int(v.shape[3]), int(v.shape[4]))
        notes.append(note)
    elif ex_v is not None:
        out_v = ex_v
        notes.append("video mask kept from the input latent")
    else:
        out_v = torch.ones((1, 1) + tuple(v.shape[2:]))
        notes.append("video: all generate (no mask given)")
    if audio_lock:
        out_a = torch.zeros((1, 1, 2, int(a.shape[-1])))
        notes.append("audio locked: the sound is kept")
    elif audio_mask is not None:
        out_a, note = audio_mask_for(audio_mask, int(a.shape[-1]))
        notes.append(note)
    elif ex_a is not None:
        out_a = ex_a
        notes.append("audio mask kept from the input latent")
    else:
        out_a = torch.ones((1, 1, 2, int(a.shape[-1])))
        notes.append("audio: all generate (no mask given)")
    out = dict(latent)
    out["noise_mask"] = _nested((out_v.to(v.device, torch.float32), out_a.to(a.device, torch.float32)))
    kept_v = float((out_v < 1).float().mean()) * 100
    kept_a = float((out_a < 1).float().mean()) * 100
    notes.append(f"protected: {kept_v:.1f}% of video cells, {kept_a:.1f}% of audio ticks")
    return out, "\n".join(notes)


def _resize(image, width, height, crop):
    import comfy.utils  # noqa: WPS433
    return comfy.utils.common_upscale(image[..., :3].movedim(-1, 1), width, height, "lanczos", crop).movedim(1, -1)


def parse_positions(text: str) -> list[int]:
    vals = [int(x) for x in re.split(r"[,\s;]+", str(text or "").strip()) if x]
    if not vals:
        raise ValueError("no keyframe positions given (e.g. 1, 18, 35)")
    return vals


def hard_keyframes(latent, vae, images, positions, indexing="1-based", crop="disabled") -> tuple[dict, str]:
    what = "H3 Hard Keyframes"
    v, a = _streams(latent, what)
    t, h, w = int(v.shape[2]), int(v.shape[3]), int(v.shape[4])
    frames = pixel_frames(t)
    pos = parse_positions(positions)
    if images.ndim != 4 or int(images.shape[0]) != len(pos):
        raise ValueError(f"{what}: {len(pos)} positions but {int(images.shape[0]) if images.ndim == 4 else 0} images. "
                         "Give one image per position (Batch Images joins stills).")
    offsets = step_offsets(t)
    anchors, notes = [], []
    for i, p in enumerate(pos):
        px = p - 1 if indexing == "1-based" else p
        if not 0 <= px < frames:
            lo, hi = (1, frames) if indexing == "1-based" else (0, frames - 1)
            raise ValueError(f"{what}: keyframe {i + 1} position {p} is outside {lo}..{hi} ({frames} frames).")
        k = max(j for j, off in enumerate(offsets) if off <= px)
        span = FRAME_PER_TOKEN[k % 5]
        if span > 1:
            near = (px // 17) * 17
            show = lambda f: f + 1 if indexing == "1-based" else f  # noqa: E731
            notes.append(f"keyframe {i + 1} at {p}: inside a {span}-frame latent step (frames {show(offsets[k])}-"
                         f"{show(offsets[k] + span - 1)}), so the whole step is protected and may hold; single-frame "
                         f"positions are {show(near)}, {show(near + 17)}")
        anchors.append((k, i, p))
    steps = sorted(x[0] for x in anchors)
    if len(set(steps)) != len(steps):
        raise ValueError(f"{what}: two keyframes fall in the same latent step; move one of them.")
    ex_v, ex_a = existing_masks(latent)
    if latent.get("noise_mask") is not None and ex_v is None:
        raise ValueError(f"{what}: the latent has a single-tensor noise mask (core Set Latent Noise Mask). Rebuild it "
                         "with H3 Set AV Noise Mask, or clear it, before adding keyframes.")
    vmask = ex_v.float().clone() if ex_v is not None else torch.ones((1, 1, t, h, w))
    amask = ex_a.float().clone() if ex_a is not None else torch.ones((1, 1, 2, int(a.shape[-1])))
    if tuple(vmask.shape) != (1, 1, t, h, w):
        raise ValueError(f"{what}: the incoming video mask {tuple(vmask.shape)} does not match the latent "
                         f"{(1, 1, t, h, w)}.")
    for k, i, p in anchors:
        if bool((vmask[:, :, k] < 1.0 - 1e-6).any()):
            raise ValueError(f"{what}: keyframe {i + 1} (position {p}) falls in latent step {k}, which the incoming "
                             "mask already protects. Move it, or change that mask first.")
    out_v = v.clone()
    for k, i, p in anchors:
        z = vae.encode(_resize(images[i:i + 1], w * H3_SPATIAL, h * H3_SPATIAL, crop))
        if getattr(z, "ndim", 0) != 5 or int(z.shape[2]) != 1:
            raise ValueError(f"{what}: keyframe {i + 1} encoded to {tuple(getattr(z, 'shape', ()))}; expected one H3 "
                             "still latent [B,C,1,h,w]. Connect the H3 video VAE.")
        out_v[:, :, k:k + 1] = z[:1].to(out_v.device, out_v.dtype)
        vmask[:, :, k] = 0.0
    out = dict(latent)
    out["samples"] = _nested((out_v, a.clone()))
    out["noise_mask"] = _nested((vmask.to(v.device), amask.to(a.device)))
    head = f"{len(anchors)} keyframe(s) written at latent steps {sorted(steps)} of {t} ({frames} frames, {w * 16}x{h * 16})"
    return out, "\n".join([head, "audio mask unchanged"] + notes)


class MiniMaxH3_SetAVNoiseMask(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MiniMaxH3_SetAVNoiseMask",
            display_name="H3 Set AV Noise Mask",
            category=CATEGORY,
            description=("Set the video and/or audio denoise masks of an H3 AV latent (white = generate, black = keep, "
                         "grey = partial). Use it instead of core's Set Latent Noise Mask, whose single mask H3 reads "
                         "as video only - protected audio is then regenerated. The video mask is snapped to H3's own "
                         "grid (latent steps of 1/4 frames, 32 px tokens). A stream you do not connect keeps the "
                         "latent's existing mask, or generates."),
            inputs=[
                io.Latent.Input("latent", tooltip="An H3 AV latent (video + audio)."),
                io.Mask.Input("video_mask", optional=True,
                              tooltip="One mask per frame at the canvas size, or a latent-size mask (H3 Mask To "
                                      "Latent Space)."),
                io.Mask.Input("audio_mask", optional=True,
                              tooltip="One mask per frame; each 1/40 s audio tick takes the largest value of its "
                                      "frames. The picture mask works here too (sound regenerates with the picture)."),
                io.Boolean.Input("video_lock", default=False, optional=True,
                                 tooltip="Keep every video frame (mask 0), e.g. to re-sample only the sound."),
                io.Boolean.Input("audio_lock", default=False, optional=True,
                                 tooltip="Keep the whole sound (mask 0), e.g. to regenerate only the picture."),
            ],
            outputs=[io.Latent.Output(display_name="latent"), io.String.Output(display_name="report")],
        )

    @classmethod
    def execute(cls, latent, video_mask=None, audio_mask=None, video_lock=False, audio_lock=False) -> io.NodeOutput:
        out, report = set_av_noise_mask(latent, video_mask, audio_mask, video_lock, audio_lock)
        return io.NodeOutput(out, report, ui={"text": [report]})


class MiniMaxH3_ClearAVNoiseMask(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MiniMaxH3_ClearAVNoiseMask",
            display_name="H3 Clear AV Noise Mask",
            category=CATEGORY,
            description="Remove the whole denoise mask (both streams) from an H3 AV latent; the samples are unchanged.",
            inputs=[io.Latent.Input("latent")],
            outputs=[io.Latent.Output(display_name="latent")],
        )

    @classmethod
    def execute(cls, latent) -> io.NodeOutput:
        out = dict(latent)
        out.pop("noise_mask", None)
        return io.NodeOutput(out)


class MiniMaxH3_HardKeyframes(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MiniMaxH3_HardKeyframes",
            display_name="H3 Hard Keyframes (protected stills)",
            category=CATEGORY,
            description=("Write still images into the target latent at chosen frames and protect those latent steps "
                         "from denoising, for frames that must appear as given (core's Add Guide only suggests them). "
                         "Single-frame positions are 1, 18, 35, 52 ... (1-based); anywhere else protects the whole "
                         "4-frame step it falls in, which can hold the picture for those frames. The decoded pixels can "
                         "still differ slightly: the VAE decodes with its neighbours. Existing masks are kept; audio "
                         "is not touched. Feed the output to the sampler's latent input."),
            inputs=[
                io.Latent.Input("latent", tooltip="The target H3 AV latent (sets the size and frame count)."),
                io.Vae.Input("vae", tooltip="The H3 video VAE (encodes each still)."),
                io.Image.Input("images", tooltip="One still per position, as a batch (Batch Images)."),
                io.String.Input("positions", default="1, 18, 35",
                                tooltip="Frame of each still, in the batch's order, e.g. 1, 18, 35."),
                io.Combo.Input("indexing", options=["1-based", "0-based"], default="1-based"),
                io.Combo.Input("crop", options=["disabled", "center"], default="disabled",
                               tooltip="disabled stretches each still to the canvas; center crops it."),
            ],
            outputs=[io.Latent.Output(display_name="latent"), io.String.Output(display_name="report")],
        )

    @classmethod
    def execute(cls, latent, vae, images, positions, indexing="1-based", crop="disabled") -> io.NodeOutput:
        out, report = hard_keyframes(latent, vae, images, positions, indexing, crop)
        return io.NodeOutput(out, report, ui={"text": [report]})


__all__ = ["MiniMaxH3_SetAVNoiseMask", "MiniMaxH3_ClearAVNoiseMask", "MiniMaxH3_HardKeyframes", "set_av_noise_mask",
           "hard_keyframes", "video_mask_for", "audio_mask_for"]
