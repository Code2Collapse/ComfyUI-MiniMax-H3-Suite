"""MiniMaxH3_SourceAVLatent - a source video + soundtrack as the masked joint AV latent H3 edits (A16 Part II).

Original implementation, GPL-3.0. The job NKD AV Latent does in the owner's masked-replacement workflow
(Nekodificador/ComfyUI-NKD-Basic-Tools nkd_av_latent.py, MIT - behaviour reference, no code taken), with the
geometry made explicit instead of implicit:

* SIZE. Core's H3 VAE centre-crops each side to a multiple of 16 before encoding (comfy/sd.py
  vae_encode_crop_pixels; crop_input stays True for H3). A frame that is not on that grid comes back smaller and
  shifted by up to 7 px, with nothing said. Here it is refused, or padded at the right / bottom with the margins
  recorded - never cropped.
* LENGTH. H3 encodes 17k+5 frames; any other count is silently shortened by the VAE. Refused with the nearest legal
  counts.
* MASK. A pixel mask handed to the sampler is resampled trilinearly in time (comfy/utils.py reshape_mask), which is
  not H3's (1,4,4,4,4) frame grouping. Here it is reduced with H3 Mask To Latent Space on the VAE's own grouping and
  emitted at the latent's exact shape, so the sampler passes it through unchanged and core pools it to tokens.
* SOUND. Audio length = core's round(frames / 24 * 40); keep / regenerate / follow the picture mask per 1/40 s tick.
* METADATA. The latent carries "mmx_geometry" (source size, padding, frames, audio ticks) so the composite and the
  geometry report know exactly what was encoded. Extra latent keys survive core's samplers (they copy the dict).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from comfy_api.latest import io

_PKG = Path(__file__).resolve().parents[1]
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from mmx_utils import h3_geometry as geo  # noqa: E402

AUDIO_MODES = ("keep", "regenerate", "follow mask")
SIZE_POLICIES = ("refuse", "pad right/bottom (recorded)")


def _nested(parts):
    import comfy.nested_tensor  # noqa: WPS433
    return comfy.nested_tensor.NestedTensor(tuple(parts))


def _encode_audio(audio_vae, audio):
    from comfy_extras.nodes_audio import VAEEncodeAudio  # noqa: WPS433 - core's own audio encode
    out = VAEEncodeAudio.execute(audio_vae, audio)
    return (out[0] if isinstance(out, (tuple, list)) else out.args[0])["samples"]


def fit_audio_ticks(samples: torch.Tensor, ticks: int) -> tuple[torch.Tensor, str]:
    """Trim or zero-pad an H3 audio latent [B, 32, 2, T] to exactly `ticks` on its last axis."""
    have = int(samples.shape[-1])
    if have == ticks:
        return samples, ""
    if have > ticks:
        return samples[..., :ticks].contiguous(), f"audio trimmed {have} -> {ticks} ticks"
    pad = samples.new_zeros(samples.shape[:-1] + (ticks - have,))
    return torch.cat([samples, pad], dim=-1), f"audio padded {have} -> {ticks} ticks with silence"


def audio_mask_from_frames(frame_mask_max: torch.Tensor, ticks: int, frames: int) -> torch.Tensor:
    """1 for every 1/40 s tick whose frames have any masked pixel, else 0. frame_mask_max [frames]."""
    out = torch.zeros(ticks)
    for t in range(ticks):
        a = int((t * geo.FPS) // geo.AUDIO_TICKS_PER_SECOND)
        b = int(-(-((t + 1) * geo.FPS) // geo.AUDIO_TICKS_PER_SECOND))
        a, b = min(a, frames - 1), min(max(b, a + 1), frames)
        out[t] = float(frame_mask_max[a:b].max() > 0)
    return out


def latent_video_mask(mask: torch.Tensor, video_vae) -> torch.Tensor:
    """Pixel mask [frames, H, W] -> [T, h, w] on H3's own grouping (H3 Mask To Latent Space, auto, max)."""
    from mmx_nodes.mask_to_latent import MiniMaxH3_MaskToLatentSpace  # noqa: WPS433
    out = MiniMaxH3_MaskToLatentSpace.execute(masks=mask, compression={"compression": "auto"},
                                              spatial_method="max", temporal_method="max",
                                              grow_spatial=0, grow_temporal=0, vae=video_vae)
    return (out.args[0] if hasattr(out, "args") else out[0]).float()


def build_source_av_latent(images, video_vae, audio_vae=None, audio=None, mask=None, audio_mode="keep",
                           size_policy=SIZE_POLICIES[0]) -> tuple[dict, str]:
    if images.ndim != 4:
        raise ValueError(f"H3 Source AV Latent: images must be a frame batch [frames, H, W, C], got {tuple(images.shape)}.")
    n, h, w = int(images.shape[0]), int(images.shape[1]), int(images.shape[2])
    if not geo.is_legal_frame_count(n):
        down, up = geo.nearest_legal_frame_counts(n)
        raise ValueError(
            f"H3 Source AV Latent: {n} frames is not a length H3 encodes (17k+5). The VAE would silently drop the "
            f"tail. Trim to {down} or pad to {up} frames first (AV Handles / H3 Frame Handles do this and trim back).")
    if mask is not None:
        if mask.ndim == 2:
            mask = mask[None].expand(n, -1, -1)
        if mask.shape[0] == 1 and n > 1:
            mask = mask.expand(n, -1, -1)
        if tuple(mask.shape) != (n, h, w):
            raise ValueError(f"H3 Source AV Latent: the mask is {tuple(mask.shape)} but the frames are {n}x{h}x{w}. "
                             "Give one mask per frame at the frame size (crop the mask with the frames).")
    pad_r, pad_b = (-w) % geo.LATENT_STRIDE, (-h) % geo.LATENT_STRIDE
    notes = []
    if pad_r or pad_b:
        if size_policy == SIZE_POLICIES[0]:
            m = geo.vae_crop_margins(w, h)
            raise ValueError(
                f"H3 Source AV Latent: {w}x{h} is not a multiple of 16. The H3 VAE would silently cut "
                f"{m['left']}+{m['right']} px from the sides and {m['top']}+{m['bottom']} px from top/bottom and the "
                f"result would sit offset from the plate. Resize or crop to a multiple of 32 (Subject Crop's "
                f"divisible_by), or set size_policy to 'pad right/bottom (recorded)'.")
        x = torch.nn.functional.pad(images[..., :3].movedim(-1, 1), (0, pad_r, 0, pad_b), mode="replicate")
        images = x.movedim(1, -1)
        if mask is not None:
            mask = torch.nn.functional.pad(mask, (0, pad_r, 0, pad_b), value=0.0)
        notes.append(f"padded {w}x{h} -> {w + pad_r}x{h + pad_b} at the right/bottom (pad pixels kept, never edited)")
    elif w % geo.TOKEN_PX or h % geo.TOKEN_PX:
        notes.append(f"{w}x{h} is on the 16 px VAE grid but not the 32 px token grid: core pads the last token "
                     "row/column internally (no shift)")

    video = video_vae.encode(images[..., :3])
    if video.ndim != 5 or video.shape[1] != 24:
        raise ValueError(f"H3 Source AV Latent: video_vae is not the MiniMax H3 video VAE (encoded {tuple(video.shape)}).")
    T, lh, lw = int(video.shape[2]), int(video.shape[3]), int(video.shape[4])
    ticks = geo.expected_audio_ticks(n)
    if audio is not None:
        if audio_vae is None:
            raise ValueError("H3 Source AV Latent: audio is connected but audio_vae is not.")
        a = _encode_audio(audio_vae, audio)
        a, note = fit_audio_ticks(a, ticks)
        if note:
            notes.append(note)
    else:
        a = torch.zeros((1, 32, 2, ticks), dtype=video.dtype, device=video.device)
        notes.append(f"no audio connected: {ticks} ticks of silent latent")

    if mask is not None:
        vm = latent_video_mask(mask.float(), video_vae)
        if tuple(vm.shape) != (T, lh, lw):
            raise ValueError(f"H3 Source AV Latent: the reduced mask is {tuple(vm.shape)} but the latent is "
                             f"{(T, lh, lw)} - the VAE's grouping and its encode disagree; report this.")
        video_mask = vm[None, None].to(video.device)
    else:
        video_mask = torch.ones((1, 1, T, lh, lw), device=video.device)
    if audio_mode == "keep":
        audio_mask = torch.zeros((1, 1, 2, ticks), device=a.device)
    elif audio_mode == "regenerate":
        audio_mask = torch.ones((1, 1, 2, ticks), device=a.device)
    else:
        if mask is None:
            raise ValueError("H3 Source AV Latent: audio_mode 'follow mask' needs a mask.")
        per_frame = mask.float().reshape(mask.shape[0], -1).amax(dim=1)[:n]
        audio_mask = audio_mask_from_frames(per_frame, ticks, n).reshape(1, 1, 1, ticks).expand(1, 1, 2, ticks)
        audio_mask = audio_mask.contiguous().to(a.device)

    geometry = {"source_px": [w, h], "encoded_px": [w + pad_r, h + pad_b], "pad_right": pad_r, "pad_bottom": pad_b,
                "frames": n, "latent": [T, lh, lw], "audio_ticks": ticks, "audio_mode": audio_mode,
                "mask": mask is not None}
    latent = {"samples": _nested((video, a)), "noise_mask": _nested((video_mask, audio_mask)),
              "mmx_geometry": geometry}
    cov = geo.mask_token_coverage(video_mask[0, 0].cpu()) if mask is not None else None
    lines = [f"{n} frames {w}x{h} -> latent {T}x{lh}x{lw} (+ {ticks} audio ticks)"]
    if cov:
        lines.append(f"mask on the token grid: {cov['regenerated_pct']}% regenerated, {cov['kept_pct']}% kept, "
                     f"{cov['partial']} partial tokens")
    lines.append(f"audio: {audio_mode}")
    lines += notes
    return latent, "\n".join(lines)


class MiniMaxH3_SourceAVLatent(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MiniMaxH3_SourceAVLatent",
            display_name="H3 Source AV Latent",
            category="MiniMax H3/Latent",
            description=(
                "Encode the source frames and their sound into the joint picture + sound latent H3 edits, with the "
                "edit mask on H3's own grid. Feed it to the sampler as latent_image. Unlike a plain VAE Encode it "
                "never crops the frame (the H3 VAE silently cuts off-grid edges, which moves the result against the "
                "plate), refuses a frame count H3 cannot encode, reduces the mask on H3's own frame grouping, and "
                "keeps or regenerates the sound per 1/40 s."),
            inputs=[
                io.Image.Input("images", tooltip="Source frames (17k+5 of them, e.g. 121 or 124), at the size the "
                                                 "model will generate - a multiple of 32 is best."),
                io.Vae.Input("video_vae", tooltip="The MiniMax H3 video VAE."),
                io.Vae.Input("audio_vae", optional=True, tooltip="The MiniMax H3 audio VAE (needed with audio)."),
                io.Audio.Input("audio", optional=True, tooltip="The source soundtrack. Without it, a silent track of "
                                                               "the right length is used."),
                io.Mask.Input("mask", optional=True, tooltip="What to regenerate, one per frame at the frame size "
                                                             "(1 = regenerate, 0 = keep). Without it everything "
                                                             "is regenerated."),
                io.Combo.Input("audio_mode", options=list(AUDIO_MODES), default="keep",
                               tooltip="keep: the soundtrack survives untouched. regenerate: all of it is new. "
                                       "follow mask: only the 1/40 s ticks whose frames are masked are new."),
                io.Combo.Input("size_policy", options=list(SIZE_POLICIES), default=SIZE_POLICIES[0],
                               tooltip="A frame that is not a multiple of 16: refuse (say how to fix it), or pad "
                                       "the right/bottom edge (recorded in the latent; the composite removes it)."),
            ],
            outputs=[
                io.Latent.Output(display_name="latent", tooltip="Joint AV latent with its noise masks."),
                io.String.Output(display_name="report", tooltip="What was encoded and what the mask keeps."),
            ],
        )

    @classmethod
    def execute(cls, images, video_vae, audio_vae=None, audio=None, mask=None, audio_mode="keep",
                size_policy=SIZE_POLICIES[0]) -> io.NodeOutput:
        latent, report = build_source_av_latent(images, video_vae, audio_vae, audio, mask, audio_mode, size_policy)
        return io.NodeOutput(latent, report, ui={"text": [report]})


__all__ = ["MiniMaxH3_SourceAVLatent", "build_source_av_latent", "fit_audio_ticks", "audio_mask_from_frames"]
