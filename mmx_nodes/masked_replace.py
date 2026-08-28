"""N4 — H3 Masked Replace (inject video, then noise-mask)."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import torch

from comfy_api.latest import io

_PKG = Path(__file__).resolve().parents[1]
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from mmx_utils.av_latent import (
    assert_nested_samples,
    build_av_noise_mask,
    inject_audio_stream,
    inject_video_stream,
    normalize_encoded_video,
    pack_latent_dict,
    validate_canvas_multiple,
    validate_crop_frame_count,
    validate_spatial_latent,
)

try:
    import comfy.model_management as mm
    import comfy.nested_tensor as nested_tensor
except Exception:  # not just ImportError: the comfy_kitchen skew raises AttributeError
    mm = None
    nested_tensor = None


class MiniMaxH3_MaskedReplace(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MiniMaxH3_MaskedReplace",
            display_name="H3 Masked Replace",
            category="MiniMax H3/Spine",
            description=(
                "Two sequential steps on an H3 AV latent:\n"
                "1) Inject encoded plate crops into the VIDEO stream (audio intact).\n"
                "2) Set noise_mask = NestedTensor((video_mask, audio_mask)).\n\n"
                "video_mask = N2 soft latent mask (1=regenerate, 0=preserve).\n"
                "audio_mask = zeros when audio is LOCKED (lipsync), ones when generative.\n\n"
                "Supply audio + audio_vae to lock the audio stream (vocal stem for lipsync).\n"
                "No mel spectrogram — H3 audio is a VAE over raw waveform.\n\n"
                "Strength is NOT a widget here — control denoise with BasicScheduler. "
                "SplitSigmas is illegal on H3; at shift 12 even the last split point is "
                "~sigma 0.80. guidance_scale is distilled (always 1.0).\n\n"
                "P5 regional denoise: video_mask from N2 is written into noise_mask; "
                "the model soft-scales denoise per token (see comfy/ldm/minimax/model.py). "
                "Chain P10 → P11 guards before SigmaShiftLocked → LegalScheduler → sampler."
            ),
            inputs=[
                io.Latent.Input("av_latent"),
                io.Image.Input("crops", tooltip="Canvas-space crops from H3 Track + Crop."),
                io.Vae.Input("vae"),
                io.Mask.Input("video_latent_mask", tooltip="Video-side mask from H3 Mask Prep (N2)."),
                io.Audio.Input(
                    "audio",
                    optional=True,
                    tooltip="Vocal stem for lipsync — locks audio when connected.",
                ),
                io.Vae.Input("audio_vae", optional=True, tooltip="Required when audio is connected."),
            ],
            outputs=[
                io.Latent.Output("av_latent"),
                io.String.Output("report"),
            ],
        )

    @classmethod
    def fingerprint_inputs(
        cls,
        av_latent,
        crops,
        vae,
        video_latent_mask,
        audio=None,
        audio_vae=None,
    ):
        parts = [
            hashlib.md5(crops.cpu().numpy().tobytes()).hexdigest(),
            hashlib.md5(video_latent_mask.cpu().numpy().tobytes()).hexdigest(),
            str(crops.shape),
        ]
        if audio is not None:
            parts.append(hashlib.md5(audio["waveform"].cpu().numpy().tobytes()).hexdigest())
        return ":".join(parts)

    @classmethod
    def _encode_video(cls, vae, crops: torch.Tensor, dev: torch.device) -> torch.Tensor:
        imgs = crops[..., :3].to(dev).float()
        try:
            encoded = vae.encode(imgs)
        except Exception as exc:
            try:
                if mm is not None:
                    mm.soft_empty_cache()
                encoded = vae.encode(imgs.cpu())
            except Exception:
                raise RuntimeError(f"H3 Masked Replace: VAE encode failed: {exc}") from exc
        return normalize_encoded_video(encoded)

    @classmethod
    def _encode_audio(cls, audio_vae, audio: dict, dev: torch.device) -> torch.Tensor:
        waveform = audio["waveform"]
        sr = int(audio["sample_rate"])
        vae_sr = int(getattr(audio_vae, "audio_sample_rate", 32000))
        wf = waveform[:1]
        if wf.ndim == 3:
            wf = wf.to(dev)
        else:
            wf = wf.unsqueeze(0).to(dev)
        if sr != vae_sr:
            try:
                import torchaudio

                wf = torchaudio.functional.resample(wf, sr, vae_sr)
            except Exception as exc:
                raise RuntimeError(
                    f"H3 Masked Replace: audio resample {sr}→{vae_sr} failed: {exc}"
                ) from exc
        try:
            z = audio_vae.encode(wf.movedim(1, -1))
        except Exception as exc:
            try:
                z = audio_vae.encode(wf.cpu().movedim(1, -1))
            except Exception:
                raise RuntimeError(f"H3 Masked Replace: audio VAE encode failed: {exc}") from exc
        return z

    @classmethod
    def execute(
        cls,
        av_latent,
        crops,
        vae,
        video_latent_mask,
        audio=None,
        audio_vae=None,
    ):
        if nested_tensor is None:
            raise RuntimeError("comfy.nested_tensor is unavailable — run inside ComfyUI.")

        if crops.ndim == 3:
            crops = crops.unsqueeze(0)

        cw, ch = int(crops.shape[2]), int(crops.shape[1])
        validate_canvas_multiple(cw, ch)

        samples = av_latent.get("samples")
        members = assert_nested_samples(samples)
        video_tmpl = members[0]
        validate_crop_frame_count(int(crops.shape[0]), video_tmpl)

        dev = crops.device
        if mm is not None:
            try:
                dev = mm.get_torch_device()
            except Exception:
                dev = crops.device

        encoded = cls._encode_video(vae, crops, dev)
        validate_spatial_latent(encoded, video_tmpl)
        members, temporal_note = inject_video_stream(members, encoded, video_tmpl)

        audio_locked = audio is not None
        if audio_locked:
            if audio_vae is None:
                raise ValueError(
                    "audio is connected but audio_vae is missing. Wire the H3 audio VAE for lipsync lock."
                )
            enc_a = cls._encode_audio(audio_vae, audio, dev)
            members = inject_audio_stream(members, enc_a)

        vmask, amask = build_av_noise_mask(
            video_latent_mask,
            members[0],
            members[1],
            audio_locked=audio_locked,
        )

        out = pack_latent_dict(
            av_latent,
            members,
            (vmask, amask),
            nested_tensor.NestedTensor,
        )

        lock_msg = "audio LOCKED (noise_mask audio=0)" if audio_locked else "audio generative (noise_mask audio=1)"
        report = (
            f"injected video latent {tuple(encoded.shape)} then set noise_mask\n"
            f"{lock_msg}\n"
            f"video_mask shape={tuple(vmask.shape)} audio_mask shape={tuple(amask.shape)}\n"
            f"{temporal_note}\n"
            f"Denoise strength: BasicScheduler only (no SplitSigmas on H3)."
        )
        return io.NodeOutput(out, report)
