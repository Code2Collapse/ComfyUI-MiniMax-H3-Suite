"""MiniMax H3 native exact-audio lock and reference-audio prep nodes."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
import torch
from comfy_api.latest import io

_PKG = Path(__file__).resolve().parents[1]
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from mmx_utils.native_audio import (  # noqa: E402
    assert_h3_av_latent,
    build_audio_lock_report,
    build_video_only_denoise_masks,
    describe_region_mask,
    existing_video_mask,
    fit_encoded_audio_to_target,
    fit_waveform_to_vae_rate,
    pack_locked_av_latent,
    patch_model_audio_lock,
    prepare_mono_batch_waveform,
    prepare_reference_audio,
    pixel_frame_count_from_audio_latent,
    trim_audio_start,
)

CATEGORY = "MiniMax H3/Audio"


def _nested_factory():
    import comfy.nested_tensor

    return comfy.nested_tensor.NestedTensor


class MiniMaxH3_NativeAudioLock(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MiniMaxH3_NativeAudioLock",
            display_name="H3 Native Exact-Audio Lock",
            category=CATEGORY,
            description=(
                "Encodes exact user audio into H3's target AV latent, fixes the audio stream at "
                "its clean timestep, and masks sampling so H3 denoises only video while jointly "
                "attending to the song. Audio shorter than the clip leaves the tail unlocked and "
                "the mouth drifts; a sample-rate mismatch that is silently resampled changes pitch."
            ),
            inputs=[
                io.Model.Input("model"),
                io.Latent.Input(
                    "av_latent",
                    tooltip="Joint MiniMax H3 AV latent (NestedTensor video+audio).",
                ),
                io.Vae.Input(
                    "audio_vae",
                    tooltip="The H3 audio VAE (not the video VAE).",
                ),
                io.Audio.Input(
                    "audio",
                    tooltip="Exact soundtrack to lock. Shorter than the clip: tail stays unlocked. "
                    "Wrong sample rate: resampled to the VAE rate (pitch may shift).",
                ),
            ],
            outputs=[
                io.Model.Output("model"),
                io.Latent.Output("av_latent"),
                io.Audio.Output("exact_audio"),
                io.String.Output("report"),
            ],
        )

    @classmethod
    def fingerprint_inputs(cls, model, av_latent, audio_vae, audio) -> str:
        h = hashlib.md5(audio["waveform"].cpu().numpy().tobytes()).hexdigest()
        samples = av_latent.get("samples")
        if samples is not None and hasattr(samples, "unbind"):
            for t in samples.unbind():
                h += ":" + hashlib.md5(t.cpu().numpy().tobytes()).hexdigest()
        return h

    @classmethod
    def execute(cls, model, av_latent, audio_vae, audio) -> io.NodeOutput:
        video_latent, target_audio_template = assert_h3_av_latent(av_latent)
        waveform, input_sr = prepare_mono_batch_waveform(audio)
        vae_rate = int(getattr(audio_vae, "audio_sample_rate", 32000))
        waveform, vae_rate, resampled = fit_waveform_to_vae_rate(waveform, input_sr, vae_rate)

        exact_audio_latent = audio_vae.encode(waveform.movedim(1, -1))
        target_t = int(target_audio_template.shape[-1])
        exact_audio_latent, fit_note = fit_encoded_audio_to_target(exact_audio_latent, target_t)

        kept = existing_video_mask(av_latent, video_latent)
        video_mask, audio_mask = build_video_only_denoise_masks(
            video_latent, exact_audio_latent, kept)
        locked = pack_locked_av_latent(
            av_latent,
            video_latent,
            exact_audio_latent,
            video_mask,
            audio_mask,
            _nested_factory(),
        )
        patched_model = patch_model_audio_lock(model)

        clip_frames = pixel_frame_count_from_audio_latent(target_audio_template)
        report = build_audio_lock_report(
            input_sample_rate=input_sr,
            vae_sample_rate=vae_rate,
            resampled=resampled,
            audio_duration_s=float(waveform.shape[-1]) / float(vae_rate),
            clip_duration_s=clip_frames / 24.0,
            fit_note=fit_note,
        )
        exact_out = {
            "waveform": audio["waveform"].clone(),
            "sample_rate": int(audio["sample_rate"]),
        }
        report = report + "\n" + describe_region_mask(kept)
        return io.NodeOutput(patched_model, locked, exact_out, report)


class MiniMaxH3_ReferenceAudio(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MiniMaxH3_ReferenceAudio",
            display_name="H3 Reference Audio (stereo guard)",
            category=CATEGORY,
            description=(
                "Force a reference audio clip to stereo 32 kHz before Ref2VA. A MONO reference "
                "crashes the sampler with an unhelpful shape mismatch, because the layout reserves "
                "two channels."
            ),
            inputs=[
                io.Audio.Input("audio"),
                io.Float.Input(
                    "max_seconds",
                    default=10.0,
                    min=0.5,
                    max=60.0,
                    step=0.5,
                    tooltip="Trim the reference to at most this long. Long references cost speed "
                    "on every sampling step.",
                ),
            ],
            outputs=[
                io.Audio.Output("audio"),
                io.String.Output("report"),
            ],
        )

    @classmethod
    def fingerprint_inputs(cls, audio, max_seconds) -> str:
        return hashlib.md5(audio["waveform"].cpu().numpy().tobytes()).hexdigest() + f":{max_seconds}"

    @classmethod
    def execute(cls, audio, max_seconds) -> io.NodeOutput:
        out, report = prepare_reference_audio(audio, max_seconds)
        return io.NodeOutput(out, report)


class MiniMaxH3_AudioTrimStart(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MiniMaxH3_AudioTrimStart",
            display_name="H3 Audio Trim Start",
            category=CATEGORY,
            description=(
                "Trim N seconds off the FRONT of an audio clip. Used by multishot chaining to drop "
                "each shot's duplicated first frame (1/24s) from video AND audio together."
            ),
            inputs=[
                io.Audio.Input("audio"),
                io.Float.Input(
                    "seconds",
                    default=0.04167,
                    min=0.0,
                    max=10.0,
                    step=0.00001,
                    tooltip="Seconds to remove from the start (0.04167 ≈ one 24fps frame).",
                ),
            ],
            outputs=[
                io.Audio.Output("audio"),
                io.String.Output("report"),
            ],
        )

    @classmethod
    def fingerprint_inputs(cls, audio, seconds) -> str:
        return hashlib.md5(audio["waveform"].cpu().numpy().tobytes()).hexdigest() + f":{seconds}"

    @classmethod
    def execute(cls, audio, seconds) -> io.NodeOutput:
        out, report = trim_audio_start(audio, seconds)
        return io.NodeOutput(out, report)
