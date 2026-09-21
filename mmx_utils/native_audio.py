"""Pure native-audio lock logic — testable without ComfyUI or VAE weights.

PORTED FROM: third_party/MiniMax-H3-NativeAudio-MusicVideo-Workflow ::
  custom_nodes/ComfyUI-H3-NativeAudioLock/__init__.py
  custom_nodes/ComfyUI-H3-Multishot/h3_advanced.py (reference-audio prep)
"""

from __future__ import annotations

from typing import Any, Callable, Mapping, Protocol, Sequence

import torch
import torch.nn.functional as F

from .h3_constants import AUDIO_LATENT_FPS, FPS


class NestedLike(Protocol):
    def unbind(self) -> Sequence[torch.Tensor]: ...


def assert_h3_av_latent(av_latent: Mapping[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    """Require a joint MiniMax H3 AV latent; name what was received on failure."""
    samples = av_latent.get("samples")
    if samples is None:
        raise ValueError(
            'LATENT dict has no "samples" key — expected a MiniMax H3 joint AV latent '
            "(NestedTensor video+audio pair)."
        )
    if not getattr(samples, "is_nested", False):
        kind = type(samples).__name__
        shape = tuple(samples.shape) if hasattr(samples, "shape") else "?"
        raise ValueError(
            f"Expected a MiniMax H3 joint AV latent (NestedTensor video+audio pair); "
            f"received samples as {kind} with shape {shape}."
        )
    members = list(samples.unbind())
    if len(members) < 2:
        raise ValueError(
            f"AV latent NestedTensor has {len(members)} stream(s); expected video+audio."
        )
    return members[0], members[1]


def resample_waveform(
    waveform: torch.Tensor,
    src_rate: int,
    dst_rate: int,
) -> tuple[torch.Tensor, bool]:
    """Resample [B,C,L] waveform; returns (tensor, did_resample)."""
    if int(src_rate) == int(dst_rate):
        return waveform, False
    try:
        import torchaudio

        out = torchaudio.functional.resample(waveform, int(src_rate), int(dst_rate))
        return out, True
    except Exception:
        # Linear fallback so CPU tests never need torchaudio.
        ratio = float(dst_rate) / float(src_rate)
        new_len = max(1, int(round(waveform.shape[-1] * ratio)))
        flat = waveform.reshape(-1, waveform.shape[-1]).float()
        idx = torch.linspace(0, flat.shape[-1] - 1, new_len, device=flat.device)
        idx_lo = idx.floor().long().clamp(max=flat.shape[-1] - 1)
        idx_hi = idx.ceil().long().clamp(max=flat.shape[-1] - 1)
        w = (idx - idx_lo.float()).unsqueeze(0)
        out = flat[:, idx_lo] * (1.0 - w) + flat[:, idx_hi] * w
        return out.reshape(*waveform.shape[:-1], new_len).to(waveform.dtype), True


def prepare_mono_batch_waveform(audio: Mapping[str, Any]) -> tuple[torch.Tensor, int]:
    """Take batch item 0 from an AUDIO dict without mutating the input."""
    wav = audio["waveform"]
    if wav.ndim == 2:
        wav = wav.unsqueeze(0)
    return wav[:1].clone(), int(audio["sample_rate"])


def fit_waveform_to_vae_rate(
    waveform: torch.Tensor,
    sample_rate: int,
    vae_rate: int,
) -> tuple[torch.Tensor, int, bool]:
    out, did = resample_waveform(waveform, sample_rate, vae_rate)
    return out, int(vae_rate), did


def fit_encoded_audio_to_target(
    encoded: torch.Tensor,
    target_t: int,
) -> tuple[torch.Tensor, str]:
    """Trim or zero-pad encoded audio latent along time; never mutate *encoded*."""
    z = encoded.clone()
    got = int(z.shape[-1])
    tgt = int(target_t)
    notes: list[str] = []
    if got > tgt:
        z = z[..., :tgt]
        notes.append(f"audio latent trimmed {got} -> {tgt} frames")
    elif got < tgt:
        z = F.pad(z, (0, tgt - got))
        notes.append(
            f"audio shorter than clip: latent padded {got} -> {tgt} "
            f"(tail unlocked — mouth may drift past your audio)"
        )
    return z, ("; ".join(notes) if notes else "")


def existing_video_mask(av_latent, video_latent: torch.Tensor):
    """The video half of whatever noise mask is already on this latent.

    A face or character swap sets a REGION mask before the audio is locked.
    If the lock ignores it the whole frame regenerates and the swap quietly
    stops being a swap, so the existing mask has to be found and kept.
    """
    mask = av_latent.get("noise_mask") if hasattr(av_latent, "get") else None
    if mask is None:
        return None
    if getattr(mask, "is_nested", False):
        parts = mask.unbind()
        mask = parts[0] if parts else None
    if mask is None:
        return None
    m = torch.as_tensor(mask)
    if m.shape == video_latent.shape:
        return m
    # A mask at latent resolution but without the channel axis is the usual
    # shape from Set Latent Noise Mask; broadcast it rather than discard it.
    try:
        return m.expand_as(video_latent).contiguous()
    except RuntimeError:
        return None


def build_video_only_denoise_masks(
    video_latent: torch.Tensor,
    audio_latent: torch.Tensor,
    existing: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Audio locked clean (mask=0); video denoises where it is allowed to.

    `existing` is a region mask already on the latent - a face or head mask
    from a swap. Without it the video mask is all ones, which is right for a
    music video and WRONG for a swap: it regenerates the whole frame.
    """
    video_mask = (torch.ones_like(video_latent) if existing is None
                  else existing.to(video_latent.dtype).clone())
    audio_mask = torch.zeros_like(audio_latent)
    return video_mask, audio_mask


def audio_duration_seconds(waveform: torch.Tensor, sample_rate: int) -> float:
    return float(waveform.shape[-1]) / float(sample_rate)


def clip_duration_seconds(frame_count: int, fps: float = FPS) -> float:
    return float(frame_count) / float(fps)


def pixel_frame_count_from_audio_latent(audio_latent: torch.Tensor, fps: float = FPS) -> int:
    audio_t = int(audio_latent.shape[-1])
    return max(1, int(round(audio_t / AUDIO_LATENT_FPS * fps)))


def describe_region_mask(existing: torch.Tensor | None) -> str:
    """Say whether a region mask survived the lock, because silence here looks
    identical to a swap that regenerated the whole frame."""
    if existing is None:
        return ("No region mask on the latent: the whole frame will be "
                "regenerated. That is right for a music video and wrong for a "
                "swap - set the region mask BEFORE this node if you wanted "
                "only the face to change.")
    frac = float((existing > 0.5).float().mean()) * 100.0
    return (f"Region mask kept: {frac:.1f}% of the latent is free to change, "
            "the rest is held. Audio is locked either way.")


def build_audio_lock_report(
    *,
    input_sample_rate: int,
    vae_sample_rate: int,
    resampled: bool,
    audio_duration_s: float,
    clip_duration_s: float,
    fit_note: str,
) -> str:
    lines = [
        f"input sample rate: {input_sample_rate} Hz",
        f"VAE sample rate: {vae_sample_rate} Hz",
        f"resample: {'yes' if resampled else 'no'}",
        f"audio duration: {audio_duration_s:.3f}s",
        f"clip duration: {clip_duration_s:.3f}s",
        "video-only denoising: in effect (audio noise mask all-zero)",
    ]
    if resampled and input_sample_rate != vae_sample_rate:
        lines.append(
            "warning: sample-rate mismatch was resampled — pitch/timbre may shift "
            "relative to the source file"
        )
    if audio_duration_s + 1e-6 < clip_duration_s:
        lines.append(
            "warning: audio shorter than the clip — tail rows stay unlocked and "
            "lip sync may drift after your audio ends"
        )
    if fit_note:
        lines.append(fit_note)
    return "\n".join(lines)


def pack_locked_av_latent(
    av_latent: Mapping[str, Any],
    video_latent: torch.Tensor,
    audio_latent: torch.Tensor,
    video_mask: torch.Tensor,
    audio_mask: torch.Tensor,
    nested_ctor: Callable[[tuple[torch.Tensor, ...]], Any],
) -> dict:
    """Shallow-copy the latent dict and attach cloned streams + nested mask."""
    locked = dict(av_latent)
    locked["samples"] = nested_ctor((video_latent.clone(), audio_latent.clone()))
    locked["noise_mask"] = nested_ctor((video_mask.clone(), audio_mask.clone()))
    return locked


def lock_audio_chunk_into_latent(
    av_latent: Mapping[str, Any],
    audio_chunk: Mapping[str, Any],
    audio_vae: Any,
    nested_ctor: Callable[[tuple[torch.Tensor, ...]], Any],
) -> dict:
    """Encode one AUDIO chunk into an AV latent with video-only denoise mask."""
    video_latent, target_audio = assert_h3_av_latent(av_latent)
    waveform, input_sr = prepare_mono_batch_waveform(audio_chunk)
    vae_rate = int(getattr(audio_vae, "audio_sample_rate", 32000))
    waveform, vae_rate, _ = fit_waveform_to_vae_rate(waveform, input_sr, vae_rate)
    encoded = audio_vae.encode(waveform.movedim(1, -1))
    encoded, _ = fit_encoded_audio_to_target(encoded, int(target_audio.shape[-1]))
    video_mask, audio_mask = build_video_only_denoise_masks(
        video_latent, encoded, existing_video_mask(av_latent, video_latent))
    return pack_locked_av_latent(
        av_latent, video_latent, encoded, video_mask, audio_mask, nested_ctor
    )


def patch_model_audio_lock(model: Any) -> Any:
    """Clone MODEL and set upstream's minimax_h3_lock_audio_clean flag."""
    patched = model.clone()
    transformer_options = patched.model_options.get("transformer_options", {}).copy()
    transformer_options["minimax_h3_lock_audio_clean"] = True
    patched.model_options["transformer_options"] = transformer_options
    return patched


# ── reference audio prep (h3_advanced.H3ReferenceAudio) ─────────────────────

VAE_SR = 32000


def prepare_reference_audio(
    audio: Mapping[str, Any],
    max_seconds: float,
    vae_sr: int = VAE_SR,
) -> tuple[dict[str, Any], str]:
    """Force stereo @ vae_sr and trim; returns new AUDIO dict + report line."""
    wav, sr = prepare_mono_batch_waveform(audio)
    notes: list[str] = []

    ch = int(wav.shape[1])
    if ch == 1:
        wav = wav.repeat(1, 2, 1)
        notes.append("mono -> stereo (duplicated)")
    elif ch > 2:
        wav = wav[:, :2].clone()
        notes.append(f"{ch}ch -> first 2")

    if sr != vae_sr:
        wav, did = resample_waveform(wav, sr, vae_sr)
        if did:
            notes.append(f"{sr} -> {vae_sr} Hz")
            sr = vae_sr

    limit = int(float(max_seconds) * sr)
    if wav.shape[-1] > limit:
        notes.append(f"trimmed {wav.shape[-1] / sr:.1f}s -> {max_seconds:.1f}s")
        wav = wav[..., :limit].clone()

    report = (
        f"{wav.shape[1]}ch @{sr}Hz, {wav.shape[-1] / sr:.2f}s"
        + (" | " + "; ".join(notes) if notes else " | already valid")
    )
    return {"waveform": wav.contiguous(), "sample_rate": sr}, report


def trim_audio_start(audio: Mapping[str, Any], seconds: float) -> tuple[dict[str, Any], str]:
    """Trim *seconds* off the front; does not mutate the input AUDIO dict."""
    sr = int(audio["sample_rate"])
    wav = audio["waveform"].clone()
    n = int(round(float(seconds) * sr))
    trimmed = wav[..., n:].clone()
    report = f"trimmed {seconds:.5f}s ({n} samples @ {sr} Hz); new length {trimmed.shape[-1] / sr:.3f}s"
    return {"sample_rate": sr, "waveform": trimmed}, report
