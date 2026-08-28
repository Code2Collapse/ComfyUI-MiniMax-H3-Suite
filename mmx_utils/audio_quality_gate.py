# Post-sampling audio QC — mel L1, spectral centroid drift, RMS.
# Mel spectrograms here are MEASUREMENT ONLY (EasyCache muffled-bass detection).
# Invariant 8 ("no mel-spectrogram conditioning") applies to model inputs, not QC.

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

_DEFAULT_SR = 32_000
_N_FFT = 1024
_HOP = 256
_N_MELS = 80
_FMIN = 20.0
_FMAX = 16_000.0
_LOW_MEL_BINS = 16


@dataclass(frozen=True)
class AudioQCResult:
    passed: bool
    mel_l1: float
    centroid_ref_hz: float
    centroid_gen_hz: float
    centroid_drift_hz: float
    rms_ref: float
    rms_gen: float
    rms_ratio: float
    low_mel_ratio: float
    report: str


def _waveform_mono(waveform: torch.Tensor) -> torch.Tensor:
    """Return [T] float32 mono."""
    w = waveform.float()
    if w.ndim == 3:
        w = w[0]
    if w.ndim == 2:
        w = w.mean(dim=0)
    return w.contiguous()


def _rms(w: torch.Tensor) -> float:
    return float(w.pow(2).mean().sqrt().item())


def _mel_transform(sample_rate: int, device: torch.device):
    import torchaudio

    return torchaudio.transforms.MelSpectrogram(
        sample_rate=int(sample_rate),
        n_fft=_N_FFT,
        hop_length=_HOP,
        n_mels=_N_MELS,
        f_min=_FMIN,
        f_max=min(float(_FMAX), sample_rate / 2.0 - 1.0),
        power=2.0,
    ).to(device)


def _mel_centroid_hz(mel: torch.Tensor, sample_rate: int) -> float:
    """Spectral centroid (Hz) from a [n_mels, frames] power mel."""
    fmax = min(float(_FMAX), sample_rate / 2.0 - 1.0)
    mel_freqs = torch.linspace(_FMIN, fmax, mel.shape[0], device=mel.device, dtype=torch.float32)
    power = mel.clamp(min=1e-10)
    # Both sums run over the MEL-BIN axis (dim=0) so each frame gets its own
    # energy-weighted mean frequency. Summing the denominator over dim=1 instead
    # gives per-bin totals of shape [n_mels, 1], which then broadcasts the
    # [frames] numerator up to [n_mels, frames] — the result is not a frequency
    # at all, and reported ~2.7e8 Hz for 32 kHz audio whose Nyquist is 16 kHz.
    numer = (power * mel_freqs.unsqueeze(1)).sum(dim=0)          # [frames]
    denom = power.sum(dim=0).clamp(min=1e-10)                    # [frames]
    centroid_per_frame = numer / denom
    return float(centroid_per_frame.mean().item())


def _compute_mel(waveform: torch.Tensor, sample_rate: int, device: torch.device) -> torch.Tensor:
    mono = _waveform_mono(waveform).to(device)
    mel_fn = _mel_transform(sample_rate, device)
    mel = mel_fn(mono.unsqueeze(0)).squeeze(0)  # [n_mels, frames]
    return mel.clamp(min=1e-10)


def lowpass_waveform(
    waveform: torch.Tensor,
    sample_rate: int,
    cutoff_hz: float = 400.0,
) -> torch.Tensor:
    """Low-pass for synthetic failure fixtures (tests only)."""
    import torchaudio

    w = _waveform_mono(waveform)
    return torchaudio.functional.lowpass_biquad(w, int(sample_rate), float(cutoff_hz))


def evaluate_audio_quality(
    reference: torch.Tensor,
    generated: torch.Tensor,
    *,
    sample_rate: int = _DEFAULT_SR,
    device: torch.device | None = None,
    mel_l1_max: float = 0.35,
    centroid_drift_hz_max: float = 800.0,
    rms_ratio_min: float = 0.55,
    low_mel_ratio_min: float = 0.45,
) -> AudioQCResult:
    """
    Compare generated audio against a reference stem.

    Fails when:
    - mean mel L1 exceeds ``mel_l1_max`` (timbre drift),
    - spectral centroid drifts beyond ``centroid_drift_hz_max`` (muffled / dull),
    - RMS ratio generated/reference falls below ``rms_ratio_min`` (level collapse),
    - low-mel energy ratio falls below ``low_mel_ratio_min`` (bass collapse).
    """
    dev = device or reference.device
    mel_ref = _compute_mel(reference, sample_rate, dev)
    mel_gen = _compute_mel(generated, sample_rate, dev)
    frames = min(mel_ref.shape[1], mel_gen.shape[1])
    mel_ref = mel_ref[:, :frames]
    mel_gen = mel_gen[:, :frames]

    mel_l1 = float((mel_ref - mel_gen).abs().mean().item())
    c_ref = _mel_centroid_hz(mel_ref, sample_rate)
    c_gen = _mel_centroid_hz(mel_gen, sample_rate)
    centroid_drift = abs(c_gen - c_ref)

    w_ref = _waveform_mono(reference)
    w_gen = _waveform_mono(generated)
    rms_ref = _rms(w_ref)
    rms_gen = _rms(w_gen)
    rms_ratio = rms_gen / max(rms_ref, 1e-8)

    low_ref = float(mel_ref[:_LOW_MEL_BINS].sum().item())
    low_gen = float(mel_gen[:_LOW_MEL_BINS].sum().item())
    low_mel_ratio = low_gen / max(low_ref, 1e-8)

    failures: list[str] = []
    if mel_l1 > float(mel_l1_max):
        failures.append(f"mel L1 {mel_l1:.4f} > {mel_l1_max:g}")
    if centroid_drift > float(centroid_drift_hz_max):
        failures.append(
            f"centroid drift {centroid_drift:.1f} Hz > {centroid_drift_hz_max:g} Hz "
            f"(ref={c_ref:.1f} gen={c_gen:.1f})"
        )
    if rms_ratio < float(rms_ratio_min):
        failures.append(f"RMS ratio {rms_ratio:.4f} < {rms_ratio_min:g}")
    if low_mel_ratio < float(low_mel_ratio_min):
        failures.append(f"low-mel energy ratio {low_mel_ratio:.4f} < {low_mel_ratio_min:g}")

    passed = not failures
    status = "PASS" if passed else "FAIL"
    report = (
        f"audio QC {status}: mel_l1={mel_l1:.4f} centroid_drift={centroid_drift:.1f}Hz "
        f"rms_ratio={rms_ratio:.4f} low_mel_ratio={low_mel_ratio:.4f}\n"
        f"centroid ref={c_ref:.1f}Hz gen={c_gen:.1f}Hz | rms ref={rms_ref:.4f} gen={rms_gen:.4f}"
    )
    if failures:
        report += "\n" + "; ".join(failures)
    return AudioQCResult(
        passed=passed,
        mel_l1=mel_l1,
        centroid_ref_hz=c_ref,
        centroid_gen_hz=c_gen,
        centroid_drift_hz=centroid_drift,
        rms_ref=rms_ref,
        rms_gen=rms_gen,
        rms_ratio=rms_ratio,
        low_mel_ratio=low_mel_ratio,
        report=report,
    )
