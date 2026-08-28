import pytest
import torch

torchaudio = pytest.importorskip("torchaudio")

from mmx_nodes.audio_quality_gate import MiniMaxH3_AudioQualityGate
from mmx_utils.audio_quality_gate import evaluate_audio_quality, lowpass_waveform


def _audio(waveform: torch.Tensor, sample_rate: int = 32_000) -> dict:
    return {"waveform": waveform, "sample_rate": sample_rate}


def _noise_clip(frames: int = 32_000, seed: int = 0) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randn(1, 1, frames)


def test_identical_audio_passes():
    w = _noise_clip()
    result = evaluate_audio_quality(w, w.clone())
    assert result.passed is True
    assert result.mel_l1 < 1e-4
    assert result.centroid_drift_hz < 1.0


def test_lowpassed_audio_fails_centroid_and_bass():
    ref = _noise_clip(seed=1)
    muffled = lowpass_waveform(ref, 32_000, cutoff_hz=300.0).unsqueeze(0).unsqueeze(0)
    result = evaluate_audio_quality(
        ref,
        muffled,
        centroid_drift_hz_max=400.0,
        low_mel_ratio_min=0.7,
        mel_l1_max=0.15,
    )
    assert result.passed is False
    assert result.centroid_gen_hz < result.centroid_ref_hz
    assert result.low_mel_ratio < 0.7


def test_silent_generated_fails_rms():
    ref = _noise_clip(seed=2)
    silent = torch.zeros_like(ref)
    result = evaluate_audio_quality(ref, silent, rms_ratio_min=0.1)
    assert result.passed is False
    assert result.rms_ratio < 0.1


def test_node_execute_passes_on_match():
    w = _noise_clip(16_000)
    out = MiniMaxH3_AudioQualityGate.execute(
        _audio(w),
        _audio(w.clone()),
    )
    passed, report = out[0], out[1]
    assert passed is True
    assert "PASS" in report


def test_node_execute_fails_on_muffled():
    ref = _noise_clip(16_000, seed=3)
    muffled = lowpass_waveform(ref, 32_000, cutoff_hz=250.0).unsqueeze(0).unsqueeze(0)
    out = MiniMaxH3_AudioQualityGate.execute(
        _audio(ref),
        _audio(muffled),
        centroid_drift_hz_max=300.0,
        low_mel_ratio_min=0.75,
        mel_l1_max=0.12,
    )
    assert out[0] is False
    assert "FAILED" in out[1]
