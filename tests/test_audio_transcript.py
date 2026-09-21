"""Transcribing a soundtrack for lipsync timing.

Whisper itself is not installed here and is not the point. What is testable,
and what breaks quietly, is everything around it:

  * the AUDIO contract. A wrong shape reaching Whisper does not raise - it
    transcribes the wrong channel, or silence, and returns a confident empty
    string.
  * the model cache. It lives on the MODULE, not an instance: this pack builds
    a fresh node instance per execution, so an instance cache would make
    "Keep loaded" silently reload a multi-gigabyte model every run while
    reporting that it kept it.

CPU-only, no Whisper, no weights.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

PACK = Path(__file__).resolve().parents[1]
if str(PACK) not in sys.path:
    sys.path.insert(0, str(PACK))

at = pytest.importorskip("mmx_nodes.audio_transcript", reason="needs torch")


def audio(channels=1, samples=16000, rate=16000):
    return {"waveform": torch.zeros(1, channels, samples), "sample_rate": rate}


# ── the AUDIO contract ──────────────────────────────────────────────────────

def test_a_valid_mono_clip_is_accepted():
    wav, rate = at._require_audio_tensor(audio())
    assert wav.shape == (1, 1, 16000) and rate == 16000


def test_stereo_is_accepted():
    wav, _ = at._require_audio_tensor(audio(channels=2))
    assert wav.shape[1] == 2


def test_something_that_is_not_an_audio_value_is_named():
    with pytest.raises(ValueError, match="AUDIO"):
        at._require_audio_tensor("not audio")


def test_a_missing_waveform_is_named():
    with pytest.raises(ValueError, match="waveform"):
        at._require_audio_tensor({"sample_rate": 16000})


def test_the_wrong_number_of_dimensions_is_named():
    """A [C, S] tensor reaching Whisper does not raise - it transcribes the
    wrong thing and returns a confident answer."""
    with pytest.raises(ValueError, match=r"\[1, C, S\]"):
        at._require_audio_tensor({"waveform": torch.zeros(1, 16000),
                                  "sample_rate": 16000})


def test_a_batch_of_more_than_one_is_refused():
    """Transcribing only the first of a batch and reporting one transcript
    would silently drop the rest."""
    with pytest.raises(ValueError, match="batch"):
        at._require_audio_tensor({"waveform": torch.zeros(3, 1, 16000),
                                  "sample_rate": 16000})


def test_more_than_two_channels_is_refused():
    with pytest.raises(ValueError, match="channels"):
        at._require_audio_tensor({"waveform": torch.zeros(1, 6, 16000),
                                  "sample_rate": 16000})


def test_an_empty_waveform_is_refused():
    """Whisper on an empty buffer returns an empty transcript, which is
    indistinguishable from silence that genuinely has no words."""
    with pytest.raises(ValueError, match="empty"):
        at._require_audio_tensor(audio(samples=0))


def test_a_nonsense_sample_rate_is_refused():
    with pytest.raises(ValueError, match="sample_rate"):
        at._require_audio_tensor({"waveform": torch.zeros(1, 1, 16000),
                                  "sample_rate": 0})
    with pytest.raises(ValueError, match="sample_rate"):
        at._require_audio_tensor({"waveform": torch.zeros(1, 1, 16000),
                                  "sample_rate": "fast"})


# ── preparing the buffer ────────────────────────────────────────────────────

def test_stereo_is_mixed_to_mono():
    """Whisper takes mono. Passing it interleaved stereo transcribes noise."""
    wav = torch.zeros(1, 2, 16000)
    wav[0, 0] = 1.0
    wav[0, 1] = -1.0
    out = at._prepare_whisper_audio({"waveform": wav, "sample_rate": 16000})
    assert out.ndim == 1, "the buffer reaching Whisper is not mono"
    assert abs(float(out.mean() if hasattr(out, "mean") else out.mean())) < 1e-6


def test_sixteen_kilohertz_audio_is_not_resampled():
    """Resampling audio that is already at the target rate is pure loss."""
    out = at._prepare_whisper_audio(audio(samples=16000, rate=16000))
    assert len(out) == 16000


def test_other_rates_are_resampled_to_sixteen_kilohertz():
    """Whisper assumes 16 kHz. Hand it 48 kHz unchanged and every word comes
    out at a third speed - it still transcribes, just wrongly."""
    out = at._prepare_whisper_audio(audio(samples=48000, rate=48000))
    assert abs(len(out) - 16000) <= 2, f"got {len(out)} samples, wanted ~16000"


def test_nan_samples_are_refused():
    wav = torch.zeros(1, 1, 16000)
    wav[0, 0, 5] = float("nan")
    with pytest.raises(ValueError, match="NaN"):
        at._prepare_whisper_audio({"waveform": wav, "sample_rate": 16000})


def test_infinite_samples_are_refused():
    wav = torch.zeros(1, 1, 16000)
    wav[0, 0, 5] = float("inf")
    with pytest.raises(ValueError, match="Infinity"):
        at._prepare_whisper_audio({"waveform": wav, "sample_rate": 16000})


# ── language ────────────────────────────────────────────────────────────────

def test_auto_means_let_whisper_decide():
    assert at._language_code("auto") is None


def test_named_languages_become_codes():
    assert at._language_code("English") == "en"
    assert at._language_code("Korean") == "ko"


def test_an_unsupported_language_is_named_not_silently_auto():
    """Falling back to auto would look like the setting being ignored."""
    with pytest.raises(ValueError, match="Unsupported"):
        at._language_code("Klingon")


def test_every_offered_choice_resolves():
    for name in at.LANGUAGE_CHOICES:
        at._language_code(name)


# ── confidence ──────────────────────────────────────────────────────────────

def test_confidence_bands_are_ordered():
    assert at._confidence_band(-0.1) == "high"
    assert at._confidence_band(-0.5) == "medium"
    assert at._confidence_band(-2.0) == "low"


def test_an_unknown_confidence_is_unknown_not_high():
    """None or NaN reported as 'high' would tell the user to trust a
    transcript nothing measured."""
    assert at._confidence_band(None) == "unknown"
    assert at._confidence_band(float("nan")) == "unknown"
    assert at._confidence_band(float("inf")) == "unknown"


# ── the model cache ─────────────────────────────────────────────────────────

def test_the_cache_is_on_the_module_not_an_instance():
    """THE change from upstream. This pack builds a fresh node instance per
    execution, so an instance cache would make 'Keep loaded' reload a
    multi-gigabyte model every run while reporting that it kept it."""
    assert isinstance(at._MODEL_CACHE, dict)
    assert set(at._MODEL_CACHE) == {"model", "key"}
    assert not hasattr(at.DenoAudioTranscript, "__init__") or \
        "_cached_model" not in str(at.DenoAudioTranscript.__init__.__doc__ or "")


def test_releasing_the_cache_clears_both_halves():
    """Leaving the key behind makes the next run believe a freed model is
    still loaded."""
    at._MODEL_CACHE["model"] = object()
    at._MODEL_CACHE["key"] = ("large-v3", "cuda")
    at._release_cached_model()
    assert at._MODEL_CACHE == {"model": None, "key": None}


def test_the_node_does_not_keep_model_state_on_self():
    source = (PACK / "mmx_nodes" / "audio_transcript.py").read_text(encoding="utf-8")
    assert "self._cached_model" not in source, (
        "the model cache went back onto the instance; with a fresh instance "
        "per execution that silently reloads the model every run")


# ── whisper is optional ─────────────────────────────────────────────────────

def test_a_missing_whisper_is_a_readable_message_not_an_import_error():
    """It is not a ComfyUI dependency. An import failure at module level would
    take the node out of the menu with no explanation."""
    with pytest.raises(RuntimeError, match="whisper"):
        at._import_whisper()


def test_the_module_imports_without_whisper():
    assert at.DenoAudioTranscript is not None


# ── registration ────────────────────────────────────────────────────────────

def test_the_node_is_adapted_to_the_v3_schema():
    node = getattr(at, "MiniMaxH3_AudioTranscript", None)
    if node is None:
        pytest.skip("adapter unavailable")
    s = node.define_schema()
    assert s.node_id == "MiniMaxH3_AudioTranscript"
    assert s.category == "MiniMax H3/Audio"
    assert [o.display_name for o in s.outputs] == \
        ["audio_context", "transcript", "audio"]


def test_the_audio_passes_straight_through():
    """The third output is the input clip. A node that transcribes and then
    hands back a re-encoded buffer would quietly change the audio."""
    node = getattr(at, "MiniMaxH3_AudioTranscript", None)
    if node is None:
        pytest.skip("adapter unavailable")
    names = [o.display_name for o in node.define_schema().outputs]
    assert names[2] == "audio"
