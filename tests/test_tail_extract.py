"""Taking the tail of a clip to start the next one.

The two failures worth pinning both accumulate across a chain of clips, which
is what makes them expensive: by clip six the drift is obvious and the cause
is five clips back.

CPU-only, torch only, no VAE.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

PACK = Path(__file__).resolve().parents[1]
if str(PACK) not in sys.path:
    sys.path.insert(0, str(PACK))

from mmx_utils.tail_extract import (  # noqa: E402
    FPS,
    MIN_H3_FRAMES,
    TailExtractError,
    align_up,
    describe,
    normalise_audio,
    slice_audio_tail,
    tail_frame_count,
)


# ── the grid ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("n", [1, 5, 6, 22, 23, 39, 100, 124, 361])
def test_alignment_always_lands_on_the_grid(n):
    assert align_up(n) % 17 == 5


def test_alignment_rounds_up_because_this_is_a_request_not_a_measurement():
    """A tail shorter than asked for carries less motion, and 5 is the floor
    already - so rounding down is the wrong direction here, unlike a reference
    video whose length is a fact about the source."""
    for n in range(1, 200):
        assert align_up(n) >= max(MIN_H3_FRAMES, n)


def test_a_count_already_on_the_grid_does_not_move():
    for n in (5, 22, 39, 56):
        assert align_up(n) == n


# ── choosing the length ─────────────────────────────────────────────────────

def test_half_a_second_becomes_a_legal_frame_count():
    count = tail_frame_count(124, 0.5, align=True)
    assert count % 17 == 5
    assert count >= 12


def test_the_tail_can_never_be_longer_than_the_clip():
    assert tail_frame_count(30, 10.0, align=True) <= 30
    assert tail_frame_count(30, 10.0, align=False) <= 30


def test_alignment_that_does_not_fit_falls_back_to_the_whole_clip():
    """Rounding up can exceed the clip. Clamping is right; silently returning
    an off-grid slice as if it were aligned is not - the report says so."""
    count = tail_frame_count(10, 0.5, align=True)
    assert count == 10
    assert "NOT on the grid" in describe(10, count, 0.5, aligned=True)


def test_unaligned_mode_gives_exactly_what_was_asked_for():
    assert tail_frame_count(124, 1.0, align=False) == int(round(1.0 * FPS))


def test_an_empty_clip_is_named_rather_than_returning_nothing():
    with pytest.raises(TailExtractError, match="no frames"):
        tail_frame_count(0, 0.5, align=True)


# ── picture and sound staying together ──────────────────────────────────────

def test_the_audio_tail_matches_the_video_tail_in_duration():
    """THE drift. The tail is chosen in frames and the audio in samples;
    round them independently and every clip in the chain inherits the offset."""
    total, sr = 124, 32000
    count = tail_frame_count(total, 0.5, align=True)
    duration = count / FPS

    audio = {"waveform": torch.zeros(1, 2, total * sr // 24),
             "sample_rate": sr}
    tail = slice_audio_tail(audio, duration)

    audio_seconds = tail["waveform"].shape[-1] / sr
    video_seconds = count / FPS
    assert audio_seconds == pytest.approx(video_seconds, abs=1.0 / sr + 1e-6)


def test_the_audio_length_follows_the_ALIGNED_count_not_the_request():
    """0.5s is 12 frames; aligned it becomes 22. The audio must be 22 frames
    long too, or the reference video is 22 frames with 12 frames of sound."""
    count = tail_frame_count(124, 0.5, align=True)
    assert count != int(round(0.5 * FPS)), "the fixture no longer aligns"

    sr = 32000
    audio = {"waveform": torch.zeros(1, 2, 124 * sr // 24), "sample_rate": sr}
    tail = slice_audio_tail(audio, count / FPS)
    assert tail["waveform"].shape[-1] == pytest.approx(
        count / FPS * sr, abs=2)


def test_the_audio_is_taken_from_the_END():
    """The point is the join: the next clip starts where this one stopped.
    Slicing from the start hands it audio from half a clip ago."""
    sr = 1000
    waveform = torch.arange(sr * 4, dtype=torch.float32).view(1, 1, -1)
    tail = slice_audio_tail({"waveform": waveform, "sample_rate": sr}, 1.0)
    assert tail["waveform"].shape[-1] == sr
    assert float(tail["waveform"][0, 0, -1]) == float(waveform[0, 0, -1])


def test_asking_for_more_audio_than_exists_returns_all_of_it():
    sr = 1000
    waveform = torch.zeros(1, 2, sr // 2)
    tail = slice_audio_tail({"waveform": waveform, "sample_rate": sr}, 10.0)
    assert tail["waveform"].shape[-1] == sr // 2


def test_the_sample_rate_survives():
    tail = slice_audio_tail(
        {"waveform": torch.zeros(1, 2, 48000), "sample_rate": 48000}, 0.5)
    assert tail["sample_rate"] == 48000


# ── audio normalisation ─────────────────────────────────────────────────────

def test_a_loud_clip_is_scaled_down():
    loud = torch.randn(1, 2, 8000) * 10.0
    assert float(normalise_audio(loud).abs().max()) < float(loud.abs().max())


def test_near_silence_is_not_amplified_into_noise():
    """The std floor of 1.0 is what stops a quiet passage being multiplied up.
    Without it a silent join comes back as a hiss at full level."""
    quiet = torch.randn(1, 2, 8000) * 1e-4
    out = normalise_audio(quiet)
    assert torch.allclose(out, quiet, atol=1e-9), "quiet audio was amplified"


def test_true_silence_survives_as_silence():
    silence = torch.zeros(1, 2, 8000)
    assert float(normalise_audio(silence).abs().max()) == 0.0


# ── the report ──────────────────────────────────────────────────────────────

def test_the_report_states_that_the_audio_follows_the_frame_count():
    text = describe(124, 22, 0.5, aligned=True)
    assert "SAME" in text
    assert "drift" in text


def test_the_report_names_the_rounding():
    text = describe(124, 22, 0.5, aligned=True)
    assert "12 -> 22" in text
    assert "wrong speed" in text


def test_the_report_is_quiet_when_nothing_was_rounded():
    assert "Rounded" not in describe(124, 24, 1.0, aligned=True)


def test_the_report_says_when_the_tail_is_the_whole_clip():
    text = describe(22, 22, 5.0, aligned=True)
    assert "entire clip" in text
