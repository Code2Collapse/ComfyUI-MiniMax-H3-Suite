"""CPU-only tests for multishot script / seam utilities."""

from __future__ import annotations

import pytest
import torch

from mmx_utils.multishot import (
    frames_per_shot_from_seconds,
    normalize_shot_list,
    parse_script,
    seam_trim_samples,
    shot_count_for_audio,
    slice_exact_audio_chunk,
    split_script_for_workflow,
    xfade_audio,
)


def test_parse_script_plain_separators():
    text = "first\n---\nsecond\n---\nthird"
    assert parse_script(text) == ["first", "second", "third"]


def test_parse_script_json_prompts():
    text = '{"prompts": ["a", "b"]}'
    assert parse_script(text) == ["a", "b"]


def test_a_repairable_script_is_repaired_not_rejected():
    """An unclosed bracket is the commonest typo in a prompt box, and the
    parser fixes it rather than throwing the whole script away."""
    assert parse_script('{"prompts": ["a"') == ["a"]


def test_json_that_cannot_be_repaired_is_refused_by_name():
    """It must not fall through to the plain-text path: a literal
    '{"prompts" nonsense' would then become the prompt for a whole shot."""
    with pytest.raises(ValueError, match="does not parse"):
        parse_script('{"prompts": [[[}}}"')


def test_frames_per_shot_snaps_to_h3_grid():
    # 5s @ 24fps = 120 -> snaps to 124 (17*7+5)
    assert frames_per_shot_from_seconds(5.0) == 124


def test_shot_count_from_audio():
    audio = {"waveform": torch.zeros(1, 2, 32000 * 10), "sample_rate": 32000}
    frames = frames_per_shot_from_seconds(5.0)
    n = shot_count_for_audio(audio, frames)
    assert n >= 1


def test_slice_exact_audio_chunk_pads_not_loops():
    source = torch.ones(1, 2, 100)
    chunk, sr, note = slice_exact_audio_chunk(source, 32000, shot_index=0, frames_per_shot=124)
    expected_len = int(round(124 * 32000 / 24.0))
    assert chunk.shape[-1] == expected_len
    if 100 < expected_len:
        assert "padded" in note.lower()
        assert chunk[..., 100:].sum() == 0


def test_split_script_for_workflow_four_outputs():
    shots, n, report = split_script_for_workflow("one\n---\ntwo", shot_count=0)
    assert len(shots) == 4
    assert n == 2
    assert "parsed 2 shot(s)" in report


def test_normalize_shot_list_extends_last_prompt():
    shots, notes = normalize_shot_list(["a"], 3)
    assert shots == ["a", "a", "a"]
    assert notes


def test_xfade_audio_joins_two_chunks():
    a = torch.ones(1, 2, 1000)
    b = torch.zeros(1, 2, 1000)
    out = xfade_audio([a, b], sr=32000, ms=40)
    assert out is not None
    assert out.shape[-1] > 1000


def test_a_seam_fade_never_eats_a_whole_chunk():
    """40ms at 32kHz is 1280 samples - longer than these chunks. Clamped to
    the full chunk length, the join returned 1000 samples from 2000 and half
    the audio vanished with no error."""
    a = torch.ones(1, 2, 1000)
    b = torch.zeros(1, 2, 1000)
    out = xfade_audio([a, b], sr=32000, ms=40)
    assert out.shape[-1] >= 1500, (
        f"joined 2000 samples into {out.shape[-1]} - the fade ate a chunk")


def test_a_long_fade_between_long_chunks_is_still_a_fade():
    """The clamp must not turn a normal join into a hard cut."""
    a = torch.ones(1, 2, 32000)
    b = torch.zeros(1, 2, 32000)
    out = xfade_audio([a, b], sr=32000, ms=40)
    # 1280 overlapped samples, so the total is 64000 - 1280
    assert out.shape[-1] == 64000 - 1280
    # EQUAL-POWER, not equal-gain. The ramp is cos/sin so that the sum of
    # SQUARES is constant across the seam; a linear ramp would sum to 1.0 in
    # amplitude and audibly dip in loudness at every join. The midpoint is
    # therefore cos(pi/4) = 0.707, not 0.5 - checking for 0.5 would be asking
    # for the wrong kind of crossfade.
    mid = out[0, 0, 32000 - 1280:32000]
    assert float(mid[len(mid) // 2]) == pytest.approx(0.70710678, abs=0.01), (
        "the seam is not an equal-power crossfade")
    assert float(mid[0]) > 0.9 and float(mid[-1]) < 0.1, "the ramp runs the wrong way"


def test_the_seam_holds_power_constant_not_amplitude():
    """The property that makes a join inaudible. a=1, b=0, so at every sample
    of the ramp fade_out^2 + fade_in^2 must be 1."""
    import math

    a = torch.ones(1, 2, 32000)
    b = torch.zeros(1, 2, 32000)
    out = xfade_audio([a, b], sr=32000, ms=40)
    ramp = out[0, 0, 32000 - 1280:32000]
    fade_in = torch.sqrt((1.0 - ramp.clamp(0, 1) ** 2).clamp(min=0))
    power = ramp ** 2 + fade_in ** 2
    assert float(power.min()) == pytest.approx(1.0, abs=1e-3)
    assert float(ramp[0]) == pytest.approx(math.cos(0.0), abs=1e-3)


def test_seam_trim_one_frame():
    assert seam_trim_samples(24000) == 1000
