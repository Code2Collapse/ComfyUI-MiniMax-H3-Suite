"""CPU-only tests for native exact-audio lock utilities."""

from __future__ import annotations

import copy

import pytest
import torch

from mmx_utils.native_audio import (
    assert_h3_av_latent,
    build_audio_lock_report,
    build_video_only_denoise_masks,
    fit_encoded_audio_to_target,
    fit_waveform_to_vae_rate,
    pack_locked_av_latent,
    prepare_mono_batch_waveform,
    resample_waveform,
)


class _FakeNested:
    is_nested = True

    def __init__(self, members):
        self._members = tuple(members)

    def unbind(self):
        return self._members


def _av_latent(video=None, audio=None):
    video = video if video is not None else torch.zeros(1, 24, 3, 8, 8)
    audio = audio if audio is not None else torch.zeros(1, 32, 2, 20)
    return {"samples": _FakeNested((video, audio))}


def test_denoise_mask_zeros_pin_audio_rows():
    video, audio = torch.ones(1, 24, 3, 8, 8), torch.ones(1, 32, 2, 15)
    vmask, amask = build_video_only_denoise_masks(video, audio)
    assert vmask.shape == video.shape
    assert amask.shape == audio.shape
    assert torch.all(vmask == 1.0)
    assert torch.all(amask == 0.0)


def test_audio_shorter_than_clip_is_padded_not_looped():
    encoded = torch.randn(1, 32, 2, 8)
    fitted, note = fit_encoded_audio_to_target(encoded, 20)
    assert fitted.shape[-1] == 20
    assert "padded" in note.lower()
    assert "drift" in note.lower()


def test_sample_rate_mismatch_resampled_and_reported():
    wav = torch.randn(1, 2, 4800)
    out, did = resample_waveform(wav, 48000, 32000)
    assert did
    assert out.shape[-1] != wav.shape[-1]
    out2, _, did2 = fit_waveform_to_vae_rate(wav, 48000, 32000)
    assert did2
    report = build_audio_lock_report(
        input_sample_rate=48000,
        vae_sample_rate=32000,
        resampled=True,
        audio_duration_s=0.1,
        clip_duration_s=0.5,
        fit_note="",
    )
    assert "resample: yes" in report
    assert "48000 Hz" in report
    assert "32000 Hz" in report
    assert "pitch" in report.lower()


def test_av_latent_round_trip_preserves_nested_structure():
    latent = _av_latent()
    video, audio_tmpl = assert_h3_av_latent(latent)
    encoded = torch.randn_like(audio_tmpl)
    vmask, amask = build_video_only_denoise_masks(video, encoded)
    locked = pack_locked_av_latent(latent, video, encoded, vmask, amask, _FakeNested)
    assert hasattr(locked["samples"], "unbind")
    v2, a2 = locked["samples"].unbind()
    assert v2.shape == video.shape
    assert a2.shape == encoded.shape
    mv, ma = locked["noise_mask"].unbind()
    assert mv.shape == video.shape
    assert ma.shape == encoded.shape


def test_non_h3_latent_refused_with_type_name():
    with pytest.raises(ValueError, match="NestedTensor"):
        assert_h3_av_latent({"samples": torch.zeros(1, 24, 3, 8, 8)})
    with pytest.raises(ValueError, match='no "samples"'):
        assert_h3_av_latent({})


def test_no_input_mutation():
    latent = _av_latent()
    audio = {"waveform": torch.randn(1, 2, 16000), "sample_rate": 32000}
    latent_copy = copy.deepcopy(latent)
    audio_copy = copy.deepcopy(audio)

    video, audio_tmpl = assert_h3_av_latent(latent)
    wav, _ = prepare_mono_batch_waveform(audio)
    encoded = torch.randn_like(audio_tmpl)
    vmask, amask = build_video_only_denoise_masks(video, encoded)
    pack_locked_av_latent(latent, video, encoded, vmask, amask, _FakeNested)

    assert torch.equal(latent["samples"].unbind()[0], latent_copy["samples"].unbind()[0])
    assert torch.equal(audio["waveform"], audio_copy["waveform"])
    assert audio["sample_rate"] == audio_copy["sample_rate"]


def test_shorter_audio_report_warns_tail_unlock():
    report = build_audio_lock_report(
        input_sample_rate=32000,
        vae_sample_rate=32000,
        resampled=False,
        audio_duration_s=2.0,
        clip_duration_s=5.0,
        fit_note="audio shorter than clip",
    )
    assert "video-only denoising" in report
    assert "shorter than the clip" in report


# ── the collision that makes faceswap + lipsync possible ────────────────────
#
# A swap sets a FACE region mask, then locks audio. The lock used to overwrite
# noise_mask with ones_like(video), so the face mask vanished and the whole
# frame regenerated - the swap silently stopped being a swap while every node
# still reported success. These pin that it survives.

def test_an_existing_region_mask_survives_the_audio_lock():
    from mmx_utils.native_audio import build_video_only_denoise_masks

    video = torch.zeros(1, 24, 7, 8, 12)
    audio = torch.zeros(1, 32, 2, 40)
    face = torch.zeros_like(video)
    face[..., 2:5, 3:7] = 1.0                 # only part of the frame is free

    vm, am = build_video_only_denoise_masks(video, audio, face)
    assert torch.equal(vm, face), "the region mask was replaced"
    assert float(am.max()) == 0.0, "audio must still be locked clean"


def test_without_a_region_mask_the_whole_frame_is_free():
    """Right for a music video, and the default."""
    from mmx_utils.native_audio import build_video_only_denoise_masks

    video = torch.zeros(1, 24, 7, 8, 12)
    audio = torch.zeros(1, 32, 2, 40)
    vm, _ = build_video_only_denoise_masks(video, audio, None)
    assert float(vm.min()) == 1.0


def test_the_existing_mask_is_found_on_a_plain_latent():
    from mmx_utils.native_audio import existing_video_mask

    video = torch.zeros(1, 24, 7, 8, 12)
    mask = torch.zeros_like(video)
    mask[..., 1:3, :] = 1.0
    found = existing_video_mask({"noise_mask": mask}, video)
    assert found is not None and torch.equal(found, mask)


def test_a_latent_with_no_mask_returns_none():
    from mmx_utils.native_audio import existing_video_mask

    assert existing_video_mask({}, torch.zeros(1, 24, 7, 8, 12)) is None


def test_a_mask_without_the_channel_axis_is_broadcast_not_discarded():
    """Set Latent Noise Mask hands over [B,T,H,W]; discarding it because the
    shape does not match exactly is how the face mask went missing."""
    from mmx_utils.native_audio import existing_video_mask

    video = torch.zeros(1, 24, 7, 8, 12)
    mask = torch.zeros(1, 1, 7, 8, 12)
    mask[..., 2:5, 3:7] = 1.0
    found = existing_video_mask({"noise_mask": mask}, video)
    assert found is not None and found.shape == video.shape
    assert float(found[0, 0, 0, 3, 4]) == 1.0


def test_the_report_warns_when_nothing_is_masked():
    from mmx_utils.native_audio import describe_region_mask

    text = describe_region_mask(None)
    assert "whole frame" in text and "swap" in text


def test_the_report_states_how_much_is_free():
    from mmx_utils.native_audio import describe_region_mask

    m = torch.zeros(1, 24, 7, 8, 12)
    m[..., :4, :] = 1.0
    assert "%" in describe_region_mask(m)
