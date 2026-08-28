import pytest
import torch

from mmx_utils.av_latent import (
    assert_nested_samples,
    build_av_noise_mask,
    expand_video_latent_mask,
    fit_audio_latent_length,
    fit_temporal_dim,
    inject_video_stream,
    validate_canvas_multiple,
    validate_h3_pixel_frame_count,
    validate_spatial_latent,
)


class _FakeNested:
    def __init__(self, members):
        self._members = tuple(members)

    def unbind(self):
        return self._members


def _av_pair(t_lat=3, h=8, w=8, audio_t=10):
    video = torch.zeros(1, 24, t_lat, h, w)
    audio = torch.zeros(1, 32, 2, audio_t)
    return video, audio


def test_assert_nested_rejects_plain_tensor():
    with pytest.raises(ValueError, match="NestedTensor"):
        assert_nested_samples(torch.zeros(1, 24, 3, 8, 8))


def test_validate_h3_frame_count():
    validate_h3_pixel_frame_count(22)
    with pytest.raises(ValueError, match="17n\\+5"):
        validate_h3_pixel_frame_count(21)


def test_validate_canvas_multiple():
    validate_canvas_multiple(512, 512)
    with pytest.raises(ValueError, match="multiple"):
        validate_canvas_multiple(500, 512)


def test_inject_video_stream_spatial_mismatch():
    video, audio = _av_pair()
    enc = torch.zeros(1, 24, 3, 7, 8)
    with pytest.raises(ValueError, match="Spatial latent mismatch"):
        validate_spatial_latent(enc, video)


def test_fit_temporal_trim_and_pad():
    video, _ = _av_pair(t_lat=3)
    enc = torch.zeros(1, 24, 5, 8, 8)
    fitted, note = fit_temporal_dim(enc, video)
    assert fitted.shape[-3] == 3
    assert "trimmed" in note

    enc2 = torch.zeros(1, 24, 1, 8, 8)
    fitted2, note2 = fit_temporal_dim(enc2, video)
    assert fitted2.shape[-3] == 3
    assert "padded" in note2


def test_inject_and_noise_mask_shapes():
    video, audio = _av_pair()
    members = [video, audio]
    enc = torch.ones_like(video)
    members, _ = inject_video_stream(members, enc, video)
    mask2d = torch.ones(3, 8, 8)
    vmask, amask = build_av_noise_mask(mask2d, members[0], members[1], audio_locked=True)
    assert vmask.shape == video.shape
    assert amask.shape == audio.shape
    assert amask.sum() == 0
    vmask2, amask2 = build_av_noise_mask(mask2d, members[0], members[1], audio_locked=False)
    assert amask2.sum() > 0


def test_fit_audio_latent_length():
    z = torch.zeros(1, 32, 2, 5)
    out = fit_audio_latent_length(z, 8)
    assert out.shape[-1] == 8
