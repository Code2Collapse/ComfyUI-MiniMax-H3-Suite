"""H3 Set / Clear AV Noise Mask and H3 Hard Keyframes (ported from seitanism/ComfyUI-H3-Motion-Context-MultiRef,
GPL-3.0, with the frame mask reduced on H3's own grid instead of trilinear resampling)."""
from __future__ import annotations

import pytest
import torch

nt = pytest.importorskip("comfy.nested_tensor")

from mmx_nodes.av_noise_mask import (  # noqa: E402
    MiniMaxH3_ClearAVNoiseMask,
    audio_mask_for,
    hard_keyframes,
    set_av_noise_mask,
    video_mask_for,
)

T, H, W, TICKS = 7, 4, 6, 37          # 22 frames, 96 x 64 px canvas


def _latent(mask=None):
    lat = {"samples": nt.NestedTensor((torch.randn(1, 48, T, H, W), torch.randn(1, 32, 2, TICKS)))}
    if mask is not None:
        lat["noise_mask"] = mask
    return lat


def test_the_video_mask_lands_on_h3_steps_not_smeared():
    m = torch.zeros(22, H * 16, W * 16)
    m[2] = 1.0                                     # frame 2 belongs to latent step 1 (frames 1-4)
    v, note = video_mask_for(m, T, H, W)
    assert v.shape == (1, 1, T, H, W) and "H3's grid" in note
    marked = [k for k in range(T) if float(v[0, 0, k].max()) > 0]
    assert marked == [1]                           # trilinear resampling would bleed into steps 0 and 2
    m2 = torch.zeros(22, H * 16, W * 16)
    m2[17, :20, :20] = 1.0                         # frame 17 = single-frame step 5; 20 px spans token (0, 0) only
    v2, _ = video_mask_for(m2, T, H, W)
    assert torch.equal(v2[0, 0, 5, :2, :2], torch.ones(2, 2)) and float(v2[0, 0, 5].sum()) == 4.0


def test_wrong_sizes_are_refused_with_the_reason():
    with pytest.raises(ValueError, match="21 frames"):
        video_mask_for(torch.zeros(21, H * 16, W * 16), T, H, W)
    with pytest.raises(ValueError, match="Resize the mask"):
        video_mask_for(torch.zeros(22, 60, 90), T, H, W)


def test_audio_ticks_take_the_largest_value_of_their_frames():
    m = torch.zeros(22, 8, 8)
    m[10, 0, 0] = 0.5                              # frame 10 = 0.417 s -> ticks 16 and 17
    a, _ = audio_mask_for(m, TICKS)
    assert a.shape == (1, 1, 2, TICKS)
    hit = [k for k in range(TICKS) if float(a[0, 0, 0, k]) > 0]
    assert hit and all(abs(float(a[0, 0, 0, k]) - 0.5) < 1e-6 for k in hit)
    assert all(15 <= k <= 18 for k in hit)


def test_set_keeps_the_other_stream_and_clear_removes_both():
    keep_a = torch.zeros(1, 1, 2, TICKS)
    lat = _latent(nt.NestedTensor((torch.ones(1, 1, T, H, W), keep_a)))
    out, report = set_av_noise_mask(lat, video_mask=torch.zeros(22, H * 16, W * 16))
    v, a = out["noise_mask"].unbind()
    assert torch.equal(a, keep_a) and float(v.max()) == 0.0
    assert "audio mask kept" in report
    cleared = MiniMaxH3_ClearAVNoiseMask.execute(out)
    cleared = cleared.args[0] if hasattr(cleared, "args") else cleared[0]
    assert "noise_mask" not in cleared and cleared["samples"] is out["samples"]
    with pytest.raises(ValueError, match="connect video_mask"):
        set_av_noise_mask(lat)
    with pytest.raises(ValueError, match="single-stream"):
        set_av_noise_mask({"samples": torch.randn(1, 16, T, H, W)}, video_mask=torch.zeros(22, 64, 96))


class _VAE:
    def __init__(self):
        self.calls = []

    def encode(self, image):
        self.calls.append(tuple(image.shape))
        return torch.full((1, 48, 1, image.shape[1] // 16, image.shape[2] // 16), float(len(self.calls)))


def test_hard_keyframes_write_and_protect_their_steps():
    vae = _VAE()
    lat = _latent()
    imgs = torch.rand(2, 40, 50, 3)
    out, report = hard_keyframes(lat, vae, imgs, "1, 18")
    v, a = out["samples"].unbind()
    mv, ma = out["noise_mask"].unbind()
    assert vae.calls == [(1, 64, 96, 3)] * 2                         # each still resized to the canvas
    assert float(v[0, 0, 0].mean()) == 1.0 and float(v[0, 0, 5].mean()) == 2.0
    assert [k for k in range(T) if float(mv[0, 0, k].max()) == 0.0] == [0, 5]
    assert torch.equal(a, lat["samples"].unbind()[1]) and float(ma.min()) == 1.0
    assert "latent steps [0, 5]" in report


def test_hard_keyframes_explain_interior_steps_and_refuse_conflicts():
    _, report = hard_keyframes(_latent(), _VAE(), torch.rand(1, 64, 96, 3), "3")
    assert "4-frame latent step" in report and "single-frame positions are 1, 18" in report
    with pytest.raises(ValueError, match="same latent step"):
        hard_keyframes(_latent(), _VAE(), torch.rand(2, 64, 96, 3), "2, 3")
    with pytest.raises(ValueError, match="2 positions but 1 images"):
        hard_keyframes(_latent(), _VAE(), torch.rand(1, 64, 96, 3), "1, 18")
    with pytest.raises(ValueError, match="outside 1..22"):
        hard_keyframes(_latent(), _VAE(), torch.rand(1, 64, 96, 3), "23")
    vm = torch.ones(1, 1, T, H, W)
    vm[:, :, 5] = 0.0
    with pytest.raises(ValueError, match="already protects"):
        hard_keyframes(_latent(nt.NestedTensor((vm, torch.ones(1, 1, 2, TICKS)))), _VAE(), torch.rand(1, 64, 96, 3),
                       "18")
    with pytest.raises(ValueError, match="single-tensor noise mask"):
        hard_keyframes(_latent(torch.ones(1, 1, T, H, W)), _VAE(), torch.rand(1, 64, 96, 3), "1")
