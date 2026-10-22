"""H3 Blend Latents By Frames, H3 Keyframe Mask, H3 Mask Levels (ported from xyzDist/H3-LongTakeNoCuts, MIT, with
keyframes mapped onto H3's real 1,4,4,4,4 latent steps) and the Set AV Noise Mask lock switches."""
from __future__ import annotations

import pytest
import torch

nt = pytest.importorskip("comfy.nested_tensor")

from mmx_nodes.av_noise_mask import set_av_noise_mask  # noqa: E402
from mmx_nodes.latent_blend_frames import (  # noqa: E402
    blend_by_frames,
    curve,
    keyframe_mask,
    mask_levels,
    parse_keyframes,
    step_weights,
)

T, H, W, TICKS = 12, 2, 3, 65          # 39 frames


def _lat(fill_v, fill_a):
    return {"samples": nt.NestedTensor((torch.full((1, 48, T, H, W), fill_v), torch.full((1, 32, 2, TICKS), fill_a)))}


def test_keyframes_parse_and_the_curve_holds_its_ends():
    assert parse_keyframes("44:1, 0:0, 22:0, 22:0.5") == [(0, 0.0), (22, 0.5), (44, 1.0)]
    c = curve([(10, 0.0), (20, 1.0)], 30)
    assert c[:11] == [0.0] * 11 and c[20:] == [1.0] * 10 and abs(c[15] - 0.5) < 1e-9
    assert curve([(0, 0.0), (10, 1.0)], 11, "hold")[9] == 0.0
    with pytest.raises(ValueError, match="frame:value"):
        parse_keyframes("22-1")


def test_step_weights_follow_h3_steps_not_a_linear_ratio():
    fw = [0.0] * 22 + [1.0] * 17                     # frames 0..21 latent 1, 22..38 latent 2
    sw = step_weights(fw, T)
    assert sw[:7] == [0.0] * 7                       # steps 0..6 hold frames 0..21 exactly
    assert sw[7:] == [1.0] * 5                       # step 7 starts at frame 22
    # the old mapping round(22 * 12 / 39) = 7 happens to agree here; at frame 18 it does not:
    sw18 = step_weights([0.0] * 18 + [1.0] * 21, T)
    assert sw18[5] == 0.0 and sw18[6] == 1.0          # frame 17 = step 5 (1 frame), 18..21 = step 6


def test_blend_uses_the_weights_and_keeps_the_chosen_audio():
    out, report = blend_by_frames(_lat(0.0, 0.0), _lat(1.0, 2.0), "0:0, 21:0, 22:1")
    v, a = out["samples"].unbind()
    assert [round(float(v[0, 0, k, 0, 0]), 4) for k in range(T)] == [0.0] * 7 + [1.0] * 5
    assert float(a.max()) == 0.0 and "noise_mask" not in out and "starts at frame 22" in report
    out2, _ = blend_by_frames(_lat(0.0, 0.0), _lat(1.0, 2.0), "0:1", audio="blend")
    assert float(out2["samples"].unbind()[1].min()) == 2.0


def test_blend_mask_limits_where_latent2_is_used():
    m = torch.zeros(39, H * 16, W * 16)
    m[:, :, :16] = 1.0                               # only the left latent column may change
    out, _ = blend_by_frames(_lat(0.0, 0.0), _lat(1.0, 0.0), "0:1", mask=m)
    v = out["samples"].unbind()[0]
    assert float(v[0, 0, :, :, 0].min()) == 1.0 and float(v[0, 0, :, :, 1:].max()) == 0.0
    with pytest.raises(ValueError, match="differ"):
        blend_by_frames(_lat(0.0, 0.0), {"samples": nt.NestedTensor((torch.zeros(1, 48, 7, H, W),
                                                                      torch.zeros(1, 32, 2, 37)))}, "0:1")


def test_keyframe_mask_and_levels():
    m = keyframe_mask("15:1, 22:0, 30:1", 40, 8, 6)
    assert m.shape == (40, 6, 8) and float(m[0].max()) == 1.0 and float(m[22].max()) == 0.0
    assert abs(float(m[18, 0, 0]) - (1 - 3 / 7)) < 1e-6
    lv = mask_levels(torch.tensor([0.0, 0.5, 1.0]), 0.3, 0.6, 1.0)
    assert torch.allclose(lv, torch.tensor([0.3, 0.45, 0.6]))


def test_video_lock_refines_only_the_sound():
    lat = _lat(0.0, 0.0)
    out, report = set_av_noise_mask(lat, video_lock=True)
    mv, ma = out["noise_mask"].unbind()
    assert float(mv.max()) == 0.0 and float(ma.min()) == 1.0 and "video locked" in report
    out2, _ = set_av_noise_mask(lat, audio_lock=True, video_mask=torch.ones(39, H * 16, W * 16))
    mv2, ma2 = out2["noise_mask"].unbind()
    assert float(ma2.max()) == 0.0 and float(mv2.min()) == 1.0
