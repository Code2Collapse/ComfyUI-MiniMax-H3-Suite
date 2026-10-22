"""H3 AV Bridge / Assemble (ported from seitanism/ComfyUI-H3-Motion-Context-MultiRef, GPL-3.0): both contexts land on
the right latent steps and audio ticks, the middle is the only generated part, and the assembly is sample-exact."""
from __future__ import annotations

import json

import pytest
import torch

nt = pytest.importorskip("comfy.nested_tensor")

from mmx_nodes.av_bridge import prepare_bridge  # noqa: E402
from mmx_utils.h3_bridge import BridgeError, assemble, check_timing, pixel_frames  # noqa: E402

FPT = (1, 4, 4, 4, 4)


def steps_for(n):
    k = c = 0
    while c < n:
        c += FPT[k % 5]
        k += 1
    return k


class VideoVAE:
    def encode(self, images):            # [N, H, W, 3] -> [1, 48, steps, H/16, W/16]; value = mean pixel
        n, h, w = int(images.shape[0]), int(images.shape[1]), int(images.shape[2])
        return torch.full((1, 48, steps_for(n), h // 16, w // 16), float(images.mean()))


class AudioVAE:
    audio_sample_rate = 32000

    def encode(self, wave):              # [B, samples, 2] -> [1, 32, 2, samples / 800]
        assert int(wave.shape[1]) % 800 == 0, "the bridge must hand the VAE an exact grid"
        return torch.full((1, 32, 2, int(wave.shape[1]) // 800), float(wave.mean()))


def target(frames=192, h=4, w=6):
    t = steps_for(frames)
    return {"samples": nt.NestedTensor((torch.zeros(1, 48, t, h, w), torch.zeros(1, 32, 2, round(frames / 24 * 40))))}


def clip(n, value, h=64, w=96):
    return torch.full((n, h, w, 3), value)


def sound(frames, value, sr=32000):
    return {"waveform": torch.full((1, 2, round(frames / 24 * sr)), value), "sample_rate": sr}


def test_timing_rules():
    check_timing(192, 39)
    for t, p in ((190, 39), (192, 40), (73, 39)):
        with pytest.raises(BridgeError):
            check_timing(t, p)


def test_bridge_protects_both_ends_and_generates_only_the_middle():
    out, info, middle, report = prepare_bridge(target(), VideoVAE(), AudioVAE(), clip(50, 0.2), sound(50, 0.1),
                                               clip(60, 0.8), sound(60, 0.4), 39)
    v, a = out["samples"].unbind()
    mv, ma = out["noise_mask"].unbind()
    assert v.shape[2] == 57 and a.shape[-1] == 320
    kept_v = [k for k in range(57) if float(mv[0, 0, k].max()) == 0.0]
    assert kept_v == list(range(12)) + list(range(45, 57))
    kept_a = [k for k in range(320) if float(ma[0, 0, 0, k]) == 0.0]
    assert kept_a == list(range(65)) + list(range(255, 320))
    assert abs(float(v[0, 0, 0].mean()) - 0.2) < 1e-6 and abs(float(v[0, 0, 56].mean()) - 0.8) < 1e-6
    assert abs(float(a[0, 0, 0, 0]) - 0.1) < 1e-6 and abs(float(a[0, 0, 0, 319]) - 0.4) < 1e-6
    assert middle == 114 and json.loads(info)["preserve_frames"] == 39 and "generate the middle 114" in report


def test_bridge_refusals():
    with pytest.raises(BridgeError, match="at least 39"):
        prepare_bridge(target(), VideoVAE(), AudioVAE(), clip(20, 0.2), None, clip(60, 0.8), None, 39)
    with pytest.raises(BridgeError, match="longer than both"):
        prepare_bridge(target(73), VideoVAE(), AudioVAE(), clip(50, 0.2), None, clip(60, 0.8), None, 39)


def test_assemble_is_frame_and_sample_exact():
    tgt, pres = 192, 39
    images = torch.arange(tgt, dtype=torch.float32).view(-1, 1, 1, 1).expand(tgt, 64, 96, 3) / 1000.0
    bridge_sound = {"waveform": torch.zeros(1, 2, 320 * 800), "sample_rate": 32000}
    a_n, b_n = 50, 60
    frames, snd, report = assemble(images, bridge_sound, clip(a_n, 0.5), sound(a_n, 0.25), clip(b_n, 0.9),
                                   sound(b_n, 0.75), tgt, pres)
    assert int(frames.shape[0]) == a_n + (tgt - 2 * pres) + b_n == 224
    assert abs(float(frames[a_n, 0, 0, 0]) - 0.039) < 1e-6           # the first generated frame is bridge frame 39
    assert abs(float(frames[a_n + 113, 0, 0, 0]) - 0.152) < 1e-6
    w = snd["waveform"]
    assert int(w.shape[-1]) == round(224 / 24 * 32000)
    a_end, m_end = round(a_n / 24 * 32000), round((a_n + 114) / 24 * 32000)
    assert float(w[..., :a_end].min()) == 0.25 and float(w[..., m_end:].min()) == 0.75
    assert "exactly the picture" in report


def test_assemble_resizes_clips_and_checks_lengths():
    images = torch.zeros(192, 64, 96, 3)
    snd = {"waveform": torch.zeros(1, 2, 320 * 800), "sample_rate": 32000}
    frames, _, report = assemble(images, snd, clip(40, 0.5, 128, 192), None, clip(40, 0.5, 32, 48), None, 192, 39)
    assert tuple(frames.shape[1:3]) == (64, 96) and "resized" in report
    with pytest.raises(BridgeError, match="Decode the whole"):
        assemble(images[:100], snd, clip(40, 0.5), None, clip(40, 0.5), None, 192, 39)


def test_pixel_frames_matches_h3():
    assert pixel_frames(57) == 192 and pixel_frames(12) == 39
