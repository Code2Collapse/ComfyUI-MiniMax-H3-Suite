"""Context taper noise (ported from beijinren/ComfyUI-H3-Context-Noise, MIT): the validated schedule, frames before the
window untouched, the latent path noises exactly the pinned steps and never the audio, containers round-trip, and
the seed reproduces everything."""
from __future__ import annotations

import pytest
import torch

from mmx_nodes.context_taper_noise import taper_frames, taper_latent
from mmx_utils.context_taper import TaperError, schedule, steps_for_frames


def test_the_validated_recipe_schedule():
    s = schedule(22, 0.45, 0.10, 3)
    assert s[:19] == [0.45] * 19
    assert [round(x, 4) for x in s[19:]] == [0.3333, 0.2167, 0.1]


@pytest.mark.parametrize("bad", [(0, 0.4, 0.1, 1), (5, 0.4, 0.1, 6), (5, 0.2, 0.3, 2), (5, 1.2, 0.1, 2)])
def test_bad_parameters_are_refused(bad):
    with pytest.raises(TaperError):
        schedule(*bad)


def test_latent_steps_for_frames():
    assert [steps_for_frames(n) for n in (5, 22, 39, 56)] == [2, 7, 12, 17]
    assert steps_for_frames(20) is None


def test_frames_only_the_tail_changes_and_alpha_matches():
    img = torch.full((30, 64, 48, 3), 0.5)
    out, text = taper_frames(img, 22, 0.45, 0.10, 3, seed=7)
    assert torch.equal(out[:8], img[:8])                      # the 8 frames before the window pass through
    assert not torch.equal(out[8:], img[8:])
    noise_last = (out[-1] - 0.5 * (1 - 0.10)) / 0.10           # out = f(1-a) + n a  ->  n in [0, 1]
    assert float(noise_last.min()) >= -1e-5 and float(noise_last.max()) <= 1 + 1e-5
    assert "<- seam" in text and "frames 8..29 of 30" in text
    again, _ = taper_frames(img, 22, 0.45, 0.10, 3, seed=7)
    assert torch.equal(out, again)                             # one seed, one result
    other, _ = taper_frames(img, 22, 0.45, 0.10, 3, seed=8)
    assert not torch.equal(out, other)


def test_frames_alpha_channel_kept_and_short_clips_noise_what_there_is():
    img = torch.rand(5, 32, 32, 4)
    out, text = taper_frames(img, 22, 0.45, 0.10, 3, seed=1, grid_mode="block_size", block_size=8)
    assert torch.equal(out[..., 3], img[..., 3])
    assert "only 5 frames arrived" in text


def _nested(video, audio):
    nt = pytest.importorskip("comfy.nested_tensor")
    return {"samples": nt.NestedTensor((video, audio))}


def test_latent_noises_the_pinned_steps_only_and_never_the_audio():
    video = torch.randn(1, 48, 32, 4, 6)                        # 32 steps: the 22-frame tail = the last 7 steps
    audio = torch.randn(1, 32, 2, 207)
    lat = _nested(video.clone(), audio.clone())
    out, text = taper_latent(lat, "22", 0.45, 0.10, 2, seed=3)
    v, a = out["samples"].unbind()
    assert type(out["samples"]).__name__ == "NestedTensor"
    assert torch.equal(v[:, :, :25], video[:, :, :25]) and not torch.equal(v[:, :, 25:], video[:, :, 25:])
    assert torch.equal(a, audio)
    assert "steps 25..31 of 32" in text and "audio untouched" in text


def test_latent_list_container_kept_and_plain_latents_refused():
    video, audio = torch.randn(1, 48, 12, 2, 2), torch.randn(1, 32, 2, 20)
    out, _ = taper_latent({"samples": [video, audio]}, "5", 0.45, 0.10, 1, seed=0)
    assert isinstance(out["samples"], list) and torch.equal(out["samples"][1], audio)
    with pytest.raises(ValueError, match="single-stream"):
        taper_latent({"samples": torch.randn(1, 16, 12, 2, 2)}, "5", 0.45, 0.10, 1, seed=0)
    with pytest.raises(ValueError, match="only 1"):
        taper_latent({"samples": [torch.randn(1, 48, 1, 2, 2), audio]}, "5", 0.45, 0.10, 1, seed=0)
