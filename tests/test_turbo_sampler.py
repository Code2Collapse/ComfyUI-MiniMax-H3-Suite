import types

import pytest
import torch

from mmx_utils.turbo_sampler import native_av_schedule, turbo_sampler


class _MS:
    def __init__(self, audio_shift=None):
        self.audio_shift = audio_shift


class _Inner:
    def __init__(self, ms):
        self.model_sampling = ms


class _ModelWrap:
    def __init__(self, ms):
        self.inner_model = _Inner(ms)


def test_native_av_schedule_true_when_audio_shift_set():
    assert native_av_schedule(_ModelWrap(_MS(audio_shift=3.0))) is True


def test_native_av_schedule_false_without_model_sampling():
    assert native_av_schedule(object()) is False


def test_turbo_sampler_native_single_step():
    class _CallableWrap:
        def __init__(self, ms):
            self.inner_model = _Inner(ms)

        def __call__(self, x, sigma, **kwargs):
            return x * 0.5

    wrap = _CallableWrap(_MS(audio_shift=3.0))
    x0 = torch.ones(1, 8)
    sigmas = torch.tensor([1.0, 0.0])
    out = turbo_sampler(wrap, x0.clone(), sigmas, disable=True)
    assert out.shape == x0.shape
    assert not torch.allclose(out, x0)


def test_turbo_sampler_legacy_requires_shapes():
    wrap = _ModelWrap(_MS())
    x0 = torch.ones(1, 16)
    sigmas = torch.tensor([1.0, 0.0])
    with pytest.raises(RuntimeError, match="video\\+audio latent"):
        turbo_sampler(wrap, x0, sigmas, disable=True)


def test_turbo_sampler_node_returns_ksampler():
    pytest.importorskip("comfy.samplers")
    from mmx_nodes.turbo_sampler import MiniMaxH3_TurboSampler

    out = MiniMaxH3_TurboSampler.execute()
    sampler = out[0]
    assert hasattr(sampler, "sampler_function")
    assert sampler.sampler_function is turbo_sampler
