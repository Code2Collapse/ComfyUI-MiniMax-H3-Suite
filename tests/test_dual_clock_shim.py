from mmx_nodes.dual_clock_shim import MiniMaxH3_DualClockShim


class _FakeSampling:
    audio_shift = 3.0


class _FakeModelNative:
    model_options = {}

    def __init__(self):
        self.model_sampling = _FakeSampling()


class _FakeModelLegacy:
    model_options = {}

    def __init__(self):
        self.model_sampling = object()


def test_dual_clock_shim_native_av_passes_through():
    m = _FakeModelNative()
    out = MiniMaxH3_DualClockShim.execute(m)
    assert out[0] is m
    assert "native ModelSamplingAV" in out[1]
    assert "no shim" in out[1].lower()


def test_dual_clock_shim_legacy_documents_turbo_sampler():
    m = _FakeModelLegacy()
    out = MiniMaxH3_DualClockShim.execute(m)
    assert out[0] is m
    assert "TurboSampler" in out[1]
    assert "passed through unchanged" in out[1]
