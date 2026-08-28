import pytest

pytest.importorskip("comfy.model_sampling")

from mmx_nodes.sigma_shift_locked import MiniMaxH3_SigmaShiftLocked
from mmx_utils.fake_patcher import FakeModelPatcher


class _FakeSampling:
    noise_scale = 1.0


def test_sigma_shift_locked_ratio_lock():
    m = FakeModelPatcher()
    m._sampling = _FakeSampling()
    out = MiniMaxH3_SigmaShiftLocked.execute(m, 12.0, 5.0, ratio_lock=True, ratio=4.0)
    patched = out[0]
    to = patched.model_options["transformer_options"]
    assert to["minimax_h3_sigma_shift_video"] == 12.0
    assert to["minimax_h3_sigma_shift_audio"] == 3.0
    assert "ratio_lock" in out[1]
