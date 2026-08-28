import pytest

from mmx_nodes.accelerator_conflict import MiniMaxH3_AcceleratorConflict
from mmx_utils.fake_patcher import FakeModelPatcher
from mmx_utils.h3_block_cache import CACHE_KEY


def test_accelerator_conflict_passes_model():
    m = FakeModelPatcher()
    out = MiniMaxH3_AcceleratorConflict.execute(m)
    assert out[0] is m
    assert "OK" in out[1]


def test_accelerator_conflict_raises_on_spectrum_cache():
    m = FakeModelPatcher()
    m.model_options["spectrum_h3_binding"] = object()
    m.model_options["transformer_options"][CACHE_KEY] = object()
    with pytest.raises(ValueError, match="Spectrum"):
        MiniMaxH3_AcceleratorConflict.execute(m)
