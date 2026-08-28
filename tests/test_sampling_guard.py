import pytest

from mmx_utils.fake_patcher import FakeModelPatcher
from mmx_utils.h3_block_cache import CACHE_KEY
from mmx_utils.sampling_guard import (
    check_accelerator_conflicts,
    check_dit_replace_conflicts_before_install,
    check_protected_layers,
    infer_total_blocks,
)


def test_infer_total_blocks():
    m = FakeModelPatcher()
    assert infer_total_blocks(m) == 50


def test_accelerator_conflict_spectrum_and_cache():
    m = FakeModelPatcher()
    m.model_options["transformer_options"][CACHE_KEY] = object()
    m.model_options["spectrum_h3_binding"] = object()
    with pytest.raises(ValueError, match="Spectrum"):
        check_accelerator_conflicts(m.model_options, total_blocks=50)


def test_accelerator_conflict_easycache_and_block_cache():
    m = FakeModelPatcher()
    m.model_options["transformer_options"]["easycache"] = object()
    m.model_options["transformer_options"][CACHE_KEY] = object()
    with pytest.raises(ValueError, match="EasyCache"):
        check_accelerator_conflicts(m.model_options, total_blocks=50)


def test_dit_replace_refuses_overwrite():
    existing = {("double_block", 0): object()}
    with pytest.raises(ValueError, match="already occupied"):
        check_dit_replace_conflicts_before_install(existing, [("double_block", 0)])


def test_protected_layer_blocks_union_index():
    m = FakeModelPatcher()
    m.model_options["transformer_options"]["patches_replace"] = {
        "dit": {("double_block", 10): object()}
    }
    with pytest.raises(ValueError, match="protected DiT block index 10"):
        check_protected_layers(m.model_options, total_blocks=50)


def _with_dit(*keys, cache=False):
    m = FakeModelPatcher()
    to = m.model_options["transformer_options"]
    if cache:
        to[CACHE_KEY] = object()
    to["patches_replace"] = {"dit": {k: object() for k in keys}}
    return m


def test_protected_layer_allows_block_cache_hooks_on_first_and_last():
    """The Block Cache T8 exemption: hooks on block 0 and the last block are legal
    ONLY while the cache is installed (`utils/sampling_guard.py:124-126`)."""
    m = _with_dit(("double_block", 0), ("double_block", 49), cache=True)
    check_protected_layers(m.model_options, total_blocks=50)  # must not raise


def test_protected_layer_exemption_is_conditional_not_blanket():
    """The same hooks WITHOUT the cache key must be rejected.

    This is the assertion that stops the previous test from being vacuous: a guard
    that never raised would have passed it. Here the identical patch set must fail
    purely because `CACHE_KEY` is absent, which proves the exemption is scoped to
    the cache rather than being a permanent hole in blocks 0 and last.
    """
    m = _with_dit(("double_block", 0), ("double_block", 49), cache=False)
    with pytest.raises(ValueError) as e:
        check_protected_layers(m.model_options, total_blocks=50)
    assert "protected DiT block" in str(e.value)


@pytest.mark.parametrize("idx", [10, 20, 30, 40])
def test_protected_layer_rejects_union_blocks_even_with_cache(idx):
    """Union injection blocks are protected even when the cache IS installed —
    the exemption covers only 0 and last."""
    m = _with_dit(("double_block", idx), cache=True)
    with pytest.raises(ValueError) as e:
        check_protected_layers(m.model_options, total_blocks=50)
    assert str(idx) in str(e.value)
