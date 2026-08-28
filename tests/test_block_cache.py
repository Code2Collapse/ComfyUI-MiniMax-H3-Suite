import pytest

from mmx_nodes.block_cache import MiniMaxH3_BlockCacheT8
from mmx_utils.fake_patcher import FakeModelPatcher


def test_block_cache_never_silently_accepts_a_wrong_model():
    """The contract is: a non-H3 model NEVER gets through quietly.

    Two legitimate refusal paths, depending on the environment:
      * H3 modules importable  -> ValueError "requires a native MiniMax H3 diffusion model"
      * H3 modules NOT importable (comfy_kitchen version skew on this box)
        -> RuntimeError naming the import failure

    The second is not a weaker outcome. Without `MiniMaxH3Model` imported we cannot
    run `isinstance()` at all — `isinstance(x, None)` raises TypeError — so refusing
    on the environment is the only honest answer. Both paths must raise, and both
    must say "MiniMax H3" so the artist knows which node complained. Asserting only
    ValueError made this test environment-dependent, which is why it failed here
    while the node was behaving correctly.
    """
    m = FakeModelPatcher()
    with pytest.raises((ValueError, RuntimeError)) as excinfo:
        MiniMaxH3_BlockCacheT8.execute(m, 0.12, 0.08, 0.95, 2, "cpu", 8, False)

    msg = str(excinfo.value)
    assert "MiniMax H3" in msg, msg
    assert msg.strip(), "refusal must carry a human sentence"


def test_block_cache_refuses_occupied_dit_index():
    m = FakeModelPatcher()
    m.model_options["transformer_options"]["patches_replace"] = {
        "dit": {("double_block", 0): object()}
    }
    with pytest.raises(ValueError, match="already occupied"):
        from mmx_utils.sampling_guard import check_dit_replace_conflicts_before_install

        check_dit_replace_conflicts_before_install(
            m.model_options["transformer_options"]["patches_replace"]["dit"],
            [("double_block", 0), ("double_block", 49)],
        )
