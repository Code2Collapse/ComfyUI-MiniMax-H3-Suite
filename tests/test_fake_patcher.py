import pytest

from mmx_utils.fake_patcher import FakeModelPatcher


def test_clone_is_independent():
    m = FakeModelPatcher()
    c = m.clone()
    c.model_options["transformer_options"]["x"] = 1
    assert "x" not in m.model_options["transformer_options"]


def test_set_model_patch_replace_occupies_dit():
    m = FakeModelPatcher()
    m.set_model_patch_replace(object(), "dit", "double_block", 3)
    dit = m.model_options["transformer_options"]["patches_replace"]["dit"]
    assert ("double_block", 3) in dit


def test_set_model_patch_replace_refuses_duplicate():
    m = FakeModelPatcher()
    m.set_model_patch_replace(object(), "dit", "double_block", 3)
    with pytest.raises(ValueError, match="occupied"):
        m.set_model_patch_replace(object(), "dit", "double_block", 3)
