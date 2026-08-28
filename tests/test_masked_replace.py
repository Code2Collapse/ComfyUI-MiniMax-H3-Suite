import pytest

from mmx_nodes.masked_replace import MiniMaxH3_MaskedReplace


def test_masked_replace_schema():
    schema = MiniMaxH3_MaskedReplace.define_schema()
    assert schema.node_id == "MiniMaxH3_MaskedReplace"
    assert "BasicScheduler" in schema.description


def test_fingerprint_deterministic():
    import torch

    crops = torch.rand(22, 64, 64, 3)
    mask = torch.rand(3, 4, 4)
    a = MiniMaxH3_MaskedReplace.fingerprint_inputs({}, crops, None, mask)
    b = MiniMaxH3_MaskedReplace.fingerprint_inputs({}, crops, None, mask)
    assert a == b
