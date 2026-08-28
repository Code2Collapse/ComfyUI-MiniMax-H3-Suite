from mmx_nodes.protected_layer_guard import MiniMaxH3_ProtectedLayerGuard
from mmx_utils.fake_patcher import FakeModelPatcher


def test_protected_layer_guard_passes_model():
    m = FakeModelPatcher()
    out = MiniMaxH3_ProtectedLayerGuard.execute(m)
    assert out[0] is m
    assert "protected layers OK" in out[1]
