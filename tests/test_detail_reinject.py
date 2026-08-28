import torch

from mmx_nodes.detail_reinject import MiniMaxH3_DetailReinject


def test_detail_reinject_execute_cpu():
    plate = torch.rand(2, 32, 32, 3)
    gen = torch.rand(2, 32, 32, 3)
    mask = torch.zeros(2, 32, 32)
    mask[:, 8:24, 8:24] = 1.0
    out = MiniMaxH3_DetailReinject.execute(plate, gen, mask, strength=0.5)
    images, report = out[0], out[1]
    assert images.shape == plate.shape
    assert torch.isfinite(images).all()
