import torch

from mmx_utils.mask_clean import spatial_dilate, temporal_despeckle


def test_temporal_despeckle_noop_on_radius_zero():
    m = torch.rand(4, 16, 16)
    assert torch.equal(temporal_despeckle(m, 0), m)


def test_spatial_dilate_expands_support():
    m = torch.zeros(1, 16, 16)
    m[:, 8, 8] = 1.0
    d = spatial_dilate(m, 2)
    assert d.sum() > m.sum()
