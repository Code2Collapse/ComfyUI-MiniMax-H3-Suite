import torch

from mmx_utils.mask_to_token import pixel_mask_to_h3_latent, quantize_soft_mask


def test_mask_token_quantize_steps():
    m = torch.tensor([0.0, 0.001, 0.5, 1.0])
    q = quantize_soft_mask(m)
    steps = torch.unique(q)
    for s in steps:
        assert abs(s.item() * 256 - round(s.item() * 256)) < 1e-6


def test_mask_to_token_shape():
    h, w, fc = 256, 384, 22
    mask = torch.zeros(fc, h, w)
    mask[:, 80:180, 120:220] = 1.0
    lat = pixel_mask_to_h3_latent(mask, width=w, height=h, frame_count=fc)
    assert lat.shape[0] == 7
    assert lat.shape[1] == h // 16
    assert lat.shape[2] == w // 16
    assert lat.max() <= 1.0
    assert lat.min() >= 0.0


def test_temporal_groups_no_bleed():
    h, w = 64, 64
    fc = 22
    mask = torch.zeros(fc, h, w)
    mask[0, 20:40, 20:40] = 1.0
    mask[10, 20:40, 20:40] = 1.0
    lat = pixel_mask_to_h3_latent(mask, width=w, height=h, frame_count=fc)
    assert lat[0].max() > 0.5
    assert lat[2].max() < 0.5 or lat[2].sum() == 0
