import torch

from mmx_nodes.mask_prep import MiniMaxH3_MaskPrep


def test_mask_prep_execute():
    mask = torch.zeros(22, 128, 128)
    mask[:, 40:88, 40:88] = 1.0
    out = MiniMaxH3_MaskPrep.execute(mask, width=512, height=512, frame_count=22)
    latent_mask, preview, report = out[0], out[1], out[2]
    assert latent_mask.ndim == 3
    assert "N4" in report
