import torch

from mmx_utils.affine_transform import affine_crop, inverse_affine_warp_batch


def _smooth_plate(h, w):
    yy, xx = torch.meshgrid(
        torch.linspace(0.0, 1.0, h), torch.linspace(0.0, 1.0, w), indexing="ij"
    )
    return torch.stack([xx, yy, (xx + yy) * 0.5], dim=-1).unsqueeze(0)


def test_affine_roundtrip_interior_exact():
    h, w = 128, 192
    plate = _smooth_plate(h, w)
    box = (40.0, 30.0, 80.0, 70.0)
    crop = affine_crop(plate, box, 96, 96)
    warped = inverse_affine_warp_batch(crop, [box], w, h)
    y0, x0 = int(box[1]) + 2, int(box[0]) + 2
    y1, x1 = int(box[1] + box[3]) - 2, int(box[0] + box[2]) - 2
    diff = (warped - plate)[:, y0:y1, x0:x1, :].abs().max()
    assert diff.item() < 1e-3
