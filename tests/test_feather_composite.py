import torch

from mmx_utils.affine_transform import affine_crop, inverse_affine_warp_batch
from mmx_utils.feather_composite import mask_confined_blend


def _fixture():
    torch.manual_seed(0)
    h, w = 128, 192
    plate = torch.rand(1, h, w, 3)
    box = (40.0, 30.0, 80.0, 70.0)
    crop = affine_crop(plate, box, 96, 96)
    return plate, crop, box, h, w


def test_stitch_exterior_torch_equal():
    plate, crop, box, h, w = _fixture()
    warped = inverse_affine_warp_batch(crop, [box], w, h)
    m = torch.zeros(1, h, w, 1)
    y0, x0 = int(box[1]) + 5, int(box[0]) + 5
    y1, x1 = int(box[1] + box[3]) - 5, int(box[0] + box[2]) - 5
    m[:, y0:y1, x0:x1, :] = 1.0
    out = mask_confined_blend(plate, warped, m, colour_match=1.0)
    exterior = m[..., 0] <= 0
    assert torch.equal(out[exterior], plate[exterior])
