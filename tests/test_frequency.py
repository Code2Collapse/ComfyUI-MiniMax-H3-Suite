import pytest
import torch

from mmx_utils.frequency import frequency_combine, frequency_separate, to_nchw


def _rand_image():
    torch.manual_seed(1)
    return torch.rand(1, 48, 64, 3)


@pytest.mark.parametrize("mode", ["divide", "subtract"])
def test_separate_combine_roundtrip(mode):
    img = _rand_image()
    lf, detail = frequency_separate(img, mode=mode, radius=4)
    recon = frequency_combine(lf, detail, mode=mode)
    err = (recon - to_nchw(img)).abs().max().item()
    assert err < 1e-5


def test_divide_mode_zero_plate_finite():
    z = torch.zeros(1, 32, 32, 3)
    lf, detail = frequency_separate(z, mode="divide", radius=4, eps=1e-3)
    assert torch.isfinite(detail).all()
    assert torch.isfinite(lf).all()
