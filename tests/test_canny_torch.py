import torch

from mmx_utils.canny_torch import canny_torch


def test_canny_torch_cpu_output_shape_and_range():
    img = torch.rand(2, 64, 64, 3)
    edges = canny_torch(img, low_threshold=0.1, high_threshold=0.3, blur_sigma=1.0)
    assert edges.shape == img.shape
    assert edges.min() >= 0.0 and edges.max() <= 1.0
    assert torch.isfinite(edges).all()


def test_canny_on_uniform_has_exactly_zero_edges():
    """A flat field has no edges anywhere — including the border.

    Asserting EXACTLY zero, not 'mostly zero'. A tolerance here would have
    hidden the original bug: the Sobel used conv2d(padding=1), which pads with
    ZEROS, so a uniform 0.5 image read as a full-frame step and lit up the whole
    1px border ring — 124 of 1024 pixels at 32x32, i.e. mean 0.121. That is a
    perfect rectangle, and N3 runs this on a CROP, so ControlNet would lock
    structure onto a box that does not exist in the picture.
    """
    img = torch.ones(1, 32, 32, 3) * 0.5
    edges = canny_torch(img, low_threshold=0.2, high_threshold=0.4)
    assert edges.sum().item() == 0.0, (
        f"{int(edges.sum().item() / 3)} spurious edge pixels on a flat field; "
        "check the Sobel padding mode"
    )


def test_canny_still_finds_a_real_edge():
    """Guards the fix above: proves we killed the border ring, not the detector."""
    img = torch.zeros(1, 32, 32, 3)
    img[:, :, 16:, :] = 1.0                      # vertical step down the middle
    e = canny_torch(img, low_threshold=0.2, high_threshold=0.4)[0, :, :, 0]
    assert e.sum().item() > 0, "no edge found on an obvious step"
    # The response must sit ON the boundary, not smear across the frame — but do
    # not hardcode a narrow window. The default blur_sigma=1.4 legitimately
    # widens the response: measured cols 13-16 at sigma 1.4, 15-16 at sigma 0.
    # So assert every responding column is within the blur's reach of the step,
    # which tests localisation without pinning the kernel width.
    cols = (e.sum(dim=0) > 0).nonzero().flatten().tolist()
    assert cols, "no responding columns"
    assert all(12 <= c <= 19 for c in cols), f"edge response away from the step: {cols}"
    # No response on the left/right frame edges — the border-ring bug.
    # (Top and bottom rows are NOT checked: this step spans the full height, so
    # the real edge legitimately passes through row 0 and row -1.)
    assert e[:, 0].sum().item() == 0 and e[:, -1].sum().item() == 0
