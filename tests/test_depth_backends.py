import pytest

from mmx_utils.depth_backends import (
    DepthBackendUnavailableError,
    normals_from_depth,
    passthrough_depth_plate,
    run_depth_backend,
    run_depthcrafter,
    run_dvd,
)
import torch


def test_passthrough_luminance_proxy_deterministic():
    img = torch.rand(2, 16, 16, 3)
    a = passthrough_depth_plate(img, None)
    b = passthrough_depth_plate(img, None)
    assert torch.equal(a, b)


def test_passthrough_warp_hints():
    img = torch.rand(1, 8, 8, 3)
    hints = torch.ones(1, 8, 8, 3) * 0.5
    out = passthrough_depth_plate(img, hints)
    assert torch.allclose(out, hints)


def test_normals_from_depth_deterministic():
    depth = torch.rand(2, 12, 12, 3)
    n1 = normals_from_depth(depth)
    n2 = normals_from_depth(depth)
    assert torch.equal(n1, n2)
    assert n1.shape == (2, 12, 12, 3)


def test_depthcrafter_raises_named_error():
    with pytest.raises(DepthBackendUnavailableError, match="DepthCrafter"):
        run_depthcrafter(torch.rand(1, 8, 8, 3))


def test_dvd_raises_third_party_not_cloned():
    with pytest.raises(DepthBackendUnavailableError, match="third_party/DVD"):
        run_dvd(torch.rand(1, 8, 8, 3))


def test_run_depth_backend_passthrough_default():
    out, note = run_depth_backend("passthrough", torch.rand(1, 4, 4, 3))
    assert out.shape[0] == 1
    assert "passthrough" in note
