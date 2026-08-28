import pytest
import torch

torchvision = pytest.importorskip("torchvision")

from mmx_utils.optical_flow import compute_flow


def test_compute_flow_single_pair_shape():
    pair = torch.rand(2, 32, 32, 3)
    flow = compute_flow(pair, size="small", device="cpu")
    assert flow.shape == (1, 2, 32, 32)
