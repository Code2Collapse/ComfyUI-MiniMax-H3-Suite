import pytest
import torch

from mmx_nodes.legal_scheduler import MiniMaxH3_LegalScheduler


def test_legal_scheduler_passthrough():
    sigmas = torch.linspace(1.0, 0.0, 5)
    out = MiniMaxH3_LegalScheduler.execute(sigmas, "simple", 1.0)
    assert torch.allclose(out[0], sigmas)


def test_legal_scheduler_rejects_split():
    sigmas = torch.linspace(1.0, 0.0, 5)
    with pytest.raises(ValueError, match="SplitSigmas"):
        MiniMaxH3_LegalScheduler.execute(sigmas, "split_sigmas", 1.0)
