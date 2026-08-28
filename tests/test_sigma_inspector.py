import torch

from mmx_nodes.sigma_inspector import MiniMaxH3_SigmaInspector


def test_sigma_inspector_report():
    sigmas = torch.tensor([1.0, 0.5, 0.0])
    report, sigma_json = MiniMaxH3_SigmaInspector.execute(sigmas)
    assert "sigma_v" in report
    assert "sigma_a" in report
    import json

    data = json.loads(sigma_json)
    assert data["sigmas_v"] == [1.0, 0.5, 0.0]
    assert len(data["steps"]) == 3
