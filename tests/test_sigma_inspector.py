import json

import torch

from mmx_nodes.sigma_inspector import MiniMaxH3_SigmaInspector


def test_sigma_inspector_report():
    sigmas = torch.tensor([1.0, 0.5, 0.0])
    report, sigma_json = MiniMaxH3_SigmaInspector.execute(sigmas=sigmas)
    assert "sigma_v" in report
    assert "analytic" in report
    data = json.loads(sigma_json)
    assert abs(data["steps"][0]["dsigma_a_dsigma_v"] - 4.0) <= 1e-6


def test_sigma_inspector_synthesises_from_steps_when_sigmas_unconnected():
    report, sigma_json = MiniMaxH3_SigmaInspector.execute(sigmas=None, steps=20)
    data = json.loads(sigma_json)
    assert len(data["sigmas_v"]) == 21
    assert data["sigmas_v"][0] == 1.0
    assert data["sigmas_v"][-1] == 0.0
    assert "sigma_v" in report


def test_sigma_inspector_connected_sigmas_path_unchanged():
    sigmas = torch.linspace(1.0, 0.0, 10)
    report, sigma_json = MiniMaxH3_SigmaInspector.execute(sigmas=sigmas, steps=99)
    data = json.loads(sigma_json)
    assert len(data["sigmas_v"]) == 10
    assert data["sigmas_v"][0] == 1.0
