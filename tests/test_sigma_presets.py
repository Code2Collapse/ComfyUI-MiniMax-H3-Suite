"""Sigma Presets (owner A2, L8.19): every preset computes with core's own scheduler code, cites templates that exist in
the installed official templates package, and matches the formulas it claims."""
from __future__ import annotations

import importlib.metadata as md
import re
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("comfy.samplers")

from mmx_utils import sigma_presets as SP  # noqa: E402


@pytest.mark.parametrize("label", SP.LABELS)
def test_every_preset_computes(label):
    sig, audio, origin = SP.preset_sigmas(label)
    p = SP.PRESETS[label]
    assert sig.ndim == 1 and len(sig) >= 2 and origin
    assert bool((sig[1:] <= sig[:-1] + 1e-6).all()), "sigmas must not increase"
    assert float(sig[-1]) <= 1e-4
    if p.schedule[0] != "manual":
        assert len(sig) == p.steps + 1
    if p.sampling[0] == "av":
        assert audio is not None and audio.shape == sig.shape and float(audio[0]) <= float(sig[0]) + 1e-6
    text = SP.how_to(label, sig, audio, origin, 0)
    assert p.sampler in text and "source:" in text and (not p.lora or "needs:" in text)


def _template_stems():
    try:
        dist = md.distribution("comfyui-workflow-templates-json")
    except md.PackageNotFoundError:
        return None
    root = Path(dist.locate_file("comfyui_workflow_templates_json"))
    return {f.stem for f in root.rglob("*.json")}


def test_cited_templates_exist():
    stems = _template_stems()
    if stems is None:
        pytest.skip("official templates package not installed")
    missing = {}
    for label, p in SP.PRESETS.items():
        cited = [t.rstrip(".,") for t in re.findall(r"\b[A-Za-z0-9][A-Za-z0-9.]*(?:_[A-Za-z0-9.]+)+", p.source)
                 if not t.startswith("supported_models") and t not in ("max_shift", "base_shift")]
        found = [t for t in cited if t in stems]
        bad = [t for t in cited if t not in stems]
        if bad or (not found and "defaultGraph" not in p.source):
            missing[label] = bad or ["(no template cited)"]
    assert not missing, missing


def test_flow_shift_matches_the_formula():
    sig, _a, _o = SP.preset_sigmas("Wan 2.2 14B T2V / I2V (template, lightx2v 4-step LoRAs)")
    t = torch.tensor([1.0, 0.75, 0.5, 0.25])
    expect = 5.0 * t / (1 + 4.0 * t)                      # time_snr_shift(5, t)
    assert torch.allclose(sig[:4], expect, atol=2e-3) and float(sig[-1]) == 0.0


def test_h3_audio_clock_is_cores():
    from comfy.ldm.minimax.model import time_shift_sigma
    sig, audio, _o = SP.preset_sigmas("MiniMax H3 Ref2VA (template, ref2v turbo LoRA)")
    assert torch.allclose(audio[:-1], time_shift_sigma(sig[:-1], 12.0, 3.0).float(), atol=1e-6)


def test_sd_eps_schedules_start_at_the_trained_sigma_max():
    sig, _a, _o = SP.preset_sigmas("SD 1.5 (ComfyUI default graph)")
    assert 14.5 < float(sig[0]) < 14.7                     # SD's 0.00085-0.012 beta schedule
    k, _a, _o = SP.preset_sigmas("SDXL (template)")
    assert 14.5 < float(k[0]) < 14.7 and len(k) == 26


def test_dedicated_schedulers_and_manual():
    ltx, _a, origin = SP.preset_sigmas("LTX-2 dev stage 1 (template, distilled LoRA)")
    assert "LTXVScheduler" in origin and len(ltx) == 13
    a, _x, _o = SP.preset_sigmas("Flux.2 dev (template)", width=1024, height=1024)
    b, _x, _o = SP.preset_sigmas("Flux.2 dev (template)", width=2048, height=2048)
    assert not torch.allclose(a, b)                        # Flux.2's schedule depends on the image size
    m, _x, origin = SP.preset_sigmas("LTX-2.3 / 2.5 stage 2 (template)", steps=10)
    assert m.tolist() == pytest.approx([0.85, 0.725, 0.4219, 0.0]) and "not changed" in origin
    t, _x, _o = SP.preset_sigmas("SDXL Turbo (template)")
    assert len(t) == 2


def test_denoise_takes_the_tail_of_a_longer_schedule():
    full, _a, _o = SP.preset_sigmas("Z-Image (template)", steps=50)
    half, _a, _o = SP.preset_sigmas("Z-Image (template)", steps=25, denoise=0.5)
    assert len(half) == 26 and torch.allclose(half, full[-26:])


def test_a_connected_model_supplies_its_own_shift():
    ms = SP.model_sampling_for(("flow", 2.0, 1.0))
    fake = SimpleNamespace(get_model_object=lambda name: ms)
    own, _a, origin = SP.preset_sigmas("Z-Image (template)", model=fake)
    base, _a, _o = SP.preset_sigmas("Z-Image (template)")
    assert "your model" in origin and not torch.allclose(own, base)


def test_node_outputs_and_graph():
    from mmx_nodes.sigma_presets import MiniMaxH3_SigmaPresets
    out = MiniMaxH3_SigmaPresets.execute("MiniMax H3 FL2VA / T2V (template, fl2v turbo LoRA)")
    res = out.result if hasattr(out, "result") else out.args
    sig, sampler, cfg, steps, graph, text = res
    assert steps == 4 and len(sig) == 5 and cfg == 1.0 and hasattr(sampler, "sample")
    assert graph.shape == (1, 400, 720, 3) and float(graph.std()) > 0.01
    assert "fl2v_turbo" in text
    with pytest.raises(ValueError):
        MiniMaxH3_SigmaPresets.execute("not a preset")
