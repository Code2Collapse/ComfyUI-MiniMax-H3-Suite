import sys

import pytest
import torch

from mmx_utils.fake_patcher import FakeModelPatcher
from mmx_utils.sigma_schedule import validate_turbo_step_contract
from mmx_utils.turbo_lora import (
    TURBO_STEP_CONTRACT,
    TURBO_STEP_CONTRACT_KEY,
    apply_turbo_lora,
    read_turbo_step_contract,
    resolve_egrid_path,
    stash_turbo_step_contract,
)


def test_stash_turbo_step_contract_on_model():
    m = FakeModelPatcher()
    out = stash_turbo_step_contract(m)
    assert out is not m
    assert read_turbo_step_contract(out) == TURBO_STEP_CONTRACT
    assert (
        out.model_options["transformer_options"][TURBO_STEP_CONTRACT_KEY]
        == TURBO_STEP_CONTRACT
    )


def test_validate_turbo_step_contract_refuses_above_16():
    sigmas = torch.linspace(1.0, 0.0, 19)  # 18 steps
    ok, notes = validate_turbo_step_contract(sigmas, TURBO_STEP_CONTRACT)
    assert ok is False
    assert any("REFUSE" in n for n in notes)


def test_validate_turbo_step_contract_warns_above_8():
    sigmas = torch.linspace(1.0, 0.0, 11)  # 10 steps
    ok, notes = validate_turbo_step_contract(sigmas, TURBO_STEP_CONTRACT)
    assert ok is True
    assert any("WARNING" in n for n in notes)


def test_legal_scheduler_enforces_turbo_contract():
    from mmx_nodes.legal_scheduler import MiniMaxH3_LegalScheduler

    m = stash_turbo_step_contract(FakeModelPatcher())
    sigmas = torch.linspace(1.0, 0.0, 19)
    with pytest.raises(ValueError, match="REFUSE"):
        MiniMaxH3_LegalScheduler.execute(sigmas, m, "simple", 1.0)


def test_pruned_base_without_grid_raises(tmp_path, monkeypatch):
    from mmx_utils import turbo_lora as tl
    from mmx_utils.turbo_lora import TurboPrunedBaseUnavailableError

    monkeypatch.setattr(tl, "resolve_egrid_path", lambda: None)

    class _Utils:
        @staticmethod
        def load_torch_file(path, safe_load=True):
            return {"blocks.0.attn.qkv_proj.lora_A.weight": torch.zeros(1)}

    sys.modules["comfy"] = type("comfy", (), {"utils": _Utils})()

    m = FakeModelPatcher()
    setattr(m.model.diffusion_model, "use_adaln_curves", True)
    lora_file = tmp_path / "turbo.safetensors"
    lora_file.write_bytes(b"placeholder")

    with pytest.raises(TurboPrunedBaseUnavailableError, match="pruned H3 base"):
        apply_turbo_lora(m, str(lora_file), 1.0)


def test_apply_turbo_lora_missing_file_raises():
    m = FakeModelPatcher()
    with pytest.raises(FileNotFoundError):
        apply_turbo_lora(m, "/nonexistent/turbo.safetensors", 1.0)


@pytest.mark.skipif(
    resolve_egrid_path() is None,
    reason="h3_silu_temb_grid.safetensors not on this machine — pruned path UNVERIFIED",
)
def test_egrid_path_resolves_when_bundled():
    assert resolve_egrid_path().is_file()


@pytest.mark.skip(
    reason="UNVERIFIED: needs real H3 turbo LoRA weights and ComfyUI model patcher on this box",
)
def test_apply_turbo_lora_full_base_with_weights():
    pass
