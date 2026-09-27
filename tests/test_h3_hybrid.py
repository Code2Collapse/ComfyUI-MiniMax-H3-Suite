"""Tests for H3 hybrid math, checkpoint I/O, and extraction."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

from mmx_utils.h3_checkpoint_io import open_checkpoint
from mmx_utils.h3_hybrid import (
    fl_projection_coeff,
    lora_state_dict_entries,
    normalize_diffusion_key,
    pair_tensor_keys,
    residual_after_projection,
    svd_lora_from_residual,
    tensor_energy_norm,
)
from mmx_utils.h3_hybrid_extract import extract_hybrid_lora
from tests.h3_hybrid_fixtures import (
    write_fp32_triplet,
    write_int8_tensorwise,
    write_shape_mismatch_triplet,
)


def test_normalize_keys():
    assert normalize_diffusion_key("model.diffusion_model.blocks.0.attn.qkv_proj.weight").startswith(
        "diffusion_model.blocks.0"
    )
    assert normalize_diffusion_key("blocks.0.attn.qkv_proj.weight").startswith("diffusion_model.blocks.0")


def test_projection_recovers_a():
    torch.manual_seed(1)
    shape = (32, 16)
    R = torch.zeros(shape)
    D = torch.randn(shape)
    H_true = torch.randn(shape)
    H_true = H_true - (H_true * D).sum() / (D * D).sum() * D
    a_target = 0.6
    S = R + a_target * D + H_true
    delta = S - R
    a = fl_projection_coeff(delta, D)
    H = residual_after_projection(delta, D, a)
    assert abs(a - a_target) < 0.05
    assert tensor_energy_norm(H - H_true) < 1e-2 * tensor_energy_norm(H_true)


def test_no_fl_projection_is_delta():
    delta = torch.randn(4, 4)
    D = torch.randn(4, 4)
    H = residual_after_projection(delta, D, 0.0)
    assert torch.allclose(H, delta)


def test_svd_lora_rank_reconstruction_regression():
    torch.manual_seed(3)
    rank_true = 8
    a = torch.randn(192, rank_true)
    b = torch.randn(rank_true, 64)
    h = a @ b

    factors8 = svd_lora_from_residual(h, rank_true)
    recon8 = factors8.up.float() @ factors8.down.float()
    rel_err = float((recon8 - h).norm() / h.norm())
    assert rel_err < 1e-2

    factors4 = svd_lora_from_residual(h, 4)
    assert factors4.captured < 0.95


def test_svd_recovers_low_rank():
    torch.manual_seed(2)
    rank = 6
    u = torch.randn(40, rank)
    v = torch.randn(rank, 40)
    H_true = u @ v
    factors = svd_lora_from_residual(H_true, rank)
    assert factors.captured > 0.99


def test_extract_on_synthetic_triplet(tmp_path):
    ref, fl, sing = write_fp32_triplet(tmp_path)
    out = tmp_path / "hybrid.safetensors"
    summary = extract_hybrid_lora(str(ref), str(fl), str(sing), str(out), rank=8, device="cpu")
    assert out.is_file()
    assert summary["tensor_count"] == 8
    lora = load_file(str(out))
    assert any(k.endswith(".lora_up.weight") for k in lora)


def test_no_fl_projection_flag(tmp_path):
    ref, fl, sing = write_fp32_triplet(tmp_path)
    out = tmp_path / "plain.safetensors"
    extract_hybrid_lora(
        str(ref), str(fl), str(sing), str(out), rank=8, no_fl_projection=True, device="cpu"
    )
    assert out.is_file()


def test_shape_mismatch_raises(tmp_path):
    ref, fl, sing = write_shape_mismatch_triplet(tmp_path)
    with pytest.raises(ValueError, match="Shape mismatch"):
        extract_hybrid_lora(str(ref), str(fl), str(sing), str(tmp_path / "x.safetensors"), device="cpu")


def test_int8_tensorwise_roundtrip(tmp_path):
    pytest.importorskip("comfy.quant_ops")
    w = torch.randn(192, 64) * 0.1
    path = tmp_path / "int8.safetensors"
    write_int8_tensorwise(path, w)
    reader = open_checkpoint(path, device="cpu")
    try:
        key = "diffusion_model.blocks.0.attn.qkv_proj.weight"
        decoded = reader.read_tensor(key)
        err = (decoded - w.float()).abs().max().item()
        assert err < 0.2
    finally:
        reader.close()


def _convrot_available() -> bool:
    try:
        from comfy.quant_ops import TensorWiseINT8Layout

        w = torch.randn(64, 64)
        TensorWiseINT8Layout.quantize(w, scale="recalculate", convrot=True, convrot_groupsize=256)
        return True
    except Exception:
        return False


@pytest.mark.skipif(
    not _convrot_available(),
    reason="TensorWiseINT8Layout convrot quantize unavailable on this CPU build",
)
def test_convrot_roundtrip(tmp_path):
    from comfy.quant_ops import TensorWiseINT8Layout

    w = torch.randn(256, 64) * 0.05
    qdata, params = TensorWiseINT8Layout.quantize(
        w.float(), scale="recalculate", convrot=True, convrot_groupsize=256
    )
    quant_conf = {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 256}
    key = "diffusion_model.blocks.0.mlp.fc1"
    sd = {
        f"{key}.weight": qdata,
        f"{key}.weight_scale": params.scale,
        f"{key}.comfy_quant": torch.tensor(list(json.dumps(quant_conf).encode()), dtype=torch.uint8),
    }
    path = tmp_path / "convrot.safetensors"
    save_file(sd, str(path))
    reader = open_checkpoint(path, device="cpu")
    try:
        decoded = reader.read_tensor(f"{key}.weight")
        err = (decoded - w.float()).abs().max().item()
        assert err < 0.5
    finally:
        reader.close()


def test_gguf_read_roundtrip(tmp_path):
    pytest.importorskip("gguf")
    try:
        from gguf import GGUFWriter
    except Exception:
        pytest.skip("gguf-py writer unavailable")

    path = tmp_path / "tiny.gguf"
    writer = GGUFWriter(str(path), "test")
    writer.add_tensor("blocks.0.attn.qkv_proj.weight", torch.randn(8, 4).numpy())
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    reader = open_checkpoint(path, device="cpu")
    try:
        t = reader.read_tensor("diffusion_model.blocks.0.attn.qkv_proj.weight")
        assert t.shape == (8, 4)
    finally:
        reader.close()


def test_lora_keys_resolve_via_unet_map():
    try:
        import comfy.lora
    except Exception:
        pytest.skip("comfy.lora unavailable")

    fake_sdk = {
        "diffusion_model.blocks.0.attn.qkv_proj.weight": torch.zeros(8, 4),
        "diffusion_model.blocks.0.attn.out_proj.weight": torch.zeros(4, 4),
        "diffusion_model.blocks.1.mlp.fc2.weight": torch.zeros(4, 8),
    }

    class _FakeModelConfig:
        unet_config: dict = {}

    class FakeModel:
        model_config = _FakeModelConfig()

        def state_dict(self):
            return fake_sdk

    emitted: dict[str, torch.Tensor] = {}
    for wkey, wt in fake_sdk.items():
        emitted.update(
            lora_state_dict_entries(wkey, svd_lora_from_residual(torch.randn(*wt.shape), 2))
        )

    key_map = comfy.lora.model_lora_keys_unet(FakeModel(), {})
    loaded = comfy.lora.load_lora(emitted, key_map, log_missing=False)
    for wkey in fake_sdk:
        assert wkey in loaded
    assert len(loaded) == len(fake_sdk)


def test_pair_tensor_keys():
    keys = {"diffusion_model.blocks.0.attn.qkv_proj.weight"}
    paired = pair_tensor_keys(
        keys,
        keys,
        keys,
        shapes_ref=lambda k: (8, 4),
        shapes_fl=lambda k: (8, 4),
        shapes_sing=lambda k: (8, 4),
    )
    assert paired == ["diffusion_model.blocks.0.attn.qkv_proj.weight"]
