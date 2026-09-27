"""Synthetic H3-layout checkpoints for hybrid extract tests."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import save_file

HIDDEN = 64
BLOCKS = (0, 1)


def _block_keys(block: int) -> list[str]:
    p = f"diffusion_model.blocks.{block}"
    return [
        f"{p}.attn.qkv_proj.weight",
        f"{p}.attn.out_proj.weight",
        f"{p}.mlp.fc1.weight",
        f"{p}.mlp.fc2.weight",
    ]


def _shapes() -> dict[str, tuple[int, int]]:
    shapes = {}
    for block in BLOCKS:
        shapes[f"diffusion_model.blocks.{block}.attn.qkv_proj.weight"] = (192, HIDDEN)
        shapes[f"diffusion_model.blocks.{block}.attn.out_proj.weight"] = (HIDDEN, HIDDEN)
        shapes[f"diffusion_model.blocks.{block}.mlp.fc1.weight"] = (256, HIDDEN)
        shapes[f"diffusion_model.blocks.{block}.mlp.fc2.weight"] = (HIDDEN, 256)
    return shapes


def write_fp32_triplet(
    tmp_path: Path,
    *,
    ref_scale: float = 0.0,
    fl_scale: float = 1.0,
    sing_a: float = 0.6,
    sing_h_rank: int = 4,
) -> tuple[Path, Path, Path]:
    """Write ref/fl/sing with controlled projection geometry on block 0 qkv."""
    shapes = _shapes()
    ref_sd: dict[str, torch.Tensor] = {}
    fl_sd: dict[str, torch.Tensor] = {}
    sing_sd: dict[str, torch.Tensor] = {}

    torch.manual_seed(0)
    for key, shape in shapes.items():
        R = torch.randn(shape, dtype=torch.float32) * 0.01 + ref_scale
        D = torch.randn(shape, dtype=torch.float32) * 0.05
        D = D - (D * R).sum() / (R * R).sum().clamp(min=1e-12) * R
        F = R + fl_scale * D
        if key.endswith("blocks.0.attn.qkv_proj.weight"):
            # svd_lowrank returns (U, S, V) - V, not V^T
            u, s, v = torch.svd_lowrank(torch.randn(shape), q=sing_h_rank, niter=2)
            H_true = (u[:, :sing_h_rank] * s[:sing_h_rank]) @ v[:, :sing_h_rank].T
            H_true = H_true - (H_true * D).sum() / (D * D).sum() * D
            S = R + sing_a * D + H_true
        else:
            S = R + 0.1 * torch.randn(shape, dtype=torch.float32)
        ref_sd[key] = R.to(torch.bfloat16)
        fl_sd[key] = F.to(torch.bfloat16)
        sing_sd[key] = S.to(torch.bfloat16)

    ref_p = tmp_path / "ref.safetensors"
    fl_p = tmp_path / "fl.safetensors"
    sing_p = tmp_path / "sing.safetensors"
    save_file(ref_sd, str(ref_p))
    save_file(fl_sd, str(fl_p))
    save_file(sing_sd, str(sing_p))
    return ref_p, fl_p, sing_p


def write_single_key_triplet(
    tmp_path: Path,
    key: str,
    ref_w: torch.Tensor,
    fl_w: torch.Tensor,
    sing_w: torch.Tensor,
    *,
    dtype=torch.bfloat16,
) -> tuple[Path, Path, Path]:
    ref_p = tmp_path / "ref_single.safetensors"
    fl_p = tmp_path / "fl_single.safetensors"
    sing_p = tmp_path / "sing_single.safetensors"
    save_file({key: ref_w.to(dtype)}, str(ref_p))
    save_file({key: fl_w.to(dtype)}, str(fl_p))
    save_file({key: sing_w.to(dtype)}, str(sing_p))
    return ref_p, fl_p, sing_p


def write_int8_per_row(path: Path, key: str, weight: torch.Tensor) -> None:
    out_f, in_f = weight.shape
    q = torch.zeros(out_f, in_f, dtype=torch.int8)
    scales = torch.zeros(out_f, dtype=torch.float32)
    for i in range(out_f):
        row = weight[i].float()
        scale = max(float(row.abs().max()) / 127.0, 1e-8)
        scales[i] = scale
        q[i] = (row / scale).round().clamp(-128, 127).to(torch.int8)
    base = key[: -len(".weight")]
    quant_conf = {"format": "int8_tensorwise"}
    sd = {
        key: q,
        # per-row scales are stored [out, 1] in the real files (read from the
        # Singularity / Comfy-Org headers); a flat [out] vector would broadcast
        # across columns instead of rows
        f"{base}.weight_scale": scales.view(out_f, 1),
        f"{base}.comfy_quant": torch.tensor(list(json.dumps(quant_conf).encode("utf-8")), dtype=torch.uint8),
    }
    save_file(sd, str(path))


def write_int8_tensorwise(path: Path, weight: torch.Tensor) -> None:
    try:
        from comfy.quant_ops import TensorWiseINT8Layout
    except Exception as exc:
        raise RuntimeError(str(exc)) from exc

    qdata, params = TensorWiseINT8Layout.quantize(weight.float())
    key = "diffusion_model.blocks.0.attn.qkv_proj"
    quant_conf = {"format": "int8_tensorwise"}
    sd = {
        f"{key}.weight": qdata,
        f"{key}.weight_scale": params.scale,
        f"{key}.comfy_quant": torch.tensor(list(json.dumps(quant_conf).encode("utf-8")), dtype=torch.uint8),
    }
    save_file(sd, str(path))


def write_shape_mismatch_triplet(tmp_path: Path) -> tuple[Path, Path, Path]:
    shapes = _shapes()
    ref_sd = {k: torch.zeros(v, dtype=torch.bfloat16) for k, v in shapes.items()}
    fl_sd = dict(ref_sd)
    sing_sd = dict(ref_sd)
    bad = "diffusion_model.blocks.0.attn.out_proj.weight"
    sing_sd[bad] = torch.zeros(32, HIDDEN, dtype=torch.bfloat16)
    ref_p = tmp_path / "ref2.safetensors"
    fl_p = tmp_path / "fl2.safetensors"
    sing_p = tmp_path / "sing2.safetensors"
    save_file(ref_sd, str(ref_p))
    save_file(fl_sd, str(fl_p))
    save_file(sing_sd, str(sing_p))
    return ref_p, fl_p, sing_p
