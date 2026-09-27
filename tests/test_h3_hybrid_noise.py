"""Noise floor, MP edge selection, parent classification, and report tests."""

from __future__ import annotations

from unittest.mock import patch

import pytest
import torch
from safetensors.torch import load_file

from mmx_utils.h3_checkpoint_io import open_checkpoint
from mmx_utils.h3_hybrid import (
    classify_parent,
    combined_h_noise_floor,
    fl_projection_coeff,
    noise_edge,
    residual_after_projection,
    svd_lora_from_residual,
)
from mmx_utils.h3_hybrid_extract import extract_hybrid_lora
from tests.h3_hybrid_fixtures import (
    write_fp32_triplet,
    write_int8_per_row,
    write_single_key_triplet,
)

_KEY = "diffusion_model.blocks.0.attn.qkv_proj.weight"


def _read_projected_h(ref_path, fl_path, sing_path, key: str) -> torch.Tensor:
    ref_r = open_checkpoint(ref_path, device="cpu")
    fl_r = open_checkpoint(fl_path, device="cpu")
    sing_r = open_checkpoint(sing_path, device="cpu")
    try:
        R = ref_r.read_tensor(key)
        F = fl_r.read_tensor(key)
        S = sing_r.read_tensor(key)
        delta = S - R
        D = F - R
        a = fl_projection_coeff(delta, D)
        return residual_after_projection(delta, D, a)
    finally:
        ref_r.close()
        fl_r.close()
        sing_r.close()


def _write_int8_triplet(
    tmp_path,
    key: str,
    ref_w: torch.Tensor,
    fl_w: torch.Tensor,
    sing_w: torch.Tensor,
):
    tmp_path.mkdir(parents=True, exist_ok=True)   # callers pass sub-folders
    ref_p = tmp_path / "ref_int8.safetensors"
    fl_p = tmp_path / "fl_int8.safetensors"
    sing_p = tmp_path / "sing_int8.safetensors"
    write_int8_per_row(ref_p, key, ref_w)
    write_int8_per_row(fl_p, key, fl_w)
    write_int8_per_row(sing_p, key, sing_w)
    return ref_p, fl_p, sing_p


def _combined_floor(ref_path, fl_path, sing_path, key: str, a: float = 1.0) -> float:
    ref_r = open_checkpoint(ref_path, device="cpu")
    fl_r = open_checkpoint(fl_path, device="cpu")
    sing_r = open_checkpoint(sing_path, device="cpu")
    try:
        return combined_h_noise_floor(
            ref_r.noise_energy(key),
            fl_r.noise_energy(key),
            sing_r.noise_energy(key),
            a,
        )
    finally:
        ref_r.close()
        fl_r.close()
        sing_r.close()


def test_noise_energy_int8_matches_measured_dequant_error(tmp_path):
    torch.manual_seed(11)
    weight = torch.randn(256, 256)
    path = tmp_path / "int8_row.safetensors"
    write_int8_per_row(path, _KEY, weight)
    reader = open_checkpoint(path, device="cpu")
    try:
        predicted = reader.noise_energy(_KEY)
        assert predicted is not None
        decoded = reader.read_tensor(_KEY)
        measured = float(((decoded - weight.float()) ** 2).sum().item())
        assert measured > 0.0
        assert abs(predicted - measured) / measured < 0.20
    finally:
        reader.close()


def test_mp_edge_keeps_low_rank_signal(tmp_path):
    shape = (128, 64)
    torch.manual_seed(21)
    R = torch.randn(shape) * 0.05
    D = torch.randn(shape) * 0.05
    F = R + D

    ref_p, fl_p, sing_p = _write_int8_triplet(tmp_path, _KEY, R, F, F)
    floor = _combined_floor(ref_p, fl_p, sing_p, _KEY, a=1.0)
    edge = noise_edge(floor, shape[0], shape[1], 1.1)

    u, _, v = torch.svd_lowrank(torch.randn(shape), q=4, niter=4)
    s_signal = torch.tensor([edge * 5.0, edge * 4.8, edge * 4.5, edge * 4.2])
    H_true = (u[:, :4] * s_signal.unsqueeze(0)) @ v[:, :4].T
    S = F + H_true

    ref, fl, sing = _write_int8_triplet(tmp_path / "sig", _KEY, R, F, S)
    out = tmp_path / "signal.safetensors"
    summary = extract_hybrid_lora(str(ref), str(fl), str(sing), str(out), rank=8, device="cpu")
    row = summary["per_tensor"][0]
    assert row["emitted"] is True
    assert row["k"] is not None
    assert 3 <= row["k"] <= 5

    lora = load_file(str(out))
    base = _KEY[: -len(".weight")]
    approx = lora[f"{base}.lora_up.weight"].float() @ lora[f"{base}.lora_down.weight"].float()
    # Judge against the SIGNAL: the decoded residual also holds int8 noise,
    # which is exactly what the edge is meant to throw away.
    H = _read_projected_h(ref, fl, sing, _KEY)
    assert float(H.norm().item()) > 0.0
    rel_err = float((approx - H_true).norm().item() / H_true.norm().item())
    # Noise also tilts the kept singular vectors; at 5x the edge on a small
    # 128x64 matrix that costs ~14 % (measured). The raw residual is ~39 % off.
    assert rel_err < 0.25
    # and the discarded part is noise-sized: the kept rank-k fit is closer to
    # the true signal than the raw decoded residual is
    assert rel_err < float((H - H_true).norm().item() / H_true.norm().item())


def test_skip_noise_only_when_s_equals_f(tmp_path):
    shape = (128, 64)
    torch.manual_seed(31)
    R = torch.randn(shape) * 0.05
    F = R + torch.randn(shape) * 0.05
    ref, fl, sing = _write_int8_triplet(tmp_path, _KEY, R, F, F)
    out = tmp_path / "skip.safetensors"
    summary = extract_hybrid_lora(str(ref), str(fl), str(sing), str(out), rank=8, device="cpu")
    row = summary["per_tensor"][0]
    assert summary["tensor_count"] == 0
    assert row["k"] == 0
    assert row["noise_only"] is True
    assert _KEY in summary["noise_only"]
    assert not load_file(str(out))


def test_rank_cap_sets_rank_capped():
    torch.manual_seed(41)
    m, n = 96, 48
    u = torch.randn(m, 12)
    v = torch.randn(12, n)
    H = u @ v
    floor = float((H * H).sum().item()) * 1e-6
    edge = noise_edge(floor, m, n, 1.1)
    factors = svd_lora_from_residual(H, rank=8, edge=edge)
    assert factors is not None
    assert factors.k == 8
    assert factors.rank_capped is True


def test_unknown_floor_emits_fixed_rank(tmp_path):
    shape = (64, 32)
    R = torch.randn(shape)
    F = R + torch.randn(shape) * 0.5
    S = F + torch.randn(shape) * 0.25
    ref, fl, sing = write_single_key_triplet(tmp_path, _KEY, R, F, S)
    out = tmp_path / "fixed_rank.safetensors"

    def _no_floor(_self, _key):
        return None

    with patch("mmx_utils.h3_checkpoint_io.SafetensorsCheckpointReader.noise_energy", _no_floor):
        summary = extract_hybrid_lora(
            str(ref), str(fl), str(sing), str(out), rank=8, device="cpu"
        )

    row = summary["per_tensor"][0]
    assert row["emitted"] is True
    assert row["k"] == 8
    assert row["noise_edge"] is None
    assert load_file(str(out))


def test_classify_parent_ref2va_and_fl2va():
    noise = 1e-6
    assert classify_parent(0.0, 1.0, 1.0, 1.0, noise, noise, noise) == "Ref2VA"
    assert classify_parent(1.0, 0.0, 1.0, 1.0, noise, noise, noise) == "FL2VA"


def test_combined_h_noise_floor_weights():
    floor = combined_h_noise_floor(2.0, 3.0, 4.0, 0.5)
    assert floor == pytest.approx(4.0 + 0.25 * 2.0 + 0.25 * 3.0)


def test_report_markdown_contains_block_table(tmp_path):
    ref, fl, sing = write_fp32_triplet(tmp_path)
    out = tmp_path / "hybrid.safetensors"
    summary = extract_hybrid_lora(
        str(ref),
        str(fl),
        str(sing),
        str(out),
        rank=8,
        device="cpu",
    )
    md = summary["markdown"]
    assert "## Decomposition summary" in md
    assert "| block | adaln parent | mean a (trunk) |" in md
    assert "mean k | capped %" in md
    assert "emitted / skipped" in md
    assert "Recommended rank:" in md or "raise --rank:" in md
