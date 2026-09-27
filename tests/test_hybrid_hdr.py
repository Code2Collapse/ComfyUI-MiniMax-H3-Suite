"""Tests for MiniMaxH3_HybridHDR LoRA application."""

from __future__ import annotations

import pytest

from mmx_utils.hybrid_hdr import _filter_patches_by_block


def test_block_filter_keeps_only_requested_blocks():
    loaded = {
        "diffusion_model.blocks.0.attn.qkv_proj.weight": object(),
        "diffusion_model.blocks.1.attn.qkv_proj.weight": object(),
        "diffusion_model.adaln_proj.linear.weight": object(),
    }
    out = _filter_patches_by_block(loaded, 1, 1)
    assert set(out.keys()) == {
        "diffusion_model.blocks.1.attn.qkv_proj.weight",
        "diffusion_model.adaln_proj.linear.weight",
    }


def test_apply_hybrid_hdr_missing_file():
    from mmx_utils.fake_patcher import FakeModelPatcher
    from mmx_utils.hybrid_hdr import apply_hybrid_hdr_lora

    with pytest.raises(FileNotFoundError):
        apply_hybrid_hdr_lora(FakeModelPatcher(), "/nonexistent/hybrid.safetensors", 1.0, 0, 49)
