"""End to end on core's real H3 model and sampler, with tiny random weights (tests/h3_tiny.py, ledger L8.35): the
suite's model patches and latent builders as ComfyUI actually runs them. Plumbing only - never image quality."""
from __future__ import annotations

import pytest
import torch

pytest.importorskip("comfy.ldm.minimax.model")
pytest.importorskip("comfy_extras.nodes_minimax_h3")

from tests import h3_tiny as H  # noqa: E402

nt = pytest.importorskip("comfy.nested_tensor")


def latent_with_mask(width=128, height=96, length=22, keep_left=True, seed=3):
    lat = H.empty_latent(width, height, length)
    v, a = lat["samples"].unbind()
    gen = torch.Generator().manual_seed(seed)
    v = torch.randn(v.shape, generator=gen)
    mv = torch.ones(1, 1, *v.shape[2:])
    if keep_left:
        mv[..., : v.shape[-1] // 2] = 0.0                  # left half preserved (on the 2x2 patch grid)
    return {"samples": nt.NestedTensor((v, a)), "noise_mask": nt.NestedTensor((mv, torch.ones(1, 1, *a.shape[2:])))}


def test_tiny_h3_samples_deterministically():
    p = H.tiny_model_patcher()
    lat = H.empty_latent()
    a = H.sample(p, lat, seed=1)
    b = H.sample(p, lat, seed=1)
    c = H.sample(p, lat, seed=2)
    va, aa = a.unbind()
    assert va.shape == (1, 24, 7, 4, 4) and aa.shape[1] == 32 and torch.isfinite(va).all()
    assert torch.equal(va, b.unbind()[0]) and not torch.equal(va, c.unbind()[0])


def test_preserved_rows_come_out_exactly_as_given():
    p = H.tiny_model_patcher()
    lat = latent_with_mask()
    out = H.sample(p, lat).unbind()[0]
    src = lat["samples"].unbind()[0]
    half = src.shape[-1] // 2
    # equal up to fp32 rounding: core's latent format normalises the latent going in and back out (max 3.6e-7 here)
    assert torch.allclose(out[..., :half], src[..., :half], rtol=0, atol=1e-6)
    assert not torch.allclose(out[..., half:], src[..., half:])


def _with_tiling(p, tile_px, overlap_px):
    from mmx_nodes.tiled_diffusion import MiniMaxH3_TiledDiffusion
    out = MiniMaxH3_TiledDiffusion.execute(p, "trapezoid", tile_px, tile_px, overlap_px)
    return (out.result if hasattr(out, "result") else out.args)[0]


def test_tiled_diffusion_single_tile_is_bit_identical():
    p = H.tiny_model_patcher()
    lat = H.empty_latent(128, 96)
    plain = H.sample(p, lat).unbind()[0]
    tiled = H.sample(_with_tiling(p, 512, 64), lat).unbind()[0]
    assert torch.equal(plain, tiled)


def test_tiled_diffusion_tiles_run_through_core_and_keep_masked_rows():
    p = H.tiny_model_patcher()
    patched = _with_tiling(p, 64, 32)                      # 4-cell tiles, 2-cell overlap on an 8 x 6 cell frame
    wrapper = patched.model_options["model_function_wrapper"]
    calls = []
    inner = wrapper.inner

    def counting(model_function, args):
        calls.append(tuple(args["input"].shape))
        return model_function(args["input"], args["timestep"], **args["c"])
    wrapper.inner = counting
    lat = latent_with_mask()
    out = H.sample(patched, lat).unbind()[0]
    wrapper.inner = inner
    src = lat["samples"].unbind()[0]
    half = src.shape[-1] // 2
    assert len(calls) == 2 * len(wrapper._plans[next(iter(wrapper._plans))].tiles) and len(calls) > 2
    assert torch.isfinite(out).all() and torch.allclose(out[..., :half], src[..., :half], rtol=0, atol=1e-6)


def test_tst_counts_every_video_attention_call():
    from mmx_utils.tst_attention import patch_model
    p = H.tiny_model_patcher()
    lat = H.empty_latent(64, 64)
    logs = []
    patched, state = patch_model(p, 0.3, log=logs.append)
    with_tst = H.sample(patched, lat).unbind()[0]
    plain = H.sample(p, lat).unbind()[0]
    assert state["layers"] == H.TINY["num_layers"]          # counted from the real calls, not the 50-layer guess
    assert state["stats"]["calls"] == H.TINY["num_layers"] * 2   # two steps, cfg 1 -> one model call per step
    assert logs and logs[-1].startswith("TST tau=0.30") and not torch.equal(with_tst, plain)


def test_region_edit_keeps_the_plate_outside_the_mask():
    from mmx_nodes.region_edit import MiniMaxH3_RegionEdit
    from tests.test_source_av_latent import FakeH3VAE
    p = H.tiny_model_patcher()
    plate = torch.rand(22, 96, 128, 3)
    mask = torch.zeros(22, 96, 128)
    mask[:, :, 64:] = 1.0                                  # regenerate the right half
    out = MiniMaxH3_RegionEdit.execute(p, FakeH3VAE(), plate, mask)
    model, latent, _report = out.result if hasattr(out, "result") else out.args
    res = H.sample(model, latent).unbind()[0]
    src = latent["samples"].unbind()[0]
    assert torch.allclose(res[..., :4], src[..., :4], rtol=0, atol=1e-6)   # the plate's 4 latent columns on the left
    assert not torch.allclose(res[..., 4:], src[..., 4:])


def test_split_upscale_runs_on_an_odd_latent():
    from comfy_extras.nodes_custom_sampler import Noise_RandomNoise
    import comfy.samplers
    import mmx_nodes.split_upscale as SU
    p = H.tiny_model_patcher()
    lat = H.empty_latent(112, 80)                          # 7 x 5 latent cells: odd both ways
    sp = SU.MiniMaxH3_SpatialSplitParams.execute(64, 64, 0.5, 0.5, 0, 1.0)
    sp = (sp.result if hasattr(sp, "result") else sp.args)[0]
    ms = p.get_model_object("model_sampling")
    sigmas = comfy.samplers.calculate_sigmas(ms, "simple", 2).cpu()
    out = SU.MiniMaxH3_SplitUpscale.execute(lat, H.text(), p, Noise_RandomNoise(1),
                                            comfy.samplers.sampler_object("euler"), sigmas,
                                            spatial_split_param=sp, color_match=False)
    res = (out.result if hasattr(out, "result") else out.args)[0]["samples"].unbind()[0]
    assert res.shape == lat["samples"].unbind()[0].shape and torch.isfinite(res).all()


def test_a_synthetic_checkpoint_loads_through_cores_loader(tmp_path):
    """Synthetic checkpoint (L8.07): saved as safetensors, loaded like UNETLoader, detected as H3 from the tensors,
    and it samples exactly like the model it was saved from."""
    p = H.tiny_model_patcher(seed=5)
    path = H.write_checkpoint(p, tmp_path / "tiny_h3.safetensors")
    loaded = H.load_checkpoint(path)
    cfg = loaded.model.model_config.unet_config
    for k in ("hidden_size", "num_layers", "num_attention_heads", "attention_head_dim", "ffn_hidden_size", "text_dim",
              "rope_inv_freq_len", "time_embed_dim"):
        assert cfg[k] == H.TINY[k], k
    assert cfg["image_model"] == "minimax_h3"
    lat = H.empty_latent()
    assert torch.equal(H.sample(p, lat).unbind()[0], H.sample(loaded, lat).unbind()[0])


def test_checkpoint_shapes_stream_the_header_only(tmp_path):
    p = H.tiny_model_patcher()
    path = H.write_checkpoint(p, tmp_path / "tiny_h3.safetensors")
    shapes = H.checkpoint_shapes(path)
    sd = p.model.diffusion_model.state_dict()
    assert shapes == {k: tuple(v.shape) for k, v in sd.items()}
    assert shapes["video_patch_proj.weight"] == (H.TINY["hidden_size"], 24 * 4)
