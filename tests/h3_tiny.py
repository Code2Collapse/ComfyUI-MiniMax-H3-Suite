"""A tiny MiniMax H3 for CPU tests (owner A15 Phase 1.3, ledger L8.35 / L8.07).

Core's own classes end to end - comfy.supported_models.MiniMaxH3 -> model_base.MiniMaxH3 -> comfy.ldm.minimax.model
.MiniMaxH3Model, wrapped in a ModelPatcher and sampled by core's CFGGuider - with tiny dimensions and random weights
(about 230 k parameters, a 2-step sample in well under a second). The packed layout, RoPE, per-row timesteps, the
nested AV latent, denoise masks, wrappers and attention patches are core's real code; only the weights are fake, so
these tests prove plumbing (shapes, masks, positions, what reaches the model), never image quality.

ComfyUI picks xformers when it is installed, which has no CPU kernel, and the tiny model runs on the CPU: the H3
attention is pointed at core's own PyTorch attention (attention patches still run, it is wrapped like every backend).
"""
from __future__ import annotations

import torch

TINY = {"image_model": "minimax_h3", "hidden_size": 64, "num_layers": 2, "token_refiner_num_layers": 1,
        "num_attention_heads": 2, "attention_head_dim": 32, "ffn_hidden_size": 128, "text_dim": 48,
        "timestep_input_dim": 32, "time_embed_hidden_size": 64, "time_embed_dim": 32, "rope_inv_freq_len": 4}


def _cpu_attention():
    import comfy.ldm.minimax.model as mm
    import comfy.ldm.modules.attention as att
    mm.optimized_attention = att.attention_pytorch        # the tiny model always runs on the CPU


def tiny_model_patcher(seed: int = 0, **overrides):
    """A ModelPatcher around a tiny H3 with seeded random weights (and a real RoPE frequency table)."""
    import comfy.model_patcher
    import comfy.supported_models as sm

    _cpu_attention()
    cfg = sm.MiniMaxH3(dict(TINY, **overrides))
    cfg.set_inference_dtype(torch.float32, None)
    model = cfg.get_model({}, device="cpu")
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in model.diffusion_model.parameters():
            p.copy_(torch.randn(p.shape, generator=gen) * 0.02)
        inv = model.diffusion_model.rope.inv_freq
        inv.copy_(1.0 / (10000 ** (torch.arange(inv.numel(), dtype=torch.float32) / inv.numel())))
    cpu = torch.device("cpu")
    return comfy.model_patcher.ModelPatcher(model, load_device=cpu, offload_device=cpu)


def text(tokens: int = 6, seed: int = 0):
    """Conditioning: random 'text encoder states' of the tiny text width."""
    gen = torch.Generator().manual_seed(seed)
    return [[torch.randn(1, tokens, TINY["text_dim"], generator=gen), {}]]


def empty_latent(width: int = 64, height: int = 64, length: int = 22):
    from comfy_extras.nodes_minimax_h3 import EmptyMiniMaxH3LatentAV
    out = EmptyMiniMaxH3LatentAV.execute(width=width, height=height, length=length)
    return (out.result if hasattr(out, "result") else out.args)[0]


def noise_like(samples, seed: int = 1):
    import comfy.nested_tensor
    import comfy.sample
    if getattr(samples, "is_nested", False):
        return comfy.nested_tensor.NestedTensor([comfy.sample.prepare_noise(t, seed) for t in samples.unbind()])
    return comfy.sample.prepare_noise(samples, seed)


def sample(patcher, latent: dict, cond=None, steps: int = 2, seed: int = 1, sampler: str = "euler",
           sigmas=None, denoise: float = 1.0):
    """Core's CFGGuider.sample on the latent dict (its noise_mask honoured); returns the output samples."""
    import comfy.samplers

    cond = cond if cond is not None else text()
    if sigmas is None:
        ms = patcher.get_model_object("model_sampling")
        total = max(steps, int(steps / max(denoise, 1e-4)))
        sigmas = comfy.samplers.calculate_sigmas(ms, "simple", total).cpu()[-(steps + 1):]
    guider = comfy.samplers.CFGGuider(patcher)
    guider.inner_set_conds({"positive": cond})
    samples = latent["samples"]
    return guider.sample(noise_like(samples, seed), samples, comfy.samplers.sampler_object(sampler), sigmas,
                         denoise_mask=latent.get("noise_mask"), seed=seed, disable_pbar=True)


def write_checkpoint(patcher, path) -> str:
    """Save the tiny model as a diffusion-model safetensors file (the layout UNETLoader reads)."""
    from safetensors.torch import save_file
    sd = {k: v.detach().contiguous().clone() for k, v in patcher.model.diffusion_model.state_dict().items()}
    save_file(sd, str(path))
    return str(path)


def load_checkpoint(path):
    """Load a checkpoint the way UNETLoader does (comfy.sd.load_diffusion_model: model detection from the tensors)."""
    import comfy.sd
    _cpu_attention()
    patcher = comfy.sd.load_diffusion_model(str(path), model_options={"dtype": torch.float32})
    patcher.load_device = torch.device("cpu")       # tests never use the GPU (core would pick it when one exists)
    return patcher


def checkpoint_shapes(path) -> dict:
    """{tensor name: shape} read with safetensors.safe_open - the header only, no weight is loaded (usable on a real
    30 GB H3 checkpoint to check what it contains)."""
    from safetensors import safe_open
    with safe_open(str(path), framework="pt", device="cpu") as f:
        return {k: tuple(f.get_slice(k).get_shape()) for k in f.keys()}
