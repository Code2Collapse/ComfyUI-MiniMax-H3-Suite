# Apache License 2.0 — MiniMax-H3 Turbo LoRA / sampler reference
# PORTED FROM: comfyui-minimax-h3-turbo :: __init__.py @ 4274783a23afcfdbea3b4876cb79effd6c510785

"""4-step H3 turbo sampler — native ModelSamplingAV vs legacy dual-schedule."""

from __future__ import annotations

import math
from typing import Any

import torch
from tqdm.auto import trange

from mmx_utils.sigma_schedule import TRAINED_SHIFT_AUDIO, TRAINED_SHIFT_VIDEO, time_shift_sigma

SHIFT_V = TRAINED_SHIFT_VIDEO
SHIFT_A = TRAINED_SHIFT_AUDIO


def _time_shift_slope(sigma: float, fr: float, to: float) -> float:
    base = float(sigma) / (fr + float(sigma) * (1.0 - fr))
    return (to * (1.0 + (fr - 1.0) * base) ** 2) / (fr * (1.0 + (to - 1.0) * base) ** 2)


def _audio_sigma(sv: float) -> float:
    return float(time_shift_sigma(sv, SHIFT_V, SHIFT_A))


def _audio_slope(sv: float) -> float:
    return _time_shift_slope(sv, SHIFT_V, SHIFT_A)


def _latent_shapes(model) -> Any:
    guider = getattr(model, "inner_model", model)
    conds = getattr(guider, "conds", None)
    if conds:
        for cond_list in conds.values():
            for c in cond_list or []:
                mc = c.get("model_conds", {}) if isinstance(c, dict) else {}
                if "latent_shapes" in mc:
                    return mc["latent_shapes"].cond
    return None


def _model_sampling(model):
    for chain in (
        ("inner_model", "inner_model", "model_sampling"),
        ("inner_model", "model_sampling"),
        ("model_sampling",),
    ):
        o = model
        try:
            for attr in chain:
                o = getattr(o, attr)
        except AttributeError:
            continue
        if o is not None:
            return o
    return None


def native_av_schedule(model) -> bool:
    """True when ComfyUI resolves H3 audio/video via ModelSamplingAV natively."""
    ms = _model_sampling(model)
    if ms is None:
        return False

    # `audio_shift` is read off the model itself, so decide on it BEFORE touching
    # comfy. Importing first and returning False on failure made native detection
    # depend on comfy.model_sampling being importable — which it is not when the
    # comfy_kitchen skew bites — and silently sent native models down the legacy
    # dual-clock path.
    if getattr(ms, "audio_shift", None) is not None:
        return True

    # Bind the SUBMODULE, not the parent package. `import comfy.model_sampling`
    # binds the local name `comfy`, and a submodule can sit in sys.modules without
    # being set as an attribute on its parent — the import then succeeds while
    # `comfy.model_sampling` raises AttributeError. The `as` form works either way.
    try:
        import comfy.model_sampling as _model_sampling_mod
    except Exception:
        return False
    av = getattr(_model_sampling_mod, "ModelSamplingAV", None)
    return av is not None and isinstance(ms, av)


@torch.no_grad()
def turbo_sampler(
    model,
    x,
    sigmas,
    extra_args=None,
    callback=None,
    disable=None,
    **kwargs,
):
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])

    if native_av_schedule(model):
        for i in trange(len(sigmas) - 1, disable=disable):
            sv, sv_n = float(sigmas[i]), float(sigmas[i + 1])
            denoised = model(x, sigmas[i] * s_in, **extra_args)
            d = (x - denoised) / sigmas[i]
            x = x + (sv_n - sv) * d
            if callback is not None:
                callback(
                    {
                        "i": i,
                        "denoised": denoised,
                        "x": x,
                        "sigma": sigmas[i],
                        "sigma_hat": sigmas[i],
                    }
                )
        return x

    shapes = _latent_shapes(model)
    if not shapes or len(shapes) < 2:
        raise RuntimeError(
            "MiniMaxH3_TurboSampler expects the MiniMax-H3 video+audio latent "
            "(EmptyMiniMaxH3LatentAV / MiniMaxH3ImageToVideo output)."
        )
    v_numel = math.prod(shapes[0][1:])
    for i in trange(len(sigmas) - 1, disable=disable):
        sv, sv_n = float(sigmas[i]), float(sigmas[i + 1])
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        out = (x - denoised) / sigmas[i]
        xv, ov = x[..., :v_numel], out[..., :v_numel]
        xa, oa = x[..., v_numel:], out[..., v_numel:]
        xv = xv + (sv_n - sv) * ov
        sl = _audio_slope(max(sv, 1e-6))
        xa = xa + (_audio_sigma(sv_n) - _audio_sigma(sv)) * (oa / sl)
        x = torch.cat([xv, xa], dim=-1)
        if callback is not None:
            callback(
                {
                    "i": i,
                    "denoised": denoised,
                    "x": x,
                    "sigma": sigmas[i],
                    "sigma_hat": sigmas[i],
                }
            )
    return x


def build_ksampler():
    """Return a ComfyUI KSAMPLER wrapping turbo_sampler."""
    try:
        import comfy.samplers
    except Exception as exc:
        raise RuntimeError(
            "MiniMaxH3_TurboSampler needs ComfyUI's comfy.samplers module."
        ) from exc
    return comfy.samplers.KSAMPLER(turbo_sampler)
