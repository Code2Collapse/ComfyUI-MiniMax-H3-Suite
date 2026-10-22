# PORTED FROM: https://github.com/supElement/ComfyUI_MinimaxH3_AutoContext@bfa7ac3 h3_tst_patch.py
# (_spectral_tension, _attention_override, _forward_wrapper, patch_model), Apache-2.0. Method: Spectral Transport
# Homeostasis, "Temporal State Transport in Video Generation: Diagnosing and Correcting Spectral Imbalance",
# arXiv 2609.08505 (ICML 2026). Kept: the tension (row entropy minus von Neumann entropy of the frame-level transport),
# gamma = exp(tau * layer weight * step weight * tension) on the video-row queries, the length guard, the chaining with
# an earlier attention override. Added: a kill switch (MMX_TST=off), the layer count read from the model (upstream
# guessed 50 until the first forward ended), and a per-sampling summary (how many attention calls
# were corrected and by how much), so an A/B has numbers. Checked against core 0.39 (comfy/ldm/minimax/model.py): the
# attention gets q as [1, heads, S, head_dim] (skip_reshape=True) and the target video rows are the LAST rows of the
# packed sequence (PackedLayout: audio then video, always the last two segments).
"""H3 TST attention: a training-free correction of the video queries against temporal flicker / small-face collapse.

For every video attention call it measures, per head, the frame-level "spectral tension" of the attention: positive
when frames over-mix, negative when they fragment. The video queries are scaled by exp(tension * tau_eff), which
sharpens over-mixed heads and softens fragmented ones. tau_eff is strongest in deep layers and early steps.
"""

from __future__ import annotations

import math
import os
from functools import partial

import torch

KILL_SWITCH = "MMX_TST"
WRAPPER_KEY = "mmx_h3_tst_attention"
ACTIVE = 0.03          # the paper's |gamma - 1| threshold for a call that departs from identity


def disabled() -> bool:
    return os.environ.get(KILL_SWITCH, "").strip().lower() in ("off", "0", "false", "no", "disable", "disabled")


def spectral_tension(q: torch.Tensor, k: torch.Tensor, v0: int, frames: int, rows_per_frame: int,
                     eps: float = 1e-8) -> torch.Tensor:
    """[heads] fp32 tension of the video rows starting at v0 of q, k [1, heads, S, head_dim] (post norm, post RoPE).
    > 0: the frames over-mix; < 0: they fragment; 0: balanced."""
    heads, head_dim = q.shape[1], q.shape[-1]
    qv = q[0, :, v0:].reshape(heads, frames, rows_per_frame, head_dim).mean(dim=2).float()
    kv = k[0, :, v0:].reshape(heads, frames, rows_per_frame, head_dim).mean(dim=2).float()
    a = (qv @ kv.transpose(-2, -1) * head_dim ** -0.5).softmax(dim=-1).clamp(min=eps)
    log_f = math.log(frames)
    h_row = -(a * a.log()).sum(dim=-1).mean(dim=-1) / log_f
    gram = a @ a.transpose(-2, -1)
    trace = gram.diagonal(dim1=-2, dim2=-1).sum(dim=-1).clamp(min=eps)
    eig = torch.linalg.eigvalsh(gram / trace[:, None, None]).clamp(min=eps)
    eig = eig / eig.sum(dim=-1, keepdim=True).clamp(min=eps)
    h_vn = -(eig * eig.log()).sum(dim=-1) / log_f
    return h_row - h_vn


def new_state(tau: float) -> dict:
    return {"tau": float(tau), "frames": None, "rows_per_frame": None, "step": 0, "total_steps": 1,
            "layers": 50,                  # a first guess; corrected by the real call count after the first forward
            "calls_this_forward": 0, "stats": {"calls": 0, "active": 0, "gmin": None, "gmax": None, "tsum": 0.0}}


def _unwrap(t):
    try:
        from comfy.ldm.modules.attention import AttentionTensorContainer
    except Exception:  # noqa: BLE001
        return t
    return t.peek() if isinstance(t, AttentionTensorContainer) else t


def attention_override(state: dict, inner=None):
    """transformer_options["optimized_attention_override"]: corrects q, then hands every argument on to `inner` (an
    override that was already set) or to core's attention function."""

    def dispatch(func, q, k, v, heads, mask, **kwargs):
        if inner is not None:
            return inner(func, q, k, v, heads, mask=mask, **kwargs)
        return func(q, k, v, heads, mask=mask, **kwargs)

    def override(func, q, k, v, heads, mask=None, **kwargs):
        frames, rows = state["frames"], state["rows_per_frame"]
        qt = _unwrap(q)
        s = qt.shape[2] if qt.ndim == 4 else 0
        if disabled() or frames is None or frames < 2 or s <= frames * rows:
            return dispatch(func, q, k, v, heads, mask, **kwargs)     # not a video attention (e.g. token refiner)
        state["calls_this_forward"] += 1
        layers = state["layers"]
        layer = (state["calls_this_forward"] - 1) % max(1, layers)
        total = state["total_steps"]
        step = min(state["step"], total - 1)
        lw = 0.5 - 0.5 * math.cos(math.pi * layer / (layers - 1)) if layers > 1 else 1.0
        sw = 0.5 + 0.5 * math.cos(math.pi * step / (total - 1)) if total > 1 else 1.0
        v0 = s - frames * rows
        tension = spectral_tension(qt, _unwrap(k), v0, frames, rows)
        gamma = torch.exp(tension * (state["tau"] * lw * sw))
        if state["tau"] != 0.0:
            qt[0, :, v0:].mul_(gamma.to(qt.dtype)[:, None, None])
        st = state["stats"]
        g = gamma.detach().float()
        st["calls"] += 1
        st["active"] += int(bool(((g - 1).abs() > ACTIVE).any()))
        st["gmin"] = float(g.min()) if st["gmin"] is None else min(st["gmin"], float(g.min()))
        st["gmax"] = float(g.max()) if st["gmax"] is None else max(st["gmax"], float(g.max()))
        st["tsum"] += float(tension.detach().float().mean())
        return dispatch(func, q, k, v, heads, mask, **kwargs)

    return override


def forward_wrapper(state, executor, x, timestep, context, transformer_options, *args, **kwargs):
    """DIFFUSION_MODEL wrapper: resets the layer counter and reads the video shape of this forward."""
    if state["calls_this_forward"] > 0:
        state["layers"] = state["calls_this_forward"]
    state["calls_this_forward"] = 0
    video = x[0] if isinstance(x, (list, tuple)) else x
    if isinstance(video, torch.Tensor) and video.ndim == 5:
        state["frames"] = int(video.shape[2])
        state["rows_per_frame"] = ((int(video.shape[3]) + 1) // 2) * ((int(video.shape[4]) + 1) // 2)
    else:
        state["frames"] = None
    sigmas = transformer_options.get("sample_sigmas")
    if sigmas is not None:
        sigma = float(timestep.flatten()[0]) / 1000.0
        state["total_steps"] = max(1, int(sigmas.numel()) - 1)
        state["step"] = int((sigmas.float().cpu() - sigma).abs().argmin())
    return executor(x, timestep, context, transformer_options, *args, **kwargs)


def summary(state: dict) -> str:
    st = state["stats"]
    if not st["calls"]:
        return "TST: no video attention call was corrected (not an H3 model, or the kill switch is on)"
    return (f"TST tau={state['tau']:.2f}: {st['calls']} video attention calls corrected over {state['layers']} layers, "
            f"{100.0 * st['active'] / st['calls']:.0f}% moved a head by more than {ACTIVE:.0%}, gamma "
            f"{st['gmin']:.3f}-{st['gmax']:.3f}, mean tension {st['tsum'] / st['calls']:+.3f}")


def sample_wrapper(state, log, executor, *args, **kwargs):
    """OUTER_SAMPLE wrapper: one summary line per sampling run."""
    state["stats"] = new_state(state["tau"])["stats"]
    try:
        return executor(*args, **kwargs)
    finally:
        log(summary(state))


def patch_model(model, tau: float, log=print):
    """A clone of `model` with the correction mounted; an attention override that was already set keeps working,
    inside this one."""
    import comfy.patcher_extension as pe

    state = new_state(tau)
    try:        # the real block count from the start: the 50-layer guess mis-weighted the first (strongest) step
        state["layers"] = max(1, len(model.model.diffusion_model.blocks))
    except Exception:  # noqa: BLE001 - not an H3 model object: keep the guess, corrected after one forward
        pass
    patched = model.clone()
    patched.add_wrapper_with_key(pe.WrappersMP.DIFFUSION_MODEL, WRAPPER_KEY, partial(forward_wrapper, state))
    patched.add_wrapper_with_key(pe.WrappersMP.OUTER_SAMPLE, WRAPPER_KEY, partial(sample_wrapper, state, log))
    to = dict(patched.model_options.get("transformer_options") or {})
    to["optimized_attention_override"] = attention_override(state, to.get("optimized_attention_override"))
    to["h3_tst_tau"] = float(tau)
    patched.model_options["transformer_options"] = to
    return patched, state
