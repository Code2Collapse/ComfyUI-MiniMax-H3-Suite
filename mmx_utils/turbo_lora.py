# Apache License 2.0 — MiniMax-H3 Turbo LoRA reference
# PORTED FROM: comfyui-minimax-h3-turbo :: __init__.py @ 4274783a23afcfdbea3b4876cb79effd6c510785

"""H3 Turbo LoRA application — bypass/merge paths and pruned-base adaln injection."""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from mmx_utils.sigma_schedule import TRAINED_SHIFT_AUDIO, TRAINED_SHIFT_VIDEO, time_shift_sigma

SHIFT_V = TRAINED_SHIFT_VIDEO
SHIFT_A = TRAINED_SHIFT_AUDIO

TURBO_STEP_CONTRACT_KEY = "minimax_h3_turbo_step_contract"
TURBO_STEP_CONTRACT = (4, 8, 16)  # (target_steps, warn_above, refuse_above)

_EGRID = None
_EGRID_SEARCH = (
    Path(__file__).resolve().parent / "h3_silu_temb_grid.safetensors",
    Path(__file__).resolve().parents[2]
    / "third_party"
    / "comfyui-minimax-h3-turbo"
    / "h3_silu_temb_grid.safetensors",
)


class TurboGridUnavailableError(RuntimeError):
    """Bundled silu(t_emb) grid missing — pruned-base adaln injection cannot run."""


class TurboPrunedBaseUnavailableError(RuntimeError):
    """Pruned H3 base or grid weights required but not present on this machine."""


def resolve_egrid_path() -> Path | None:
    for p in _EGRID_SEARCH:
        if p.is_file():
            return p
    return None


def _egrid():
    global _EGRID
    if _EGRID is not None:
        return _EGRID
    path = resolve_egrid_path()
    if path is None:
        raise TurboGridUnavailableError(
            "h3_silu_temb_grid.safetensors is not on this machine. "
            "Copy it from the comfyui-minimax-h3-turbo release into mmx_utils/ "
            "or third_party/comfyui-minimax-h3-turbo/. "
            "Pruned-base Turbo LoRA adaln injection cannot run without it."
        )
    try:
        import comfy.utils
    except Exception as exc:
        raise TurboPrunedBaseUnavailableError(
            "MiniMaxH3_TurboLoRA needs ComfyUI (comfy.utils) to load the adaln grid."
        ) from exc
    _EGRID = comfy.utils.load_torch_file(str(path))["silu_t_emb_grid"]
    return _EGRID


def _unique_t(timestep, shift_v, shift_a, payload):
    sigma_v = (timestep.flatten()[0] / 1000.0).float().clamp(min=1e-6)
    t_v = float(1.0 - sigma_v)
    t_a = float(1.0 - time_shift_sigma(sigma_v, shift_v, shift_a))
    s = {t_v, t_a}
    refs = payload.get("refs") or ()
    if payload.get("keyframes") or any(r.get("kind") == "image" for r in refs):
        s.add(max(t_v, float(payload.get("visual_cond_noise_aug", 0.999))))
    if any(r.get("kind") == "audio" and r.get("ref_audio_t", 0) > 0 for r in refs):
        s.add(max(t_a, float(payload.get("audio_cond_noise_aug", 1.0))))
    return sorted(s)


def _interp_egrid(unique_t, egrid, device, dtype):
    egrid = egrid.to(device)
    n = egrid.shape[0]
    rows = []
    for t in unique_t:
        pos = min(max(t, 0.0), 1.0) * (n - 1)
        i0 = min(int(math.floor(pos)), n - 2)
        rows.append(torch.lerp(egrid[i0].float(), egrid[i0 + 1].float(), pos - i0))
    return torch.stack(rows).to(dtype)


def _make_adaln_forward(base, a, b, shared, table=None, egrid=None):
    def forward(t_emb):
        x = base.linear(F.silu(t_emb) if base.apply_silu else t_emb)
        st = None
        if table is not None and egrid is not None and not base.apply_silu:
            try:
                tb = table.to(t_emb.device, torch.float32)
                idx = torch.cdist(t_emb.detach().float(), tb).argmin(dim=1)
                st = egrid.to(t_emb.device)[idx]
            except Exception:
                st = None
        if st is None:
            st = shared.get("silu_temb")
        if st is not None and st.shape[0] == x.shape[0]:
            av = a.to(x.device, x.dtype)
            bv = b.to(x.device, x.dtype)
            sv = st.to(x.device, x.dtype)
            x = x + (bv @ (av @ sv.T)).T
        x = x.view(x.shape[0] * base.modalities, base.expand * base.hidden)
        return x.chunk(base.expand, dim=-1)

    return forward


def _FrugalLoRAClass():
    try:
        import comfy.lora
        import comfy.weight_adapter
    except Exception as exc:
        raise RuntimeError("Turbo LoRA needs comfy.lora and comfy.weight_adapter.") from exc

    class FrugalLoRA(comfy.weight_adapter.LoRAAdapter):
        """In-place additive LoRA bypass — see upstream docstring."""

        def bypass_forward(self, org_forward, x, *args, **kwargs):
            base_out = org_forward(x, *args, **kwargs)
            if getattr(self, "is_conv", False):
                return super().bypass_forward(org_forward, x, *args, **kwargs)
            up, down, alpha = self.weights[0], self.weights[1], self.weights[2]
            rank = down.shape[0]
            scale = (alpha / rank if alpha is not None else 1.0) * getattr(self, "multiplier", 1.0)
            down = down.to(dtype=x.dtype)
            up = up.to(dtype=x.dtype)
            return base_out.add_(F.linear(F.linear(x, down), up), alpha=scale)

    return FrugalLoRA


def _apply_bypass_lora(new_model, lora, modules, strength):
    import comfy.lora
    import comfy.weight_adapter

    FrugalLoRA = _FrugalLoRAClass()
    key_map = {m: f"diffusion_model.{m}.weight" for m in modules}
    loaded = comfy.lora.load_lora(lora, key_map, log_missing=False)
    manager = comfy.weight_adapter.BypassInjectionManager()
    sd_keys = set(new_model.model.state_dict().keys())
    n = 0
    for key, adapter in loaded.items():
        if key not in sd_keys:
            continue
        if isinstance(adapter, comfy.weight_adapter.LoRAAdapter):
            adapter = FrugalLoRA(adapter.loaded_keys, adapter.weights)
        elif not isinstance(adapter, comfy.weight_adapter.WeightAdapterBase):
            continue
        manager.add_adapter(key, adapter, strength=strength)
        n += 1
    injections = manager.create_injections(new_model.model)
    if manager.get_hook_count() > 0:
        new_model.set_injections("bypass_lora", injections)
    return n


def _apply_merge_lora(new_model, lora, modules, strength):
    import comfy.lora

    key_map = {m: f"diffusion_model.{m}.weight" for m in modules}
    loaded = comfy.lora.load_lora(lora, key_map, log_missing=False)
    return len(new_model.add_patches(loaded, strength))


def _int8_fused_fc2(dm, modules):
    import comfy.utils

    fused = []
    for m in modules:
        if not m.endswith(".mlp.fc2"):
            continue
        try:
            w = comfy.utils.get_attr(dm, m + ".weight")
        except Exception:
            continue
        if (
            getattr(w, "_layout_cls", None) == "TensorWiseINT8Layout"
            and not getattr(getattr(w, "_params", None), "transposed", False)
        ):
            fused.append(m)
    return fused


def _inject_adaln_egrid(new_model, dm, lora, adaln, strength):
    import comfy.patcher_extension

    egrid = _egrid()
    shared = {"silu_temb": None}
    shift_v = float(getattr(dm, "sigma_shift_video", SHIFT_V))
    shift_a = float(getattr(dm, "sigma_shift_audio", SHIFT_A))

    tt = None
    for _n, _t in list(dm.named_buffers()) + list(dm.named_parameters()):
        if _n.endswith("adaln_t_table"):
            tt = _t
            break
    if tt is not None and tt.shape[0] != egrid.shape[0]:
        tt = None

    def wrap(executor, *args, **kwargs):
        ts = args[1] if len(args) > 1 else kwargs.get("timestep")
        ctx = args[2] if len(args) > 2 else kwargs.get("context")
        payload = kwargs.get("minimax_payload") or {}
        us = _unique_t(ts, shift_v, shift_a, payload)
        shared["silu_temb"] = _interp_egrid(us, egrid, ctx.device, ctx.dtype)
        return executor(*args, **kwargs)

    new_model.add_wrapper_with_key(
        comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, "h3turbo", wrap
    )
    for name in adaln:
        a = lora[name + ".lora_A.weight"]
        b = lora[name + ".lora_B.weight"] * strength
        key = "diffusion_model." + name.rsplit(".linear", 1)[0]
        new_model.add_object_patch(
            key + ".forward",
            _make_adaln_forward(new_model.get_model_object(key), a, b, shared, tt, egrid),
        )


def _add_dbg_wrapper(new_model, dm, tag, mode):
    import comfy.patcher_extension

    st = {"n": 0}

    def wrap(executor, *args, **kwargs):
        if st["n"] < 6:
            st["n"] += 1
        return executor(*args, **kwargs)

    new_model.add_wrapper_with_key(
        comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, "h3turbo_dbg", wrap
    )


def stash_turbo_step_contract(model) -> Any:
    """Attach turbo step contract to transformer_options; return cloned model."""
    new_model = model.clone()
    to = new_model.model_options.setdefault("transformer_options", {})
    to[TURBO_STEP_CONTRACT_KEY] = TURBO_STEP_CONTRACT
    return new_model


def read_turbo_step_contract(model) -> tuple[int, int, int] | None:
    if model is None:
        return None
    opts = getattr(model, "model_options", {}) or {}
    to = opts.get("transformer_options", {}) or {}
    contract = to.get(TURBO_STEP_CONTRACT_KEY)
    if contract is None:
        return None
    return tuple(int(x) for x in contract)


def apply_turbo_lora(
    model,
    lora_path: str,
    strength: float,
    *,
    low_vram: bool = False,
) -> Any:
    """Apply turbo LoRA to a ComfyUI MODEL patcher. Raises on pruned path without grid."""
    try:
        import comfy.utils
    except Exception as exc:
        raise RuntimeError("MiniMaxH3_TurboLoRA needs ComfyUI.") from exc

    if not lora_path or not os.path.isfile(lora_path):
        raise FileNotFoundError(f"Turbo LoRA weights not found: {lora_path!r}")

    lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
    dm = model.model.diffusion_model
    pruned = bool(getattr(dm, "use_adaln_curves", False))
    modules = sorted({k.rsplit(".lora_", 1)[0] for k in lora if ".lora_" in k})

    if pruned and resolve_egrid_path() is None:
        raise TurboPrunedBaseUnavailableError(
            "This checkpoint uses the pruned H3 base (adaln curves), but "
            "h3_silu_temb_grid.safetensors is not on this machine. "
            "Copy the grid file from comfyui-minimax-h3-turbo into mmx_utils/ "
            "and ensure the pruned H3 base weights are loaded. "
            "Cannot apply Turbo LoRA without both."
        )

    new_model = stash_turbo_step_contract(model)
    mode = "merge" if low_vram else "bypass"

    if pruned:
        backbone = [m for m in modules if "adaln_proj" not in m]
        adaln = [m for m in modules if "adaln_proj" in m]
    else:
        backbone, adaln = modules, []

    n_fc2 = 0
    if low_vram:
        n = _apply_merge_lora(new_model, lora, backbone, strength)
    else:
        fc2_fused = set(_int8_fused_fc2(dm, backbone))
        bypass_mods = [m for m in backbone if m not in fc2_fused]
        n = _apply_bypass_lora(new_model, lora, bypass_mods, strength)
        if fc2_fused:
            n_fc2 = _apply_merge_lora(new_model, lora, sorted(fc2_fused), strength)
            n += n_fc2

    if pruned and adaln:
        _inject_adaln_egrid(new_model, dm, lora, adaln, strength)

    _add_dbg_wrapper(new_model, dm, "pruned" if pruned else "full", mode)
    return new_model
