# PORTED FROM: https://github.com/rdh9616/ComfyUI-Open-H3-Regenerate@46ff778 __init__.py (Apache-2.0)
# Copyright 2026 rdh9616. Licensed under the Apache License, Version 2.0 (THIRD_PARTY_LICENSES/
# ComfyUI-Open-H3-Regenerate-LICENSE.txt, NOTICE in ComfyUI-Open-H3-Regenerate-NOTICE.txt).
# Modified by Code2Collapse, 2026-11: V3 schema, Suite upscaler engine, comfy.audio resample, mmap-safe cache.
# Distributed as part of a GPL-3.0 work.
# Changes in this file (spec docs/research/H3_REGEN_STD_spec.md, work order H3-REGEN-STD-001):
#   * V3 io.ComfyNode nodes with Suite ids MiniMaxH3_OpenRegen*, a `report` output appended last on most nodes;
#   * Latent Upscale runs the Suite's own engine (mmx_nodes/latent_upscale_model.upscale, S1) instead of calling another
#     pack's internal execute(); .safetensors upscalers only; device auto = ComfyUI's device (CUDA / ROCm / XPU);
#   * Refine Latent: 17k+5 maths from mmx_utils/av_latent_ops (T must be 5k+2), the noise mask in the layout
#     MiniMaxH3_SetAVNoiseMask writes, draft audio resampled with comfy.audio.resample (no torchaudio);
#   * LoRA cache: every kept tensor is copied off the safetensors memory map (Windows error 1224 fix);
#   * Fast VAE Decode: interrupt checks between decoder layers / frames, model released in `finally`;
#   * Save Refined Latent writes atomically (tmp + os.replace); Load uses fingerprint_inputs = name + mtime.
#   * OpenH3VideoToPrompt is not ported (owner decision D8 default: use MiniMaxH3_PromptWriter / MiniMaxH3_Vision).
#   * 2026-11-25 (owner: one pipeline): IC options appended to LoRA / Sigmas / Prepare Draft / Refine Latent. Their
#     logic is PORTED FROM https://github.com/slmonker/ComfyUI-OpenH3Regenerate2KIC@a85bd96 nodes.py (Apache-2.0,
#     author slmonker) via mmx_utils/open_regen_ic.py; defaults keep the rdh9616 behaviour.
"""H3 Open Regen (standard): regenerate a draft MiniMax H3 video (or AV latent) at 1080p / 2K / 4K.

draft -> Prepare Draft -> VAE Encode -> Latent Upscale -> Refine Latent -> sampler (FL2VA + Open Regen LoRA +
BlockSparseAttention, Open Regen Sigmas, Euler, CFG 1) -> Save Refined Latent -> Fast VAE Decode -> Finish Frames.
"""

from __future__ import annotations

import copy
import gc
import hashlib
import logging
import math
import os
import re
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from comfy_api.latest import io

_PKG = Path(__file__).resolve().parents[1]
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from mmx_nodes import latent_upscale_model as LU  # noqa: E402
from mmx_utils import open_regen_core as core  # noqa: E402

LOG = logging.getLogger("MiniMaxSuite.open_regen")
CATEGORY = "MiniMax H3/Regenerate STD"
ATTRIBUTION = "Ported from rdh9616/ComfyUI-Open-H3-Regenerate (Apache-2.0); fast decoder by NVIDIA (Apache-2.0)."

# ---- owner decisions (spec "Owner decisions"): defaults applied here, each kept in one place so it is easy to change
D2_START_SIGMA_DEFAULT = 0.0          # D2: 0 = NVIDIA release grid (upstream node default); the workflows set 0.8
D5_FAST_VAE_FILTER = "fast_h3_vae_noncausal"   # D5: default file filter for the fast decoder (strict load)
D9_LORA_DTYPE_DEFAULT = "bf16"        # D9: RAM copy of the LoRA factors (bf16 6.2 GB, fp32 12.4 GB)
# D4 (.safetensors only), D8 (no Video->Prompt), D11 (core 0.39 only), D13 (no upstream-id aliases): by construction.

OWN_TABLE = "(model's own table)"
NO_CURVE = "none"
SIZES = ["768p (1344x768)", "576p (1024x576)", "keep (snap to 32)"]
RESOLUTIONS = ["768p", "1080p", "2k", "4k"]
AUDIO_MODES = ["silence (reference)", "draft audio (AUDIO input)", "keep latent audio"]
CROPS = ["auto (1088 -> 1080, 2176 -> 2160)", "none"]
# IC options folded into the STD nodes (owner 2026-11-25: one pipeline; rdh9616's nodes, slmonker's IC recipe)
SIGMA_GRIDS = ["regenerate (steps / start_sigma)", "IC up3", "IC up4", "IC hf8", "custom"]
FPS_MAPS = ["nearest (regenerate)", "floor (IC official)"]
AUDIO_ASSET = "silence asset file (IC, 124 frames)"
LATENT_DIR = "open_h3_latents"          # under ComfyUI's output folder (same as upstream, so its files load)
LATENT_FORMAT = "open_h3_refined_latent_v1"
NEWEST = "(newest)"


def _pick(options, *needles):
    for needle in needles:
        for name in options:
            if needle in name.lower():
                return name
    return options[0] if options else ""


def _files(folder: str) -> list[str]:
    try:
        import folder_paths
        return list(folder_paths.get_filename_list(folder))
    except Exception:  # noqa: BLE001
        return []


def _full_path(folder: str, name: str) -> str:
    import folder_paths
    path = folder_paths.get_full_path(folder, name)
    if path is None or not os.path.isfile(path):
        raise FileNotFoundError(f"{name} not found in models/{folder}. Put the file there and refresh ComfyUI.")
    return path


def _nested(parts):
    import comfy.nested_tensor
    return comfy.nested_tensor.NestedTensor(tuple(parts))


def _h3_video(latent, what):
    """(video [B,24,T,h,w], audio or None) of a plain H3 video latent or an H3 AV latent; anything else refused."""
    s = latent.get("samples") if isinstance(latent, dict) else None
    if s is None:
        raise ValueError(f"{what}: expected a LATENT")
    if isinstance(s, torch.Tensor) and (s.ndim != 5 or s.shape[1] != 24):
        raise ValueError(f"{what}: expected an H3 video latent [B, 24, T, H, W], got {tuple(s.shape)} - connect the "
                         "MiniMax H3 video VAE's encode (or an H3 audio + video latent)")
    video, audio = LU._parts(latent, what)
    if video.ndim != 5 or video.shape[1] != 24:
        raise ValueError(f"{what}: expected an H3 video latent [B, 24, T, H, W], got {tuple(video.shape)}")
    return video, audio


def _mm():
    import comfy.model_management as mm
    return mm


# ====================================================================================================== LoRA + curves

_LORA_CACHE: dict = {}


def _off_map(tensor, dtype):
    """A tensor that owns its memory (never a view of the safetensors memory map, which on Windows keeps the file
    locked: error 1224 on overwrite). A dtype change already copies; same dtype -> clone."""
    return tensor.to(dtype) if tensor.dtype != dtype else tensor.clone()


def load_lora_pairs(path, dtype):
    """{diffusers module: (A, B in ComfyUI row order, ComfyUI weight key, slot)} of the merged LoRA, read one tensor at a
    time; the last file stays cached under (path, mtime, dtype)."""
    from safetensors import safe_open
    key = (path, os.path.getmtime(path), dtype)
    if key not in _LORA_CACHE:
        _LORA_CACHE.clear()
        gc.collect()
        pairs = {}
        with safe_open(path, framework="pt", device="cpu") as f:
            for module in core.lora_modules(set(f.keys())):
                target, slot = core.comfy_target(module)
                b = _off_map(f.get_tensor(module + ".lora_B.weight"), dtype)
                if slot == "swiglu":
                    b = core.swap_swiglu_halves(b)
                pairs[module] = (_off_map(f.get_tensor(module + ".lora_A.weight"), dtype), b, target, slot)
        _LORA_CACHE[key] = pairs
    return _LORA_CACHE[key]


def read_table(path):
    from safetensors import safe_open
    with safe_open(path, framework="pt", device="cpu") as f:
        names = [k for k in f.keys() if k.endswith("adaln_t_table")]
        if not names:
            raise ValueError(f"{os.path.basename(path)} has no adaln_t_table (pick a pruned / curve-form H3 checkpoint, "
                             "e.g. minimax_h3_ref2va_pruned_int8_convrot)")
        return f.get_tensor(names[0]).to(torch.float32).clone().contiguous()


def table_sha256(table):
    return hashlib.sha256(table.to(torch.float32).contiguous().numpy().tobytes()).hexdigest()


class CurveState:
    """Shared by every block's wrapper: where the model time of a forward comes from."""

    def __init__(self, table, mode, model_table=None):
        self.table = table              # reference curve (1025 x 8) the Sharp branches were trained on
        self.mode = mode                # "same": the AdaLN input already is that curve; "invert": find t on the
        self.model_table = model_table  # model's own curve; "recorded": t from the time embedder
        self.t = None
        self.warned = False
        self._device_tables = {}

    def table_on(self, device):
        if device not in self._device_tables:
            self._device_tables[device] = self.table.to(device)
        return self._device_tables[device]

    def coords(self, adaln_input):
        device = adaln_input.device
        if self.mode == "same":
            return adaln_input.to(torch.float32)
        if self.mode == "invert":
            t, err = core.invert_table(self.model_table, adaln_input.to(torch.float32))
            if err > 1e-3 and not self.warned:
                self.warned = True
                LOG.warning("H3 Open Regen Sharp curve: AdaLN input is %.2g off the model's curve", err)
            return core.table_coords(self.table_on(device), t)
        t = self.t
        if t is None or t.shape[0] != adaln_input.shape[0]:
            if not self.warned:
                self.warned = True
                LOG.warning("H3 Open Regen Sharp curve: no matching timestep recorded, curve branches skipped")
            return None
        return core.table_coords(self.table_on(device), t.to(device))


def _subclass(module, name, body):
    """Shallow copy of `module` with a subclass of its class: same parameters / child modules, so every weight key and
    ComfyUI's casting / LoRA patching are unchanged; only forward differs."""
    out = copy.copy(module)
    out.__class__ = type(name, (module.__class__,), body)
    return out


def _curve_linear(linear, weight, state, strength):
    def forward(self, *args, **kwargs):
        out = super(self.__class__, self).forward(*args, **kwargs)
        coords = state.coords(args[0] if args else kwargs["input"])
        if coords is None:
            return out
        w = self._open_h3_weights.get(coords.device)
        if w is None:
            w = self._open_h3_weights[coords.device] = weight.to(coords.device, torch.float32)
        return out + (coords @ w.t() * strength).to(out.dtype)

    wrapped = _subclass(linear, "OpenH3SharpCurveLinear", {"forward": forward})
    object.__setattr__(wrapped, "_open_h3_weights", {})
    return wrapped


def _recording_embedder(embedder, state):
    def forward(self, t):
        state.t = t.detach().to(torch.float32)
        return super(self.__class__, self).forward(t)

    return _subclass(embedder, "OpenH3RecordingTimeEmbedder", {"forward": forward})


def apply_lora(model, lora_path, strength, dtype):
    """(patched clone, report lines) for the merged PEFT LoRA at `lora_path`."""
    import comfy.lora
    pairs = load_lora_pairs(lora_path, dtype)
    state_dict = model.model.state_dict()
    lora, to_load = {}, {}
    for module, (a, b, key, slot) in pairs.items():
        if key not in state_dict:
            raise ValueError(f"H3 Open Regen LoRA: {key} is not in the model - connect a MiniMax H3 model "
                             f"(FL2VA for the standard pipeline); LoRA module {module}")
        rows, cols = state_dict[key].shape[0], state_dict[key].shape[-1]
        if a.shape[1] != cols:
            raise ValueError(f"H3 Open Regen LoRA: {key} has {cols} input columns, LoRA {module} has {a.shape[1]} - "
                             "connect a MiniMax H3 model")
        if isinstance(slot, int):
            if rows != 3 * b.shape[0]:
                raise ValueError(f"H3 Open Regen LoRA: {key} has {rows} rows, LoRA {module} gives {b.shape[0]} per "
                                 "q/k/v - connect a MiniMax H3 model")
            target = (key, (0, slot * b.shape[0], b.shape[0]))
        else:
            if rows != b.shape[0]:
                raise ValueError(f"H3 Open Regen LoRA: {key} has {rows} rows, LoRA {module} gives {b.shape[0]} - "
                                 "connect a MiniMax H3 model")
            target = key
        lora[module + ".lora_A.weight"], lora[module + ".lora_B.weight"] = a, b
        to_load[module] = target
    patches = comfy.lora.load_lora(lora, to_load)
    m = model.clone()
    applied = m.add_patches(patches, strength)
    weights = len({k if isinstance(k, str) else k[0] for k in applied})
    ram = sum(a.numel() * a.element_size() + b.numel() * b.element_size() for a, b, _, _ in pairs.values())
    return m, [f"LoRA {os.path.basename(lora_path)}: {len(pairs)} modules mapped -> {weights} ComfyUI weights "
               f"({len(applied)} patches; the released file gives 312 -> 208), strength {strength:g}, "
               f"{str(dtype).replace('torch.', '')} RAM copy {ram / 2 ** 30:.2f} GB"]


def _lora_keys(path):
    from safetensors import safe_open
    with safe_open(path, framework="pt", device="cpu") as f:
        return list(f.keys())


def _apply_plain_lora(model, name, strength, slot):
    """(patched clone, report line) for a ComfyUI-keyed LoRA (core loader, model only)."""
    from mmx_nodes import open_regen_ic as IC
    path = _full_path("loras", name)
    sd = IC._load_safetensors(path, f"H3 Open Regen LoRA ({slot})")
    try:
        patched, n = IC._apply_comfy(model, sd, strength)
    finally:
        del sd
    if n == 0:
        raise ValueError(f"H3 Open Regen LoRA: {slot} LoRA {name!r} matched no weight of this model - load the MiniMax "
                         "H3 Ref2VA (IC) / FL2VA model, or pick a ComfyUI-keyed H3 LoRA")
    return patched, f"{slot}: {name} strength {strength:g}, ComfyUI keys, {n} weights patched"


def apply_lora_recipe(model, lora_name, strength, dtype, speed_lora="none", speed_strength=1.0, sharp_lora="none",
                      sharp_strength=1.0):
    """Standard Regenerate (merged Diffusers LoRA) or the IC chain speed -> LMS Sharp -> IC LoRA, one node.

    lora_name may be NVIDIA's merged Regenerate LoRA (Diffusers, rank 576), the official IC LoRA (Diffusers, rank 384;
    same key layout, so the same exact q/k/v-slice + SwiGLU mapping applies) or a ComfyUI-keyed conversion of it
    (e.g. slmonker's minimax_h3_open_regenerate_2k_ic_comfy_bf16)."""
    from mmx_utils import open_regen_ic as R
    lines = []
    chosen = {"speed": speed_lora or "none", "sharp": sharp_lora or "none", "ic": lora_name}
    ic_chain = any(chosen[s] not in ("none", "") for s in ("speed", "sharp"))
    if ic_chain:
        lines += R.order_warnings(chosen)
    for slot, name, st in (("speed", chosen["speed"], speed_strength), ("sharp", chosen["sharp"], sharp_strength)):
        if name in ("none", ""):
            continue
        model, line = _apply_plain_lora(model, name, st, slot)
        lines.append(line)
    path = _full_path("loras", lora_name)
    fmt = R.detect_lora_format(_lora_keys(path))
    if fmt == "diffusers":
        m, rep = apply_lora(model, path, strength, dtype)
        pairs = load_lora_pairs(path, dtype)
        rank = int(next(iter(pairs.values()))[0].shape[0]) if pairs else 0
        kind = ("IC recipe: official Open-H3-Regenerate-2K-IC LoRA" if rank != 576 or ic_chain
                else "Regenerate recipe: NVIDIA merged refiner + Turbo + LMS Sharp LoRA")
        lines += rep
        lines.append(f"{kind} (Diffusers keys, rank {rank})")
    else:
        m, line = _apply_plain_lora(model, lora_name, strength, "ic")
        lines.append(line)
        lines.append("IC recipe: ComfyUI-keyed Open-H3-Regenerate-2K-IC LoRA")
    return m, lines


def apply_curves(m, curve_path, table, strength):
    """Add the Sharp curve branches to every block's AdaLN projection of the patched clone `m`; returns a report line."""
    from safetensors import safe_open
    dm = m.get_model_object("diffusion_model")
    curve_form = bool(getattr(dm, "use_adaln_curves", False))
    own = dm.adaln_t_table.detach().to("cpu", torch.float32) if curve_form else None
    if table is None:
        if own is None:
            raise ValueError("H3 Open Regen LoRA: this H3 model has no adaln_t_table (it is not a pruned / curve-form "
                             "checkpoint) - pick a pruned checkpoint in curve_table")
        table = own
    sha = table_sha256(table)
    tag = "reference table (sha256 match)" if sha == core.REFERENCE_TABLE_SHA256 else \
        f"NOT the reference table (sha256 {sha[:12]}, no match)"
    if curve_form:
        mode = "same" if own.shape == table.shape and torch.equal(own, table) else "invert"
    else:
        mode = "recorded"
    state = CurveState(table, mode, own)
    name = os.path.basename(curve_path)
    with safe_open(curve_path, framework="pt", device="cpu") as f:
        blocks = core.curve_blocks(set(f.keys()))
        if len(blocks) != len(dm.blocks):
            raise ValueError(f"H3 Open Regen LoRA: {name} has {len(blocks)} curve branches for {len(dm.blocks)} H3 "
                             "blocks - pick sharp_curve_branches.safetensors for this model")
        for index, (key_a, key_b) in sorted(blocks.items()):
            a = f.get_tensor(key_a).to(torch.float32).clone()
            b = f.get_tensor(key_b).to(torch.float32).clone()
            target = f"diffusion_model.blocks.{index}.adaln_proj.linear"
            linear = m.get_model_object(target)
            if b.shape[0] != linear.weight.shape[0] or a.shape[1] != table.shape[1] or b.shape[1] != a.shape[0]:
                raise ValueError(f"H3 Open Regen LoRA: {name} branch {index} {tuple(b.shape)} @ {tuple(a.shape)} does "
                                 f"not fit {target} {tuple(linear.weight.shape)} with a {table.shape[1]}-column table")
            m.add_object_patch(target, _curve_linear(linear, b @ a, state, strength))
    if mode == "recorded":
        m.add_object_patch("diffusion_model.time_embedder",
                           _recording_embedder(m.get_model_object("diffusion_model.time_embedder"), state))
    how = {"same": "same: the model's AdaLN curve is the reference curve",
           "invert": "invert: model time read back from the model's own curve, branches use the reference curve",
           "recorded": "recorded: full (non-pruned) model, model time recorded from the time embedder"}[mode]
    return f"Sharp curve branches: {len(blocks)} at {strength:g}, {tag}; mode {how}"


class MiniMaxH3_OpenRegenLoRA(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        loras = _files("loras")
        tables = _files("diffusion_models")
        lora_opts = loras or ["(put open_h3_regenerate_2k_lora.safetensors in models/loras)"]
        return io.Schema(
            node_id="MiniMaxH3_OpenRegenLoRA",
            has_intermediate_output=True,  # L9.290: core resends this ui when the node is served from cache
            display_name="H3 Open Regen · LoRA (refiner + Turbo + Sharp)",
            category=CATEGORY,
            description=("Applies NVIDIA's merged Open-H3-Regenerate-2K LoRA (refiner + LightX2V FL2VA Turbo + LMS "
                         "Sharp, Diffusers keys converted to ComfyUI's H3 keys) and LMS Sharp's 50 AdaLN curve "
                         "branches to an H3 FL2VA model. Then BlockSparseAttention, then the sampler. " + ATTRIBUTION),
            search_aliases=["open h3 regenerate lora", "regen lora", "h3 refiner lora"],
            inputs=[
                io.Model.Input("model", tooltip="A MiniMax H3 model (FL2VA for the standard pipeline)."),
                io.Combo.Input("lora_name", options=lora_opts, default=_pick(lora_opts, "open_h3_regenerate"),
                               tooltip="models/loras/open_h3_regenerate_2k_lora.safetensors (merged refiner + Turbo + "
                                       "LMS Sharp, Diffusers PEFT keys)."),
                io.Combo.Input("curve_name", options=[NO_CURVE] + loras,
                               default=_pick(loras, "sharp_curve_branches") or NO_CURVE,
                               tooltip="models/loras/sharp_curve_branches.safetensors: the 50 AdaLN curve branches "
                                       "(part of the reference recipe). none = skip."),
                io.Combo.Input("curve_table", options=[OWN_TABLE] + tables,
                               default=_pick(tables, "ref2va_pruned_int8_convrot", "ref2va_pruned") or OWN_TABLE,
                               tooltip="The curve the Sharp branches read: adaln_t_table (1025 x 8) of the reference "
                                       "checkpoint minimax_h3_ref2va_pruned_int8_convrot. Only that tensor is read."),
                io.Float.Input("strength", default=1.0, min=-2.0, max=2.0, step=0.01,
                               tooltip="LoRA strength. The merge is exact at 1.0 (strengths are folded in)."),
                io.Float.Input("curve_strength", default=1.0, min=-2.0, max=2.0, step=0.01,
                               tooltip="Strength of the Sharp curve branches."),
                io.Combo.Input("lora_dtype", options=["bf16", "fp32"], default=D9_LORA_DTYPE_DEFAULT, advanced=True,
                               tooltip="RAM copy of the rank-576 factors: bf16 about 6.2 GB, fp32 about 12.4 GB."),
                # IC recipe (slmonker Open-H3-Regenerate-2K-IC), appended 2026-11-25: speed -> LMS Sharp -> lora_name
                io.Combo.Input("speed_lora", options=["none"] + loras, default="none", optional=True,
                               tooltip="IC recipe, applied FIRST: the speed LoRA (HyperFlow 8-step; DMAD 4-step "
                                       "accepted). none = standard Regenerate recipe."),
                io.Float.Input("speed_strength", default=1.0, min=-2.0, max=2.0, step=0.01, optional=True,
                               tooltip="Model strength of the speed LoRA."),
                io.Combo.Input("sharp_lora", options=["none"] + loras, default="none", optional=True,
                               tooltip="IC recipe, applied SECOND: LMS Sharp r64 as a plain LoRA. none = standard "
                                       "Regenerate recipe."),
                io.Float.Input("sharp_strength", default=1.0, min=-2.0, max=2.0, step=0.01, optional=True,
                               tooltip="Model strength of the LMS Sharp LoRA."),
            ],
            outputs=[io.Model.Output("model", tooltip="A clone of the input H3 model with the merged Open Regen LoRA "
                                                     "patches (and, unless curve_name is none or curve_strength 0, "
                                                     "the Sharp curve branches on every block's AdaLN projection). "
                                                     "Wire to BlockSparseAttention, then the sampler."),
                     io.String.Output("report", tooltip="Text: LoRA modules mapped to ComfyUI weights, strength, RAM "
                                                        "copy size, and how the curve branches were applied (or "
                                                        "that they are off).")],
        )

    @classmethod
    def execute(cls, model, lora_name, curve_name, curve_table, strength, curve_strength,
                lora_dtype=D9_LORA_DTYPE_DEFAULT, speed_lora="none", speed_strength=1.0, sharp_lora="none",
                sharp_strength=1.0) -> io.NodeOutput:
        dtype = torch.bfloat16 if lora_dtype == "bf16" else torch.float32
        m, report = apply_lora_recipe(model, lora_name, float(strength), dtype, speed_lora, float(speed_strength),
                                      sharp_lora, float(sharp_strength))
        if curve_name != NO_CURVE and float(curve_strength) != 0:
            if any("IC recipe" in line for line in report):
                report.append("WARNING: Sharp curve branches are on with the IC LoRA; the IC recipe uses LMS Sharp "
                              "as a plain LoRA (sharp_lora) and no curve branches - set curve_name to none")
            table = None if curve_table == OWN_TABLE else read_table(_full_path("diffusion_models", curve_table))
            report.append(apply_curves(m, _full_path("loras", curve_name), table, float(curve_strength)))
        else:
            report.append("Sharp curve branches: off (not the reference recipe)")
        text = "\n".join(report)
        LOG.info(text)
        return io.NodeOutput(m, text, ui={"text": [text]})


# ====================================================================================================== sigmas

class MiniMaxH3_OpenRegenSigmas(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MiniMaxH3_OpenRegenSigmas",
            has_intermediate_output=True,  # L9.290: core resends this ui when the node is served from cache
            display_name="H3 Open Regen · Sigmas",
            category=CATEGORY,
            description=("The refine's Euler sigmas (CFG 1): 3 steps 0.909375 / 0.857 / 0.667 (NVIDIA release), 2 steps "
                         "0.7 / 0.494, or the shift-6 grid from start_sigma. Explicit sigmas make the model's sigma "
                         "shift irrelevant. " + ATTRIBUTION),
            search_aliases=["open h3 regenerate sigmas", "regen sigmas"],
            inputs=[
                io.Int.Input("steps", default=3, min=1, max=8,
                             tooltip="Euler steps. 3 = release setting; 2 = about 30% faster, softer fine texture. "
                                     "Other counts need start_sigma > 0."),
                io.Float.Input("start_sigma", default=D2_START_SIGMA_DEFAULT, min=0.0, max=1.0, step=0.001,
                               tooltip="0 = the release grid for 2 / 3 steps. Above 0: the refiner's shift-6 grid "
                                       "from this start noise (the example workflows use 0.8)."),
                io.Combo.Input("grid", options=SIGMA_GRIDS, default=SIGMA_GRIDS[0], optional=True,
                               tooltip="regenerate: steps / start_sigma above. IC up3 (default IC refine, 0.835 -> 0 "
                                       "in 3 steps), IC up4, IC hf8: the fixed Open-H3-Regenerate-2K-IC grids "
                                       "(steps / start_sigma ignored). custom: custom_sigmas."),
                io.String.Input("custom_sigmas", default="", optional=True,
                                tooltip="grid = custom: comma list ending at 0, never increasing, inside [0, 1]."),
            ],
            outputs=[io.Sigmas.Output("sigmas", tooltip="SIGMAS for the refine sampler (Euler, CFG 1): the NVIDIA "
                                                       "release grid for 2 / 3 steps, or the shift-6 grid from "
                                                       "start_sigma."),
                     io.String.Output("report", tooltip="Text: the sigma values and which grid produced them.")],
        )

    @classmethod
    def execute(cls, steps, start_sigma=D2_START_SIGMA_DEFAULT, grid=None, custom_sigmas="") -> io.NodeOutput:
        grid = grid or SIGMA_GRIDS[0]
        if grid == SIGMA_GRIDS[0]:
            values = core.refine_sigmas(int(steps), float(start_sigma))
            src = (f"shift-6 grid from start_sigma {float(start_sigma):g}" if start_sigma and float(start_sigma) > 0
                   else f"NVIDIA release grid ({int(steps)} steps)")
        elif grid in SIGMA_GRIDS:
            from mmx_utils import open_regen_ic as R
            if grid == "custom":
                if not (custom_sigmas or "").strip():
                    raise ValueError("H3 Open Regen Sigmas: grid = custom needs custom_sigmas (e.g. 0.8, 0.6, 0.3, 0)")
                values, _ = R.build_sigmas("up3", custom_sigmas)
                src = "custom list"
            else:
                values, _ = R.build_sigmas(grid.split()[-1], "")
                src = f"Open-H3-Regenerate-2K-IC {grid.split()[-1]} grid"
        else:
            raise ValueError(f"H3 Open Regen Sigmas: grid must be one of {', '.join(SIGMA_GRIDS)}")
        report = f"{', '.join(f'{v:.6f}' for v in values)} ({src})"
        return io.NodeOutput(torch.tensor(values, dtype=torch.float32), report, ui={"text": [report]})


# ====================================================================================================== frames

def prepare_draft(images, size, max_frames, source_fps=24.0, chunk=16, fps_map=FPS_MAPS[0], min_real_frames=0):
    """(frames [P,H,W,3], real frame count N, width, height, report).

    fps_map floor = the IC preprocessing (source frame int(i * fps / 24), slmonker / NVIDIA IC); nearest = rdh9616."""
    if not isinstance(images, torch.Tensor) or images.ndim != 4 or images.shape[0] == 0:
        raise ValueError("H3 Open Regen Prepare Draft: no frames - connect an IMAGE batch [frames, H, W, 3]")
    fps = float(source_fps)
    if math.isnan(fps) or math.isinf(fps) or fps < 0:
        raise ValueError(f"H3 Open Regen Prepare Draft: source_fps must be a number >= 0, got {source_fps}")
    if fps_map not in FPS_MAPS:
        raise ValueError(f"H3 Open Regen Prepare Draft: fps_map must be one of {', '.join(FPS_MAPS)}")
    notes = []
    n_in = images.shape[0]
    if fps and abs(fps - core.FPS) > 1e-3:
        if fps_map == FPS_MAPS[1]:
            out_n = max(1, int(n_in * core.FPS / fps))
            while out_n > 1 and int((out_n - 1) * fps / core.FPS) >= n_in:
                out_n -= 1
            idx = [int(i * fps / core.FPS) for i in range(out_n)]
            how = "floor, IC official"
        else:
            idx = core.resample_indices(n_in, fps)
            how = "nearest, rounded"
        images = images[idx]
        notes.append(f"fps map {fps:g} -> 24: {n_in} source frames -> {len(idx)} frames ({how})")
    else:
        notes.append("24 fps: no resampling" if fps else "source_fps 0: treated as 24 fps, no resampling")
    count = images.shape[0] if int(max_frames) == 0 else min(images.shape[0], int(max_frames))
    if int(min_real_frames) and count < int(min_real_frames):
        raise ValueError(f"H3 Open Regen Prepare Draft: the draft gives only {count} real frames at 24 fps; "
                         f"min_real_frames is {int(min_real_frames)} (Open-H3-Regenerate-2K-IC needs 113, about "
                         "4.7 s). Use a longer draft, or set min_real_frames to 0 for the standard recipe.")
    src_h, src_w = int(images.shape[1]), int(images.shape[2])
    width, height = core.draft_size(str(size).split()[0], src_w, src_h)
    sw, sh, x0, y0 = core.cover_crop_box(src_w, src_h, width, height)
    out = []
    for start in range(0, count, chunk):
        part = images[start:min(count, start + chunk), :, :, :3].movedim(-1, 1).to(torch.float32)
        if (sw, sh) != (src_w, src_h):
            part = F.interpolate(part, size=(sh, sw), mode="bilinear", align_corners=False,
                                 antialias=sw < src_w or sh < src_h)
        out.append(part[:, :, y0:y0 + height, x0:x0 + width].clamp(0, 1).movedim(1, -1))
    frames = torch.cat(out)
    padded = core.align_frame_count(count)
    if padded > count:
        frames = torch.cat([frames, frames[-1:].expand(padded - count, -1, -1, -1)])
    notes.append(f"{src_w}x{src_h} -> cover {sw}x{sh}, centre crop at (x {x0}, y {y0}) -> {width}x{height}")
    notes.append(f"N = {count} real frames, padded to P = {padded} (17k+5; the tail repeats the last frame)"
                 + (" - shorter than 5 frames, padded to the 5-frame minimum" if count < 5 else ""))
    return frames.contiguous(), count, width, height, "\n".join(notes)


class MiniMaxH3_OpenRegenPrepareDraft(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MiniMaxH3_OpenRegenPrepareDraft",
            has_intermediate_output=True,  # L9.290: core resends this ui when the node is served from cache
            display_name="H3 Open Regen · Prepare Draft",
            category=CATEGORY,
            description=("Resamples the draft to 24 fps, cuts it to max_frames, cover-scales (bilinear) and centre-crops "
                         "it to 768p / 576p (or keeps it on a 32 px grid) and holds the last frame up to H3's 17k+5 "
                         "frames (121 -> 124 = 37 latent steps). Wire frame_count to Save Refined Latent and Finish "
                         "Frames. " + ATTRIBUTION),
            search_aliases=["open h3 regenerate prepare draft", "regen prepare"],
            inputs=[
                io.Image.Input("images", tooltip="The draft frames [frames, H, W, 3], 0..1."),
                io.Combo.Input("size", options=SIZES, default=SIZES[0],
                               tooltip="768p for H3 drafts, 576p for drafts from other video models (as the reference). "
                                       "Portrait drafts get the portrait size. keep = each side rounded to 32 px."),
                io.Int.Input("max_frames", default=121, min=0, max=3600,
                             tooltip="Frames kept after resampling to 24 fps; 121 = 5 s (reference); 0 = all. The "
                                     "count is padded up to 17k+5 frames."),
                io.Float.Input("source_fps", default=24.0, min=0.0, max=240.0, step=0.001,
                               tooltip="Frame rate of the frames in fps (Get Video Components fps). Anything other "
                                       "than 24 is resampled to 24 fps (nearest frame). 0 = treat as 24."),
                io.Combo.Input("fps_map", options=FPS_MAPS, default=FPS_MAPS[0], optional=True,
                               tooltip="How a non-24 fps draft is resampled: nearest (rdh9616 Regenerate) or floor "
                                       "int(i * fps / 24) (the official Open-H3-Regenerate-2K-IC preprocessing)."),
                io.Int.Input("min_real_frames", default=0, min=0, max=3600, optional=True,
                             tooltip="Stop with a clear error when the draft has fewer real frames at 24 fps. IC "
                                     "recipe: 113 (with max_frames 124). 0 = no check."),
            ],
            outputs=[io.Image.Output("images", tooltip="IMAGE batch at 24 fps, cover-scaled and centre-cropped to "
                                                       "width x height, padded to 17k+5 frames by repeating the last "
                                                       "frame. Wire to the H3 VAE Encode."),
                     io.Int.Output("frame_count", tooltip="Real frames before the 17k+5 padding. Wire to Save "
                                                          "Refined Latent and Finish Frames, which drop the "
                                                          "padding."),
                     io.Int.Output("width", tooltip="Width of the prepared frames in px (multiple of 32)."),
                     io.Int.Output("height", tooltip="Height of the prepared frames in px (multiple of 32)."),
                     io.String.Output("report", tooltip="Text: fps resampling, scale / crop box, real and padded "
                                                        "frame counts.")],
        )

    @classmethod
    def execute(cls, images, size, max_frames, source_fps=24.0, fps_map=None, min_real_frames=0) -> io.NodeOutput:
        frames, count, width, height, report = prepare_draft(images, size, max_frames, source_fps,
                                                             fps_map=fps_map or FPS_MAPS[0],
                                                             min_real_frames=int(min_real_frames or 0))
        return io.NodeOutput(frames, count, width, height, report, ui={"text": [report]})


def finish_frames(images, frame_count, crop, audio=None):
    """(images, audio or None, report): keep the real frames, crop the latent-grid overhang, cut the soundtrack."""
    if not isinstance(images, torch.Tensor) or images.ndim != 4 or images.shape[0] == 0:
        raise ValueError("H3 Open Regen Finish: no frames - connect the decoded IMAGE batch")
    n_in = images.shape[0]
    if int(frame_count):
        images = images[:int(frame_count)]
    notes = [f"kept {images.shape[0]} of {n_in} frames" + (" (frame_count 0 = all)" if not int(frame_count) else "")]
    if audio is not None:
        sr = int(audio["sample_rate"])
        keep = round(images.shape[0] / core.FPS * sr)
        waveform = audio["waveform"][..., :keep]
        pad = keep - waveform.shape[-1]
        audio = {**audio, "waveform": F.pad(waveform, (0, pad))}
        notes.append(f"audio cut to {keep} samples ({keep / sr:.3f} s at {sr} Hz) from t = 0"
                     + (f", {pad} samples of silence added" if pad > 0 else ""))
    if str(crop).startswith("auto"):
        h, w = int(images.shape[1]), int(images.shape[2])
        top, left, oh, ow = core.finish_crop(h, w)
        if (oh, ow) != (h, w):
            images = images[:, top:top + oh, left:left + ow]
            notes.append(f"centre crop {w}x{h} -> {ow}x{oh}")
    return images.contiguous(), audio, "\n".join(notes)


class MiniMaxH3_OpenRegenFinish(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MiniMaxH3_OpenRegenFinish",
            has_intermediate_output=True,  # L9.290: core resends this ui when the node is served from cache
            display_name="H3 Open Regen · Finish Frames",
            category=CATEGORY,
            description=("Keeps the real frames (drops the 17k+5 padding), crops the latent-grid overhang (1088 -> 1080, "
                         "2176 -> 2160 rows; columns for portrait) and cuts the soundtrack to frame_count / 24 s from "
                         "t = 0. Then Create Video at 24 fps. " + ATTRIBUTION),
            search_aliases=["open h3 regenerate finish", "regen finish"],
            inputs=[
                io.Image.Input("images", tooltip="Decoded frames."),
                io.Int.Input("frame_count", default=121, min=0, max=100000,
                             tooltip="Real frames to keep (Prepare Draft / Load Refined Latent frame_count). 0 = all."),
                io.Combo.Input("crop", options=CROPS, default=CROPS[0],
                               tooltip="auto: centre crop of the latent-grid overhang (1088 -> 1080 px, 2176 -> "
                                       "2160 px). none: keep the computed size."),
                io.Audio.Input("audio", optional=True,
                               tooltip="Soundtrack to cut (or pad with silence) to the kept frames / 24 s."),
            ],
            outputs=[io.Image.Output("images", tooltip="The first frame_count frames, centre-cropped (auto) from "
                                                       "1088 / 2176 to 1080 / 2160 px. Wire to Create Video at "
                                                       "24 fps."),
                     io.Audio.Output("audio", tooltip="The audio input cut (or padded with silence) to the kept "
                                                      "frames / 24 s from t = 0; empty (None) when no audio is "
                                                      "connected.")],
        )

    @classmethod
    def execute(cls, images, frame_count, crop, audio=None) -> io.NodeOutput:
        out, sound, report = finish_frames(images, frame_count, crop, audio)
        return io.NodeOutput(out, sound, ui={"text": [report]})


# ====================================================================================================== latents

def upscale_latent(latent, model_name, resolution, precision="bf16", device="auto", chunk_latent_steps=0,
                   method="model", engine=None):
    """(latent of the same kind, kept W, kept H, report). `engine` = the Suite upscaler (injectable for tests)."""
    what = "H3 Open Regen Latent Upscale"
    engine = engine or LU.upscale
    video, _audio = _h3_video(latent, what)
    lat_h, lat_w = int(video.shape[-2]), int(video.shape[-1])
    if resolution not in core.RESOLUTIONS:
        raise ValueError(f"{what}: resolution must be one of {', '.join(core.RESOLUTIONS)}")
    passes, (tw, th, kw, kh) = core.upscale_passes(lat_w, lat_h, resolution)
    lines = [f"{lat_w * 16}x{lat_h * 16} px -> computed {tw}x{th} px, kept {kw}x{kh} px ({resolution}); "
             f"{len(passes)} pass{'es' if len(passes) > 1 else ''} planned"]
    in_ratio, out_ratio = lat_w / lat_h, tw / th
    if abs(in_ratio / out_ratio - 1.0) > 0.02:
        lines.append(f"note: the draft's ratio {in_ratio:.3f} differs from the target {out_ratio:.3f}: the upscaler "
                     "resizes to the exact target (upstream semantics, no aspect planner)")
    cur = latent
    for i, (w, h) in enumerate(passes, 1):
        v, _ = _h3_video(cur, what)
        if (int(v.shape[-1]) * 16, int(v.shape[-2]) * 16) == (w, h):
            lines.append(f"pass {i}: already {w}x{h}, skipped")
            continue
        cur, rep = engine(cur, method, model_name, w, h, device, precision, int(chunk_latent_steps), 4, True,
                          cells=(h // 16, w // 16))
        lines.append(f"pass {i} -> {w}x{h}:\n  " + rep.replace("\n", "\n  "))
    v, _ = _h3_video(cur, what)
    if (int(v.shape[-1]) * 16, int(v.shape[-2]) * 16) != (tw, th):
        raise ValueError(f"{what}: the upscaler returned {int(v.shape[-1]) * 16}x{int(v.shape[-2]) * 16}, expected "
                         f"{tw}x{th}")
    return cur, kw, kh, "\n".join(lines)


class MiniMaxH3_OpenRegenUpscale(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        models = LU._model_names()
        return io.Schema(
            node_id="MiniMaxH3_OpenRegenUpscale",
            has_intermediate_output=True,  # L9.290: core resends this ui when the node is served from cache
            display_name="H3 Open Regen · Latent Upscale",
            category=CATEGORY,
            description=("The trained H3 3D latent upscaler (the Suite's engine) straight onto the regenerate target "
                         "grid, no decode / re-encode. 4K runs two passes (to 2K, then 4K) so each stays within the "
                         "trained range. width / height = the KEPT size (1080 for a 1088-row grid, 2160 for 2176). "
                         + ATTRIBUTION),
            search_aliases=["open h3 regenerate upscale", "regen latent upscale"],
            inputs=[
                io.Latent.Input("latent", tooltip="Draft latent: H3 video [B, 24, T, h, w] (VAE Encode) or an H3 audio "
                                                  "+ video latent (the audio passes through)."),
                io.Combo.Input("model_name", options=models,
                               default=_pick(models, "minimax_h3_latent_upscaler_3d"),
                               tooltip="models/latent_upscale_models, .safetensors only (minimax_h3_latent_upscaler_3d_"
                                       "conv_v1_bf16 / _fp16). Pickle (.pth) files are not listed: they can run code "
                                       "when loaded."),
                io.Combo.Input("resolution", options=RESOLUTIONS, default="2k",
                               tooltip="Computed sizes 1344x768, 1920x1088, 2560x1440, 3840x2176 px (portrait swaps "
                                       "W/H); 1088 / 2176 are cropped to 1080 / 2160 by Finish Frames."),
                io.Combo.Input("precision", options=["bf16", "fp16", "fp32"], default="bf16",
                               tooltip="Reference: bf16. On the CPU it always runs fp32."),
                io.Combo.Input("device", options=["auto", "cpu"], default="auto", advanced=True,
                               tooltip="auto: ComfyUI's device (CUDA, ROCm or XPU)."),
                io.Int.Input("chunk_latent_steps", default=0, min=0, max=4096, advanced=True,
                             tooltip="0: the whole clip in one pass (reference). > 0: chunks of this many latent steps, "
                                     "cross-faded; less VRAM, approximate."),
            ],
            outputs=[io.Latent.Output("latent", tooltip="The upscaled latent, same kind as the input (video or "
                                                       "audio + video, audio passed through), on the computed "
                                                       "grid (e.g. 1920x1088 px for 2k). Wire to Refine Latent."),
                     io.Int.Output("width", tooltip="KEPT output width in px after Finish Frames' crop (e.g. "
                                                    "1920 for 2k; the latent grid itself may be larger)."),
                     io.Int.Output("height", tooltip="KEPT output height in px after Finish Frames' crop (1080 "
                                                     "for a 1088-row grid, 2160 for 2176)."),
                     io.String.Output("report", tooltip="Text: source and target sizes, planned passes and each "
                                                        "upscaler pass's own report.")],
        )

    @classmethod
    def execute(cls, latent, model_name, resolution, precision, device="auto", chunk_latent_steps=0) -> io.NodeOutput:
        out, kw, kh, report = upscale_latent(latent, model_name, resolution, precision, device, chunk_latent_steps)
        return io.NodeOutput(out, kw, kh, report, ui={"text": [report]})


def _asset_silence(silence_file, audio_t, what):
    from mmx_nodes import open_regen_ic as IC
    from mmx_utils import open_regen_ic as R
    if not silence_file or silence_file == IC.NO_FILE:
        raise ValueError(f"{what}: '{AUDIO_ASSET}' needs silence_audio_124f.safetensors in models/{IC.ASSET_FOLDER} "
                         "(pick it in silence_file), or choose 'silence (reference)' with the H3 audio VAE")
    path = IC._full_path(IC.ASSET_FOLDER, silence_file, what)
    sd = IC._load_safetensors(path, what)
    if "audio_latents" not in sd:
        raise ValueError(f"{what}: {silence_file!r} has no 'audio_latents' tensor - not the IC silence asset")
    sound = R.asset_to_latent(sd["audio_latents"])
    if sound.shape[-1] != audio_t:
        raise ValueError(f"{what}: the silence asset holds {sound.shape[-1]} audio latents (124 frames); this clip "
                         f"needs {audio_t}. Use 'silence (reference)' with the H3 audio VAE for other lengths.")
    return sound


def _silence_files():
    try:
        from mmx_nodes import open_regen_ic as IC
        return IC._asset_names() or [IC.NO_FILE]
    except Exception:  # noqa: BLE001 - the list is only a convenience; the node still loads
        return ["(none)"]


def _silence_default():
    files = _silence_files()
    return _pick(files, "silence_audio_124f")


def refine_latent(latent, audio_mode, audio_vae=None, audio=None, reference_latent=None, silence_file=None):
    """(AV latent with video regenerate / audio pinned, frame_count, report).

    reference_latent (core MiniMax H3 Reference to Video LATENT, the draft as <Video 1>) = the IC recipe: the target
    keeps that latent's other keys and must match its video shape (slmonker OpenH3ICComposeLatent semantics)."""
    what = "H3 Open Regen Refine Latent"
    video, latent_audio = _h3_video(latent, what)
    ref_parts = None
    if reference_latent is not None:
        ref = reference_latent.get("samples") if isinstance(reference_latent, dict) else None
        if not getattr(ref, "is_nested", False) or len(ref.tensors) != 2:
            raise ValueError(f"{what}: reference_latent must be the LATENT output of core 'MiniMax H3 Reference to "
                             "Video' (an audio + video latent)")
        ref_parts = ref.tensors
        if tuple(video.shape) != tuple(ref_parts[0].shape):
            raise ValueError(f"{what}: the upscaled video latent {tuple(video.shape)} does not match the Reference to "
                             f"Video target {tuple(ref_parts[0].shape)}. Set Reference to Video width / height to the "
                             "upscale target (e.g. 2560 x 1440) and its length to Prepare Draft's padded frame count "
                             "(e.g. 124).")
        video = video.to(dtype=ref_parts[0].dtype, device=ref_parts[0].device)
    t = int(video.shape[2])
    try:
        frames = core.frames_for_latent_t(t)
    except ValueError:
        raise ValueError(f"{what}: the video latent has {t} latent steps; an H3 clip has 5k+2 (17k+5 frames). Use "
                         "Prepare Draft before VAE Encode.") from None
    audio_t = core.audio_latent_t(frames)
    notes = [f"T = {t} latent steps -> {frames} frames, audio {audio_t} latent steps (40 Hz)"]
    if audio_mode == AUDIO_MODES[2]:
        if latent_audio is None:
            raise ValueError(f"{what}: 'keep latent audio' needs an H3 audio + video latent; this one is video only. "
                             "Pick 'silence (reference)' and connect the H3 audio VAE.")
        sound = latent_audio
        notes.append("audio: the latent's own audio stream")
    elif audio_mode in AUDIO_MODES[:2]:
        if audio_vae is None:
            raise ValueError(f"{what}: '{audio_mode}' needs the H3 audio VAE connected to audio_vae")
        rate = int(getattr(audio_vae, "audio_sample_rate", 32000))
        length = round(frames / core.FPS * rate)     # reference: 124 frames -> 5.17 s -> 207 latents
        if audio_mode == AUDIO_MODES[0]:
            waveform = torch.zeros(1, 2, length)
            notes.append(f"audio: {length / rate:.3f} s of silence at {rate} Hz, encoded (reference)")
        else:
            if audio is None:
                raise ValueError(f"{what}: 'draft audio (AUDIO input)' needs the draft's AUDIO connected to audio")
            wav = audio["waveform"]
            if wav.ndim == 2:
                wav = wav.unsqueeze(0)
            if wav.shape[0] > 1:
                notes.append(f"audio batch of {wav.shape[0]}: item 0 used")
            waveform = wav[:1].to(torch.float32)
            sr = int(audio["sample_rate"])
            if sr != rate:
                import comfy.audio
                waveform = comfy.audio.resample(waveform, sr, rate)
            waveform = waveform.expand(-1, 2, -1) if waveform.shape[1] == 1 else waveform[:, :2]
            got = waveform.shape[-1]
            waveform = F.pad(waveform, (0, max(0, length - got)))[..., :length]
            notes.append(f"audio: draft soundtrack {sr} -> {rate} Hz, {'padded' if got < length else 'cut'} to "
                         f"{length / rate:.3f} s, encoded")
        sound = audio_vae.encode(waveform.movedim(1, -1))
    elif audio_mode == AUDIO_ASSET:
        sound = _asset_silence(silence_file, audio_t, what)
        notes.append(f"audio: IC silence asset {silence_file}")
    else:
        raise ValueError(f"{what}: audio_mode must be one of {', '.join(AUDIO_MODES + [AUDIO_ASSET])}")
    if isinstance(sound, dict):
        sound = sound.get("samples")
    if ref_parts is not None:
        from mmx_utils import open_regen_ic as R
        sound = sound.to(device=ref_parts[1].device, dtype=ref_parts[1].dtype)
        if sound.ndim == 3:
            sound = sound.unsqueeze(0)
        sound = R.fit_audio_columns(sound, ref_parts[1])
        out = R.pinned_av_latent(reference_latent, latent, video, sound, True, _nested, "open_h3_ic",
                                 {"audio": audio_mode, "pinned": True, "via": "H3 Open Regen Refine Latent"})
        notes.append(f"IC target: Reference to Video latent kept (keys {', '.join(sorted(out))}); noise mask: "
                     "video 1 (regenerated), audio 0 (held clean)")
        return out, frames, "\n".join(notes)
    sound = sound.to(video.device, torch.float32)
    if sound.ndim == 3:
        sound = sound.unsqueeze(0)
    if sound.shape[-1] < audio_t:
        sound = torch.cat([sound, sound[..., -1:].expand(*sound.shape[:-1], audio_t - sound.shape[-1])], dim=-1)
    sound = sound[..., :audio_t].contiguous()
    result = {k: v for k, v in latent.items() if k not in ("samples", "noise_mask")}
    result["samples"] = _nested((video, sound))
    # the layout MiniMaxH3_SetAVNoiseMask writes: video [1,1,T,h,w] = 1 (re-noised), audio [1,1,2,ticks] = 0 (pinned)
    result["noise_mask"] = _nested((torch.ones((1, 1) + tuple(video.shape[2:]), device=video.device),
                                    torch.zeros((1, 1) + tuple(sound.shape[2:]), device=video.device)))
    notes.append("noise mask: video 1 (regenerated), audio 0 (held clean)")
    return result, frames, "\n".join(notes)


class MiniMaxH3_OpenRegenRefineLatent(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MiniMaxH3_OpenRegenRefineLatent",
            has_intermediate_output=True,  # L9.290: core resends this ui when the node is served from cache
            display_name="H3 Open Regen · Refine Latent",
            category=CATEGORY,
            description=("H3 audio + video latent for the refine: the upscaled video is re-noised by the sampler (start "
                         "sigma 0.909 or 0.8), the audio stream is masked out and stays clean. The output soundtrack is "
                         "not affected - give Finish Frames the draft's audio. " + ATTRIBUTION),
            search_aliases=["open h3 regenerate refine latent", "regen refine"],
            inputs=[
                io.Latent.Input("latent", tooltip="Upscaled latent (video or AV); T must be 5k+2 latent steps."),
                io.Combo.Input("audio_mode", options=AUDIO_MODES + [AUDIO_ASSET], default=AUDIO_MODES[0],
                               tooltip="Audio stream held clean during the refine. Reference: encoded silence. "
                                       "The IC silence asset (models/h3_assets, 124 frames only) is the upstream IC "
                                       "file; encoded silence works for every length."),
                io.Vae.Input("audio_vae", optional=True,
                             tooltip="The H3 audio VAE (needed for silence and draft audio)."),
                io.Audio.Input("audio", optional=True, tooltip="Draft soundtrack, for 'draft audio (AUDIO input)'."),
                io.Latent.Input("reference_latent", optional=True,
                                tooltip="IC recipe: the LATENT output of core 'MiniMax H3 Reference to Video' with the "
                                        "prepared draft on ref_video_0 (<Video 1>) and the IC prompt. Its width / "
                                        "height / length must equal the upscale target. Empty = standard Regenerate."),
                io.Combo.Input("silence_file", options=_silence_files(), default=_silence_default(), optional=True,
                               tooltip="File in models/h3_assets for audio_mode = silence asset file "
                                       "(silence_audio_124f.safetensors)."),
            ],
            outputs=[io.Latent.Output("latent", tooltip="H3 audio + video latent with a noise mask: video 1 "
                                                       "(regenerated by the sampler), audio 0 (held clean). Wire "
                                                       "to the refine sampler."),
                     io.Int.Output("frame_count", tooltip="Pixel frames of the 5k+2 latent (17k+5, padding "
                                                          "included), not the draft's real frame count."),
                     io.String.Output("report", tooltip="Text: latent steps -> frames, audio length and source, "
                                                        "noise-mask layout.")],
        )

    @classmethod
    def execute(cls, latent, audio_mode, audio_vae=None, audio=None, reference_latent=None,
                silence_file=None) -> io.NodeOutput:
        out, frames, report = refine_latent(latent, audio_mode, audio_vae, audio, reference_latent, silence_file)
        return io.NodeOutput(out, frames, report, ui={"text": [report]})


# ====================================================================================================== fast VAE decode

_FAST_VAE: dict = {}


def _decoder_class():
    from mmx_utils.fast_h3_vae.lean import LeanNonCausalViTDecoder

    class InterruptibleLeanDecoder(LeanNonCausalViTDecoder):
        """NVIDIA's lean decoder, unchanged, with ComfyUI's cancel checked before every layer."""

        def _layer(self, li, blk, x, g):
            _mm().throw_exception_if_processing_interrupted()
            return super()._layer(li, blk, x, g)

    return InterruptibleLeanDecoder


def build_fast_vae(latents_mean, latents_std):
    from mmx_utils.fast_h3_vae.decoder import KEEP
    # released H3 VAE decoder config (MiniMax-H3 vae/config.json): 32 heads x 64, RoPE theta 100 on 3/4 of each head
    return _decoder_class()(latents_mean, latents_std, KEEP[24], time_norm=5, heads=32, dim_head=64, rope_theta=100.0,
                            rope_dim_ratio=0.75, eps=1e-5, latent_channels=24)


def load_fast_vae(path, builder=None):
    key = (path, os.path.getmtime(path))
    if key not in _FAST_VAE:
        _FAST_VAE.clear()
        gc.collect()
        import comfy.utils
        if builder is None:
            from comfy.ldm.minimax.vae import LATENTS_MEAN, LATENTS_STD
            model = build_fast_vae(LATENTS_MEAN, LATENTS_STD)
        else:
            model = builder()
        sd = comfy.utils.load_torch_file(path, safe_load=True)
        try:
            model.load_state_dict(sd, strict=True)
        except RuntimeError as exc:
            raise ValueError(f"{os.path.basename(path)} is not NVIDIA's fast H3 VAE decoder (24 layers, non-causal): "
                             f"pick fast_h3_vae_noncausal_24l.safetensors (models/vae). Load error: "
                             f"{str(exc).splitlines()[0]}") from None
        finally:
            del sd
        model.eval().requires_grad_(False)
        _FAST_VAE[key] = model
    return _FAST_VAE[key]


def _native_decode(vae, video, why):
    LOG.warning("H3 Open Regen fast VAE decode: %s - decoding with the connected H3 VAE instead", why)
    images = vae.decode(video)
    return images.reshape(-1, *images.shape[-3:]) if images.ndim == 5 else images


def fast_decode(samples, model, keep_loaded=False, vae=None, device=None):
    """(images [B*F, 16H, 16W, 3], report)."""
    what = "H3 Open Regen Fast VAE Decode"
    mm = _mm()
    z, _ = _h3_video(samples, what)
    device = device or mm.get_torch_device()
    B, _, T, lat_h, lat_w = z.shape
    frames = core.decoded_frames(T)
    weights = sum(p.numel() * p.element_size() for p in model.parameters())
    # x (fp32) + q / k / v / attention output (bf16) for the whole clip + one window's / chunk's temporaries
    need = weights + T * lat_h * lat_w * 26 * 1024 + 3 * 2 ** 30
    est = f"VRAM estimate ~{need / 2 ** 30:.1f} GB ({T} x {lat_h} x {lat_w} latent, {weights / 2 ** 30:.1f} GB weights)"
    if device.type == "cuda":
        mm.free_memory(need, device)
        free = mm.get_free_memory(device)
        if vae is not None and need > free:
            images = _native_decode(vae, z, f"needs ~{need / 2 ** 30:.0f} GB, {free / 2 ** 30:.0f} GB free")
            return images, f"fallback: core VAE decode (fast decoder needs more VRAM than is free); {est}"
    out = torch.empty(B * frames, lat_h * 16, lat_w * 16, 3, device=mm.intermediate_device())
    mean = torch.tensor((.485, .456, .406), device=device).view(1, 3, 1, 1, 1)
    std = torch.tensor((.229, .224, .225), device=device).view(1, 3, 1, 1, 1)
    started = time.perf_counter()
    oom = False
    try:
        model.to(device)
        with torch.inference_mode(), torch.autocast(device.type, dtype=torch.bfloat16):
            for b in range(B):
                def sink(f, pixels, base=b * frames):         # pixels [1, 3, n, H, W], ImageNet-normalized
                    mm.throw_exception_if_processing_interrupted()
                    part = (pixels.float() * std + mean).clamp_(0, 1)[0].permute(1, 2, 3, 0)
                    out[base + f:base + f + part.shape[0]].copy_(part)
                model(z[b:b + 1].to(device, torch.float32).transpose(1, 2), sink=sink)
    except torch.OutOfMemoryError:
        if vae is None:
            raise RuntimeError(f"{what}: the fast H3 VAE decoder ran out of VRAM ({tuple(z.shape)} latent, needs ~"
                               f"{need / 2 ** 30:.0f} GB): connect the H3 video VAE to this node's `vae` input to fall "
                               "back to the core decoder automatically") from None
        oom = True
    finally:
        if not keep_loaded:
            model.to(mm.unet_offload_device())
        mm.soft_empty_cache()
    if oom:
        out = None
        gc.collect()
        return _native_decode(vae, z, "out of VRAM"), f"fallback: core VAE decode after an out-of-VRAM error; {est}"
    report = (f"fast decoder on {device}: {tuple(z.shape)} latent -> {out.shape[0]} frames "
              f"{lat_w * 16}x{lat_h * 16} px in {time.perf_counter() - started:.1f} s; {est}"
              + ("; decoder kept loaded" if keep_loaded else "; decoder offloaded"))
    return out, report


class MiniMaxH3_OpenRegenFastVAEDecode(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        vaes = _files("vae")
        opts = vaes or ["(put fast_h3_vae_noncausal_24l.safetensors in models/vae)"]
        return io.Schema(
            node_id="MiniMaxH3_OpenRegenFastVAEDecode",
            has_intermediate_output=True,  # L9.290: core resends this ui when the node is served from cache
            display_name="H3 Open Regen · Fast VAE Decode",
            category=CATEGORY,
            description=("NVIDIA's fast H3 VAE decoder (StreamRender-H3): the released decoder pruned to 24 layers and "
                         "distilled, whole 5-latent blocks with window attention, no tiling (plain PyTorch SDPA). "
                         "Slightly lossy (the quality figure is not verified here). VRAM grows with the clip (~26 KB per "
                         "latent token + 6.5 GB weights). Connect the H3 video VAE to `vae` as the fallback. "
                         + ATTRIBUTION),
            search_aliases=["open h3 regenerate fast vae", "fast h3 vae decode", "streamrender"],
            inputs=[
                io.Latent.Input("samples", tooltip="H3 video or AV latent (the audio stream is ignored)."),
                io.Combo.Input("vae_name", options=opts, default=_pick(opts, D5_FAST_VAE_FILTER),
                               tooltip="models/vae/fast_h3_vae_noncausal_24l.safetensors (6.5 GB). Loaded strictly: "
                                       "any other file is refused."),
                io.Boolean.Input("keep_loaded", default=False,
                                 tooltip="Keep the decoder on the GPU after decoding (6.5 GB). Off = offload and "
                                         "empty the cache."),
                io.Vae.Input("vae", optional=True,
                             tooltip="H3 video VAE: used instead when the clip does not fit the fast decoder or it "
                                     "runs out of VRAM."),
            ],
            outputs=[io.Image.Output("images", tooltip="Decoded IMAGE batch, one frame per decoded pixel frame "
                                                       "(all 17k+5, padding included), 16x the latent size. Wire "
                                                       "to Finish Frames."),
                     io.String.Output("report", tooltip="Text: device, latent shape, frames and size, time, VRAM "
                                                        "estimate, and whether the core VAE fallback was used.")],
        )

    @classmethod
    def execute(cls, samples, vae_name, keep_loaded=False, vae=None) -> io.NodeOutput:
        model = load_fast_vae(_full_path("vae", vae_name))
        images, report = fast_decode(samples, model, bool(keep_loaded), vae)
        LOG.info("H3 Open Regen fast VAE decode: %s", report)
        return io.NodeOutput(images, report, ui={"text": [report]})


# ====================================================================================================== latent on disk

def latent_dir():
    import folder_paths
    return os.path.join(folder_paths.get_output_directory(), LATENT_DIR)


def saved_latents():
    folder = latent_dir()
    if not os.path.isdir(folder):
        return []
    names = [n for n in os.listdir(folder) if n.endswith(".safetensors")]
    return sorted(names, key=lambda n: os.path.getmtime(os.path.join(folder, n)), reverse=True)


def resolve_latent(choice):
    if choice == NEWEST:
        files = saved_latents()
        return files[0] if files else ""
    return os.path.basename(str(choice))


def _atomic_save(tensors, path, meta):
    from safetensors.torch import save_file
    tmp = path + ".tmp"
    try:
        save_file(tensors, tmp, metadata=meta)
        try:
            os.replace(tmp, path)
        except OSError:
            gc.collect()
            os.replace(tmp, path)
    finally:
        try:
            os.remove(tmp)
        except OSError:
            pass


def save_refined(samples, frame_count, filename_prefix, audio=None):
    """Write <prefix>_<NNNNN>.safetensors; returns its name."""
    what = "H3 Open Regen Save Refined Latent"
    video, sound = _h3_video(samples, what)
    tensors = {"video": video.detach().to("cpu", torch.float32).contiguous()}
    if sound is not None:
        tensors["audio_latent"] = sound.detach().to("cpu", torch.float32).contiguous()
    meta = {"format": LATENT_FORMAT, "frame_count": str(int(frame_count))}
    if audio is not None:
        tensors["waveform"] = audio["waveform"].detach().to("cpu", torch.float32).contiguous()
        meta["sample_rate"] = str(int(audio["sample_rate"]))
    folder = latent_dir()
    os.makedirs(folder, exist_ok=True)
    prefix = os.path.basename(str(filename_prefix).strip()) or "open_h3_regen"
    pat = re.compile(re.escape(prefix) + r"_(\d+)\.safetensors$")
    index = 1 + max([int(m.group(1)) for m in (pat.match(n) for n in os.listdir(folder)) if m] + [0])
    name = f"{prefix}_{index:05d}.safetensors"
    _atomic_save(tensors, os.path.join(folder, name), meta)
    LOG.info("H3 Open Regen refined latent saved: output/%s/%s", LATENT_DIR, name)
    return name


def load_refined(choice):
    """(latent dict, frame_count, audio, file name)."""
    from safetensors import safe_open
    name = resolve_latent(choice)
    if not name:
        raise FileNotFoundError(f"H3 Open Regen Load Refined Latent: no saved latents in output/{LATENT_DIR}/ yet")
    path = os.path.join(latent_dir(), name)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"H3 Open Regen Load Refined Latent: no saved latent output/{LATENT_DIR}/{name}")
    with safe_open(path, framework="pt", device="cpu") as f:
        meta = f.metadata() or {}
        if meta.get("format") != LATENT_FORMAT:
            raise ValueError(f"H3 Open Regen Load Refined Latent: {name} is not a saved Open-H3 refined latent "
                             f"(format {meta.get('format')!r}, expected {LATENT_FORMAT})")
        tensors = {k: f.get_tensor(k).clone() for k in f.keys()}
    if "video" not in tensors:
        raise ValueError(f"H3 Open Regen Load Refined Latent: {name} has no 'video' tensor")
    video = tensors["video"]
    samples = _nested((video, tensors["audio_latent"])) if "audio_latent" in tensors else video
    frame_count = int(meta.get("frame_count", "0") or 0)
    if "waveform" in tensors:
        audio = {"waveform": tensors["waveform"], "sample_rate": int(meta["sample_rate"])}
    else:   # silent track of the right length, so Create Video always gets audio (as upstream)
        t = int(video.shape[2])
        frames = frame_count or (5 if t <= 2 else (t - 2) // 5 * 17 + 5)
        audio = {"waveform": torch.zeros(1, 2, round(frames / core.FPS * 48000)), "sample_rate": 48000}
    return {"samples": samples}, frame_count, audio, name


class MiniMaxH3_OpenRegenSaveLatent(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MiniMaxH3_OpenRegenSaveLatent",
            display_name="H3 Open Regen · Save Refined Latent",
            category=CATEGORY,
            description=(f"Saves the refined latent (+ frame count and soundtrack) to output/{LATENT_DIR}/ and passes it "
                         "on. Put it between the sampler and the decoder: the file is written before decoding starts, "
                         "so a failed decode never costs the refine - run Load Refined Latent -> Fast VAE Decode -> "
                         "Finish Frames instead. Same format as upstream. " + ATTRIBUTION),
            search_aliases=["open h3 save refined latent", "regen save latent"],
            inputs=[
                io.Latent.Input("samples", tooltip="The sampler's output (video or AV latent), before decoding."),
                io.Int.Input("frame_count", default=0, min=0, max=100000,
                             tooltip="Real frames, stored with the latent (Prepare Draft frame_count); 0 = all."),
                io.String.Input("filename_prefix", default="open_h3_regen",
                                tooltip=f"File name under output/{LATENT_DIR}/; a 5-digit counter is appended."),
                io.Audio.Input("audio", optional=True,
                               tooltip="Soundtrack for the final video (stored with the latent)."),
            ],
            outputs=[io.Latent.Output("samples", tooltip="The input latent, unchanged (passed on after the file "
                                                         "is written). Wire to the decoder.")],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, samples, frame_count, filename_prefix, audio=None) -> io.NodeOutput:
        name = save_refined(samples, frame_count, filename_prefix, audio)
        return io.NodeOutput(samples, ui={"text": [f"saved output/{LATENT_DIR}/{name}"]})


class MiniMaxH3_OpenRegenLoadLatent(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        try:
            files = saved_latents()
        except Exception:  # noqa: BLE001
            files = []
        return io.Schema(
            node_id="MiniMaxH3_OpenRegenLoadLatent",
            has_intermediate_output=True,  # L9.290: core resends this ui when the node is served from cache
            display_name="H3 Open Regen · Load Refined Latent",
            category=CATEGORY,
            description=(f"Loads a latent saved by Save Refined Latent (or by upstream Open-H3 Regenerate) from "
                         f"output/{LATENT_DIR}/, with its frame count and soundtrack (silence when none was stored). "
                         + ATTRIBUTION),
            search_aliases=["open h3 load refined latent", "regen load latent"],
            inputs=[
                io.Combo.Input("latent_file", options=[NEWEST] + files, default=NEWEST,
                               tooltip=f"Files in output/{LATENT_DIR}/, newest first. '{NEWEST}' picks the latest file "
                                       "when the workflow runs."),
            ],
            outputs=[io.Latent.Output("samples", tooltip="The saved latent: H3 video, or audio + video when an "
                                                         "audio latent was stored. Wire to Fast VAE Decode."),
                     io.Int.Output("frame_count", tooltip="Real frame count stored with the latent (0 if it was "
                                                          "saved with 0 = all). Wire to Finish Frames."),
                     io.Audio.Output("audio", tooltip="The stored soundtrack, or silence (48 kHz stereo) of the "
                                                      "clip's length when none was stored.")],
        )

    @classmethod
    def fingerprint_inputs(cls, latent_file):
        try:
            name = resolve_latent(latent_file)
            path = os.path.join(latent_dir(), name) if name else ""
            if path and os.path.isfile(path):
                return f"{name}:{os.stat(path).st_mtime_ns}"
        except Exception:  # noqa: BLE001
            pass
        return f"missing:{latent_file}"

    @classmethod
    def execute(cls, latent_file) -> io.NodeOutput:
        latent, frames, audio, name = load_refined(latent_file)
        return io.NodeOutput(latent, frames, audio, ui={"text": [f"loaded output/{LATENT_DIR}/{name}"]})


__all__ = [
    "MiniMaxH3_OpenRegenLoRA",
    "MiniMaxH3_OpenRegenSigmas",
    "MiniMaxH3_OpenRegenPrepareDraft",
    "MiniMaxH3_OpenRegenUpscale",
    "MiniMaxH3_OpenRegenRefineLatent",
    "MiniMaxH3_OpenRegenFastVAEDecode",
    "MiniMaxH3_OpenRegenFinish",
    "MiniMaxH3_OpenRegenSaveLatent",
    "MiniMaxH3_OpenRegenLoadLatent",
]
