# PORTED FROM: https://github.com/slmonker/ComfyUI-OpenH3Regenerate2KIC@a85bd96 nodes.py
#              (TASK, SIGMAS, _cover_resize, _fit_audio, OpenH3ICDraft.prepare, OpenH3ICSigmas.build,
#              OpenH3ICComposeLatent._fit_silence) and tools/convert_ic_lora.py (Diffusers -> ComfyUI key layout).
#              Apache-2.0, author slmonker.
# Modified by Code2Collapse, 2026-11: V3 schema, run-time silence option, asset length check, LoRA chain node,
# appended outputs. Distributed as part of a GPL-3.0 work.
"""Open-H3 Regenerate 2K IC: pure helpers (torch only, no ComfyUI import).

IC sigma grids, the verbatim TASK prompt, the floor fps index map, draft validation, cover / stretch resize,
draft-audio fitting, silence-latent fitting, the pinned-audio AV latent and the Diffusers->ComfyUI IC LoRA plan
(q|k|v row slices of the fused qkv_proj, SwiGLU fc1 B-half swap; the same maths as upstream's offline converter,
applied at load time instead).
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from mmx_utils.open_regen_core import comfy_target, swap_swiglu_halves

TASK = (
    "Enhance <Video 1> into a high-quality version of the same video. "
    "Preserve the scene, objects, camera motion, timing, and composition of every frame. "
    "Restore fine details and remove degradation. Do not continue or change the scene."
)
PROMPT_JOIN = "\n\nScene description: "

SIGMAS = {
    "up3": [0.834942, 0.696852, 0.468753, 0.0],
    "up4": [0.923077, 0.834942, 0.696852, 0.468753, 0.0],
    "hf8": [1.0, 0.99391, 0.984288, 0.966064, 0.923077, 0.834942, 0.696852, 0.468753, 0.0],
}

FPS = 24
MIN_REAL_FRAMES = 113
FRAME_MIN, FRAME_MAX = 124, 362
SILENT_RATE = 32000
ASSET_FRAMES = 124            # the upstream silence asset encodes exactly 124 frames (207 audio latents)
RESIZE_CHUNK = 16             # frames resized per interpolate call (same result, lower peak memory)
IC_MODULES = 312              # PEFT linears in the official IC LoRA: 52 blocks x 6


# ----------------------------------------------------------------------------------------------- prompt / sigmas

def build_prompt(caption) -> str:
    return TASK + PROMPT_JOIN + (caption or "").strip()


def build_sigmas(sampler: str, custom: str = "") -> tuple[list[float], dict]:
    """(values, report). ``custom`` (comma list) overrides ``sampler``; it must end at 0, never increase and stay
    in [0, 1]. Upstream messages kept."""
    custom = (custom or "").strip()
    if custom:
        values = []
        for part in custom.split(","):
            part = part.strip()
            if not part:
                continue
            try:
                values.append(float(part))
            except ValueError:
                raise ValueError(f"custom_sigmas: {part!r} is not a number") from None
        if len(values) < 2:
            raise ValueError("custom_sigmas needs at least two values (a start sigma and the final 0.0)")
        if any(not math.isfinite(v) for v in values):
            raise ValueError("custom_sigmas values must be finite")
        if abs(values[-1]) > 1e-12:
            raise ValueError("custom_sigmas must end in 0.0")
        name = "custom"
    else:
        if sampler not in SIGMAS:
            raise ValueError("Unknown IC sampler: %s" % sampler)
        values = list(SIGMAS[sampler])
        name = sampler
    if any(b > a + 1e-12 for a, b in zip(values, values[1:])):
        raise ValueError("custom_sigmas must be non-increasing")
    if any(v < 0.0 or v > 1.0 for v in values):
        raise ValueError("custom_sigmas values must be in [0, 1]")
    return values, {"sampler": name, "sigmas": values, "steps": max(0, len(values) - 1)}


# ----------------------------------------------------------------------------------------------- draft

def validate_draft(source_count: int, width: int, height: int, frame_count: int, output_fps: int) -> None:
    if source_count < 1:
        raise ValueError("The draft video contains no frames")
    if int(width) % 32 or int(height) % 32 or int(width) < 32 or int(height) < 32:
        raise ValueError("target_width and target_height must be multiples of 32")
    if int(output_fps) != FPS:
        raise ValueError("Open-H3-Regenerate-2K-IC sampling is fixed at 24 fps; set output_fps=24")
    if (int(frame_count) - 5) % 17:
        raise ValueError("frame_count must be a MiniMax-H3 valid length: 5 + 17*k (124, 141, ...)")
    if not FRAME_MIN <= int(frame_count) <= FRAME_MAX:
        raise ValueError(f"frame_count must be in {FRAME_MIN}..{FRAME_MAX} (got {int(frame_count)})")


def floor_index_map(source_count: int, source_fps: float, frame_count: int,
                    output_fps: int = FPS) -> tuple[list[int], int]:
    """IC resampler: ``wanted[i] = int(i * source_fps / 24)`` (FLOOR, unlike STD's round). Returns (indices clamped
    to the last source frame, real) where ``real`` counts the indices that fall inside the source."""
    if source_fps <= 0:
        raise ValueError("source_fps must be > 0")
    wanted = [int(i * float(source_fps) / float(output_fps)) for i in range(int(frame_count))]
    real = sum(1 for idx in wanted if idx < source_count)
    if real < MIN_REAL_FRAMES:
        raise ValueError(
            f"The draft yields only {real} real frames at 24 fps. "
            f"Open-H3-Regenerate-2K-IC needs at least {MIN_REAL_FRAMES} (~4.7 s)."
        )
    return [min(idx, source_count - 1) for idx in wanted], real


def _rgb_nchw(frames: torch.Tensor) -> torch.Tensor:
    if frames.ndim != 4 or frames.shape[-1] < 3:
        raise ValueError("OpenH3ICDraft expects IMAGE [T,H,W,C]")
    return frames[..., :3].movedim(-1, 1).float()


def cover_resize(frames: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """[T,H,W,C] -> [T,height,width,3]: scale to cover, centre crop, clamp 0..1."""
    out = []
    for i in range(0, frames.shape[0], RESIZE_CHUNK):
        x = _rgb_nchw(frames[i:i + RESIZE_CHUNK])
        h, w = x.shape[-2:]
        scale = max(height / h, width / w)
        sh, sw = max(height, round(h * scale)), max(width, round(w * scale))
        if (sh, sw) != (h, w):
            x = F.interpolate(x, size=(sh, sw), mode="bilinear", align_corners=False, antialias=scale < 1.0)
        top, left = (sh - height) // 2, (sw - width) // 2
        out.append(x[..., top:top + height, left:left + width].movedim(1, -1).clamp(0.0, 1.0))
    return torch.cat(out, 0).contiguous()


def stretch_resize(frames: torch.Tensor, height: int, width: int) -> torch.Tensor:
    out = []
    for i in range(0, frames.shape[0], RESIZE_CHUNK):
        x = _rgb_nchw(frames[i:i + RESIZE_CHUNK])
        x = F.interpolate(x, size=(int(height), int(width)), mode="bilinear", align_corners=False, antialias=True)
        out.append(x.movedim(1, -1).clamp(0.0, 1.0))
    return torch.cat(out, 0).contiguous()


def _waveform(audio) -> torch.Tensor:
    if not isinstance(audio, dict) or "waveform" not in audio or "sample_rate" not in audio:
        raise ValueError("AUDIO must be a dict with 'waveform' and 'sample_rate'")
    waveform = audio["waveform"]
    if not isinstance(waveform, torch.Tensor) or waveform.ndim != 3:
        raise ValueError("AUDIO waveform must be [B,C,L]")
    waveform = waveform.float().clone()
    if waveform.shape[1] == 1:
        waveform = waveform.repeat(1, 2, 1)
    return waveform


def fit_audio(audio, sample_count: int) -> dict:
    """None -> silent stereo at 32 kHz; else mono->stereo, zero-pad or cut to ``sample_count`` at the SOURCE rate."""
    if audio is None:
        return {"waveform": torch.zeros((1, 2, int(sample_count)), dtype=torch.float32), "sample_rate": SILENT_RATE}
    waveform = _waveform(audio)
    if waveform.shape[-1] < sample_count:
        pad = waveform.new_zeros((waveform.shape[0], waveform.shape[1], sample_count - waveform.shape[-1]))
        waveform = torch.cat((waveform, pad), dim=-1)
    elif waveform.shape[-1] > sample_count:
        waveform = waveform[..., :sample_count]
    return {"waveform": waveform, "sample_rate": int(audio["sample_rate"])}


def prepare_audio(mode: str, audio, frames: int, output_fps: int = FPS) -> dict:
    if mode == "silent" or audio is None:
        return fit_audio(None, int(round(frames / float(output_fps) * SILENT_RATE)))
    if mode == "passthrough":
        return {"waveform": _waveform(audio), "sample_rate": int(audio["sample_rate"])}
    if mode == "trim_pad":
        return fit_audio(audio, int(round(frames / float(output_fps) * int(audio["sample_rate"]))))
    raise ValueError(f"Unknown audio_mode {mode!r}")


# ----------------------------------------------------------------------------------------------- silence / latent

def silence_sample_count(frames: int, sample_rate: int) -> int:
    return int(round(int(frames) / float(FPS) * int(sample_rate)))


def asset_to_latent(raw: torch.Tensor) -> torch.Tensor:
    """Upstream asset ``audio_latents`` [2, 32, T] -> [1, 32, 2, T]."""
    if not isinstance(raw, torch.Tensor) or raw.ndim != 3 or raw.shape[0] != 2 or raw.shape[1] != 32:
        shape = tuple(raw.shape) if isinstance(raw, torch.Tensor) else type(raw).__name__
        raise ValueError(f"silence asset 'audio_latents' must be [2, 32, T], got {shape}")
    return raw.permute(1, 0, 2).unsqueeze(0)


def fit_audio_columns(audio: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Trim, or repeat the last column, so ``audio`` [B,32,2,T] matches ``target``'s length. Batch is expanded
    from 1. Never zero-pads (a zero latent is not encoded silence)."""
    if audio.ndim != 4 or tuple(audio.shape[1:3]) != tuple(target.shape[1:3]):
        raise ValueError(f"Silence latent {tuple(audio.shape)} does not match target {tuple(target.shape)}")
    if audio.shape[0] != target.shape[0]:
        if audio.shape[0] != 1:
            raise ValueError(f"Silence latent batch {audio.shape[0]} does not match target batch {target.shape[0]}")
        audio = audio.expand(target.shape[0], -1, -1, -1)
    t = target.shape[-1]
    if audio.shape[-1] > t:
        audio = audio[..., :t]
    elif audio.shape[-1] < t:
        audio = torch.cat((audio, audio[..., -1:].expand(*audio.shape[:-1], t - audio.shape[-1])), dim=-1)
    return audio.contiguous()


def pinned_av_latent(base: dict, extra: dict, video: torch.Tensor, audio: torch.Tensor, pin: bool,
                     nested_factory, tag_key: str | None = None, tag=None) -> dict:
    """Copy of ``base`` updated with the non-``samples`` keys of ``extra``; samples = (video, audio); noise mask
    (ones video, zeros audio) when pinned, removed otherwise."""
    out = dict(base)
    out.update({k: v for k, v in (extra or {}).items() if k != "samples"})
    out["samples"] = nested_factory((video, audio))
    if pin:
        out["noise_mask"] = nested_factory((torch.ones_like(video), torch.zeros_like(audio)))
    else:
        out.pop("noise_mask", None)
    if tag_key:
        out[tag_key] = tag
    return out


# ----------------------------------------------------------------------------------------------- LoRA chain

SLOTS = ("speed", "sharp", "ic")
SLOT_NEEDLES = {
    "speed": ("hyperflow", "dmad", "turbo", "lightx2v", "step"),
    "sharp": ("minimax_h3_lms", "lms_", "sharp"),
    "ic": ("open_regenerate_2k_ic", "open_h3_regenerate_2k_ic", "regenerate_2k_ic"),
}


def pick_default(names, needles, fallback="none") -> str:
    for needle in needles:
        for n in names:
            if needle in n.lower():
                return n
    return fallback


def slot_role(name: str) -> str | None:
    low = (name or "").lower()
    for slot in ("ic", "sharp", "speed"):     # IC first: its name also contains "2k"/"step"-free tokens
        if any(n in low for n in SLOT_NEEDLES[slot]):
            return slot
    return None


def order_warnings(chosen: dict) -> list[str]:
    """Warnings when a slot is empty or a file looks like it belongs to another slot."""
    out = []
    for slot in SLOTS:
        name = chosen.get(slot, "none")
        if name in (None, "", "none"):
            if slot != "ic":
                out.append(f"WARNING: {slot} slot is 'none' - the documented chain is speed -> LMS Sharp -> IC")
            continue
        role = slot_role(name)
        if role is not None and role != slot:
            out.append(f"WARNING: {name!r} sits in the {slot} slot but looks like a {role} LoRA - the documented "
                       "order is speed (HyperFlow 8-step or DMAD 4-step) -> LMS Sharp r64 -> Open-H3 IC")
    return out


def detect_lora_format(keys) -> str:
    keys = list(keys)
    if any(k.startswith("transformer.") and k.endswith(".lora_A.weight") for k in keys):
        return "diffusers"
    if any(k.startswith(("diffusion_model.", "lora_unet_", "model.diffusion_model.", "transformer_blocks."))
           or ".lora_down." in k or ".lora_up." in k or k.endswith(".lora_A.weight") for k in keys):
        return "comfy"
    sample = ", ".join(keys[:3])
    raise ValueError(f"unknown LoRA key format (first keys: {sample or 'none'}); set ic_format explicitly or use the "
                     "ComfyUI-keyed / official Diffusers-keyed IC LoRA")


def diffusers_plan(sd: dict, expected_modules: int = IC_MODULES) -> tuple[dict, dict, dict]:
    """PEFT IC LoRA -> (lora, to_load, expect) for ``comfy.lora.load_lora``.

    lora: tensors under the PEFT base names (fc1 B halves swapped [value;gate] -> [gate;value]);
    to_load: base -> ``diffusion_model.<target>.weight`` or (that key, (0, row offset, rows)) for q/k/v slices of
    the fused qkv_proj; expect: target weight key -> required row count (3 x slice rows for qkv, B rows otherwise).
    """
    bases = sorted(k[:-len(".lora_A.weight")] for k in sd if k.endswith(".lora_A.weight"))
    if len(bases) != expected_modules:
        raise ValueError(f"expected {expected_modules} IC LoRA modules (Diffusers keys), found {len(bases)}")
    lora, to_load, expect = {}, {}, {}
    for base in bases:
        bkey = base + ".lora_B.weight"
        if bkey not in sd:
            raise ValueError(f"missing {bkey}")
        a, b = sd[base + ".lora_A.weight"], sd[bkey]
        try:
            key, slot = comfy_target(base)          # shared with STD (mmx_utils/open_regen_core.py)
        except KeyError as exc:
            raise ValueError(f"unexpected IC LoRA module {base}") from exc
        if isinstance(slot, int):                   # q / k / v: a row slice of the fused qkv_proj
            rows = int(b.shape[0])
            to_load[base] = (key, (0, slot * rows, rows))
            prev = expect.setdefault(key, 3 * rows)
            if prev != 3 * rows:
                raise ValueError(f"q/k/v output widths differ at {base}")
        elif slot == "swiglu":                      # fc1: B halves [value;gate] -> [gate;value]
            if b.shape[0] % 2:
                raise ValueError("fc1 LoRA output is not a SwiGLU pair")
            b = swap_swiglu_halves(b)
            to_load[base] = key
            expect[key] = int(b.shape[0])
        else:
            to_load[base] = key
            expect[key] = int(b.shape[0])
        lora[base + ".lora_A.weight"] = a
        lora[bkey] = b
        if base + ".alpha" in sd:
            lora[base + ".alpha"] = sd[base + ".alpha"]
    return lora, to_load, expect


def check_targets(expect: dict, model_shapes: dict) -> None:
    """Every target weight must exist in the model with the expected row count (wrong base -> error)."""
    missing = [k for k in expect if k not in model_shapes]
    if missing:
        raise ValueError(f"{len(missing)} of {len(expect)} IC LoRA target weights are not in this model (first: "
                         f"{missing[0]}). Load the Ref2VA (or hybrid FL2VA/Ref2VA) MiniMax H3 model.")
    bad = [(k, expect[k], tuple(model_shapes[k])) for k in expect if int(model_shapes[k][0]) != expect[k]]
    if bad:
        k, rows, shape = bad[0]
        raise ValueError(f"IC LoRA rows do not fit the model: {k} needs {rows} rows, model has {shape} "
                         f"({len(bad)} mismatches). Wrong base model?")
