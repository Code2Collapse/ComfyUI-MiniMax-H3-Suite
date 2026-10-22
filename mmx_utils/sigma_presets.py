"""Sigma presets per model, taken from ComfyUI's official workflow templates (owner ORDERS 11.6, ledger L8.19).

Every preset cites where its numbers come from: the template(s) in comfyui-workflow-templates-json 0.1.104 that ship
these settings (extracted by the work area's tools/c2c_sigma_presets_from_templates.py), the model's trained shift in
core's comfy/supported_models.py, or, for SD 1.5 (no template ships it any more), the frontend's own default graph
(src/scripts/defaultGraph.ts, frontend 1.53.10). The curves are computed by core's own code (comfy.model_sampling,
comfy.samplers.calculate_sigmas, and the LTXV / Flux2 / SDTurbo scheduler nodes), so they follow the installed ComfyUI.
A preset that depends on a speed LoRA or a distilled model says so: its step count is wrong without it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Optional

import torch

SOURCE = "official ComfyUI templates, comfyui-workflow-templates-json 0.1.104"


@dataclass(frozen=True)
class Preset:
    label: str
    family: str
    sampling: tuple                 # ("eps",) | ("flow", shift, multiplier) | ("flux", shift) | ("av", shift_v, shift_a)
    schedule: tuple                 # ("calc", scheduler) | ("ltxv", max_shift, base_shift, stretch, terminal)
    #                                 | ("flux2",) | ("sdturbo",) | ("manual", [sigmas...])
    steps: int
    sampler: str
    cfg: float
    guidance: str
    source: str
    lora: str = ""
    notes: tuple = field(default_factory=tuple)


PRESETS: dict[str, Preset] = {p.label: p for p in (
    Preset("MiniMax H3 Ref2VA (template, ref2v turbo LoRA)", "MiniMax H3", ("av", 12.0, 3.0), ("calc", "simple"), 20,
           "res_multistep", 1.0, "BasicGuider (no negative prompt)",
           "templates video_minimax_h3_r2v, video_minimax_h3_multiframe_reference, "
           "video_minimax_h3_fun_controlnet_union; shift 12 / 3 = core supported_models.MiniMaxH3",
           "minimax_h3_ref2v_turbo_4step_v0.1 at 1.0 (in the template)"),
    Preset("MiniMax H3 FL2VA / T2V (template, fl2v turbo LoRA)", "MiniMax H3", ("av", 12.0, 3.0), ("calc", "simple"),
           4, "res_multistep", 1.0, "BasicGuider (no negative prompt)",
           "templates video_minimax_h3_i2v, video_minimax_h3_t2v, video_minimax_h3_i2v_continuation; shift 12 / 3 = "
           "core supported_models.MiniMaxH3",
           "minimax_h3_fl2v_turbo_8step_v1.0 at 1.0 (in the template; it runs 4 steps although the LoRA is named "
           "8step)"),
    Preset("FastH3 8-step (template, distilled model)", "MiniMax H3", ("av", 10.0, 3.0), ("calc", "simple"), 8,
           "res_multistep", 1.0, "BasicGuider (no negative prompt)",
           "templates video_fastvideo_fasth3_t2v, video_fastvideo_fasth3_i2v (MiniMaxH3SigmaShift 10 / 3)",
           "none: fastvideo_fasth3_8step_v2 is itself the distilled model"),
    Preset("Wan 2.1 14B T2V / I2V (template)", "Wan", ("flow", 8.0, 1000), ("calc", "simple"), 20, "uni_pc", 6.0,
           "KSampler cfg 6 with a negative prompt",
           "templates image_to_video_wan (20 steps), text_to_video_wan (30 steps); shift 8 = core "
           "supported_models.WAN21_T2V"),
    Preset("Wan 2.2 14B T2V / I2V (template, lightx2v 4-step LoRAs)", "Wan", ("flow", 5.0, 1000), ("calc", "simple"),
           4, "euler", 1.0, "KSamplerAdvanced cfg 1, two models",
           "templates video_wan2_2_14B_t2v, video_wan2_2_14B_i2v (ModelSamplingSD3 shift 5)",
           "wan2.2_*_lightx2v_4steps_lora on BOTH models at 1.0",
           ("High-noise model for steps 0-2, low-noise model for steps 2-4: SplitSigmas at step 2.",)),
    Preset("Wan Animate 2 (template)", "Wan", ("flow", 5.0, 1000), ("calc", "simple"), 6, "lcm", 1.0,
           "SamplerCustom cfg 1", "template video_wan_animate2 (ModelSamplingSD3 shift 5)"),
    Preset("Flux.1 dev (template)", "Flux", ("flux", 1.15), ("calc", "simple"), 20, "euler", 1.0,
           "FluxGuidance 3.5, cfg 1", "templates flux1_dev_uso_reference_image_gen and the other flux1-dev templates; "
           "shift 1.15 = core ModelSamplingFlux default"),
    Preset("Flux.1 schnell (template)", "Flux", ("flow", 1.0, 1.0), ("calc", "simple"), 4, "euler", 1.0, "cfg 1",
           "templates flux_schnell, flux_schnell_full_text_to_image; core supported_models.FluxSchnell"),
    Preset("Flux.2 dev (template)", "Flux", ("none",), ("flux2",), 20, "euler", 1.0, "FluxGuidance 4",
           "templates image_flux2, image_flux2_fp8 (Flux2Scheduler at the image size)"),
    Preset("Flux.2 Klein base (template)", "Flux", ("none",), ("flux2",), 20, "euler", 5.0, "CFGGuider cfg 5",
           "templates image_flux2_klein_image_edit_4b_base, image_flux2_klein_image_edit_9b_base"),
    Preset("Flux.2 Klein distilled (template)", "Flux", ("none",), ("flux2",), 4, "euler", 1.0, "CFGGuider cfg 1",
           "templates image_flux2_klein_image_edit_4b_distilled, image_flux2_klein_9b_kv_image_edit"),
    Preset("SDXL (template)", "Stable Diffusion", ("eps",), ("calc", "karras"), 25, "dpmpp_2m", 7.0, "KSampler cfg 7",
           "template image_sdxl_simple"),
    Preset("SDXL Turbo (template)", "Stable Diffusion", ("eps",), ("sdturbo",), 1, "euler_ancestral", 1.0, "cfg 1",
           "template sdxlturbo_example (SDTurboScheduler)"),
    Preset("SD 1.5 (ComfyUI default graph)", "Stable Diffusion", ("eps",), ("calc", "normal"), 20, "euler", 8.0,
           "KSampler cfg 8", "frontend 1.53.10 src/scripts/defaultGraph.ts (no template ships SD 1.5 any more)"),
    Preset("SD 3.5 Large (template)", "Stable Diffusion", ("flow", 3.0, 1000), ("calc", "simple"), 30, "euler", 4.0,
           "KSampler cfg 4", "template sd3.5_large_blur; shift 3 = core supported_models.SD3"),
    Preset("LTX-2 dev stage 1 (template, distilled LoRA)", "LTX", ("none",), ("ltxv", 2.05, 0.95, True, 0.1), 12,
           "euler_ancestral", 1.0, "CFGGuider cfg 1",
           "template video_ltx2_i2v_lora (LTXVScheduler 12 steps, max_shift 2.05, base_shift 0.95, stretch, "
           "terminal 0.1)", "ltx-2-19b-distilled-lora-384 at 1.0 (in the template)",
           ("Stage 2 of that template: the 'LTX-2 distilled stage 2' preset.",)),
    Preset("LTX-2 distilled stage 2 (template)", "LTX", ("none",), ("manual", [0.909375, 0.725, 0.421875, 0.0]), 3,
           "euler_ancestral", 1.0, "CFGGuider cfg 1",
           "templates video_ltx2_i2v_lora, video_ltx2_canny_to_video, video_ltx2_depth_to_video (ManualSigmas)"),
    Preset("LTX-2.3 / 2.5 stage 1 (template)", "LTX", ("none",),
           ("manual", [1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0]), 8, "euler_ancestral",
           1.0, "CFGGuider cfg 1",
           "templates video_ltx2_5_i2v, video_ltx2_5_flf2v, video_ltx2_3_t2v (ManualSigmas)",
           "LTX-2.3 dev: ltx_2.3_22b_distilled_1.1_lora at 0.5 (in the template); LTX-2.5: the distilled "
           "transformer", ("Stage 2: the 'LTX-2.3 / 2.5 stage 2' preset.",)),
    Preset("LTX-2.3 / 2.5 stage 2 (template)", "LTX", ("none",), ("manual", [0.85, 0.725, 0.4219, 0.0]), 3, "euler",
           1.0, "CFGGuider cfg 1", "templates video_ltx2_5_i2v, video_ltx2_3_t2v, video_ltx2_3_i2v (ManualSigmas)"),
    Preset("HunyuanVideo (template)", "Hunyuan", ("flow", 7.0, 1000), ("calc", "simple"), 20, "euler", 1.0,
           "BasicGuider, FluxGuidance 6",
           "template hunyuan_video_text_to_video (ModelSamplingSD3 shift 7); core supported_models.HunyuanVideo"),
    Preset("HunyuanVideo 1.5 (template)", "Hunyuan", ("flow", 7.0, 1000), ("calc", "simple"), 20, "euler", 6.0,
           "CFGGuider cfg 6",
           "templates video_hunyuan_video_1.5_720p_t2v, _i2v (shift 7; their super-resolution stage uses shift 2)"),
    Preset("Qwen-Image (template, Lightning 8-step LoRA)", "Qwen", ("flow", 3.1, 1.0), ("calc", "simple"), 8, "euler",
           1.0, "KSampler cfg 1", "template image_qwen_image (ModelSamplingAuraFlow shift 3.1)",
           "Qwen-Image-Lightning-8steps-V1.0 at 1.0 (in the template)"),
    Preset("Qwen-Image 2512 (template)", "Qwen", ("flow", 3.1, 1.0), ("calc", "simple"), 50, "euler", 4.0,
           "KSampler cfg 4", "templates image_qwen_Image_2512, image_qwen_Image_2512_controlnet (shift 3.1)",
           "", ("The template also wires a Qwen-Image-2512-Lightning-4steps LoRA node; these 50 steps / cfg 4 are "
                "its KSampler values.",)),
    Preset("Z-Image (template)", "Z-Image", ("flow", 3.0, 1.0), ("calc", "simple"), 25, "res_multistep", 4.0,
           "KSampler cfg 4", "template image_z_image (ModelSamplingAuraFlow shift 3; core supported_models.ZImage)"),
    Preset("Z-Image Turbo (template)", "Z-Image", ("flow", 3.0, 1.0), ("calc", "simple"), 8, "res_multistep", 1.0,
           "KSampler cfg 1", "template image_z_image_turbo (ModelSamplingAuraFlow shift 3)",
           "none: z_image_turbo is itself the distilled model"),
    Preset("Chroma1-HD (template)", "Other", ("flow", 1.0, 1.0), ("calc", "beta"), 26, "euler", 3.5, "CFGGuider cfg 3.5",
           "template image_chroma_text_to_image (ModelSamplingAuraFlow shift 1, beta scheduler)"),
    Preset("HiDream-I1 Full (template)", "Other", ("flow", 3.0, 1000), ("calc", "simple"), 50, "uni_pc", 5.0,
           "KSampler cfg 5", "template hidream_i1_full (ModelSamplingSD3 shift 3)"),
    Preset("HiDream-I1 Dev (template)", "Other", ("flow", 6.0, 1000), ("calc", "normal"), 28, "lcm", 1.0,
           "KSampler cfg 1", "template hidream_i1_dev (ModelSamplingSD3 shift 6)"),
    Preset("HiDream-I1 Fast (template)", "Other", ("flow", 3.0, 1000), ("calc", "normal"), 16, "lcm", 1.0,
           "KSampler cfg 1", "template hidream_i1_fast (ModelSamplingSD3 shift 3)"),
)}

LABELS = list(PRESETS)


def model_sampling_for(spec: tuple):
    """A core model_sampling object for a preset's sampling spec (no weights needed)."""
    import comfy.model_sampling as M

    kind = spec[0]
    if kind == "eps":
        class _EPS(M.ModelSamplingDiscrete, M.EPS):
            pass
        return _EPS(None)
    if kind == "flow":
        class _Flow(M.ModelSamplingDiscreteFlow, M.CONST):
            pass
        return _Flow(SimpleNamespace(sampling_settings={"shift": float(spec[1]), "multiplier": float(spec[2])}))
    if kind == "flux":
        class _Flux(M.ModelSamplingFlux, M.CONST):
            pass
        return _Flux(SimpleNamespace(sampling_settings={"shift": float(spec[1])}))
    if kind == "av":
        class _AV(M.ModelSamplingAV, M.CONST):
            pass
        return _AV(SimpleNamespace(sampling_settings={"shift": float(spec[1]), "audio_shift": float(spec[2])}))
    raise ValueError(f"no model sampling for {spec!r}")


def _out0(o):
    r = o.result if hasattr(o, "result") else (o.args if hasattr(o, "args") else o)
    return r[0] if isinstance(r, (tuple, list)) else r


def _with_denoise(full_fn, steps: int, denoise: float) -> torch.Tensor:
    """BasicScheduler's denoise: the last `steps` steps of a schedule of steps / denoise."""
    if denoise >= 0.9999:
        return full_fn(steps)
    total = max(steps, int(steps / max(denoise, 1e-4)))
    return full_fn(total)[-(steps + 1):]


def preset_sigmas(label: str, steps: int = 0, denoise: float = 1.0, model=None, width: int = 1024,
                  height: int = 1024) -> tuple[torch.Tensor, Optional[torch.Tensor], str]:
    """(video sigmas, audio sigmas for H3 or None, where the curve came from)."""
    p = PRESETS[label]
    n = int(steps) if int(steps) > 0 else p.steps
    kind = p.schedule[0]
    audio = None
    if kind == "calc":
        import comfy.samplers
        if model is not None:
            ms = model.get_model_object("model_sampling")
            origin = "your model's own sampling (its shift patches included)"
        else:
            ms = model_sampling_for(p.sampling)
            origin = "the preset's model sampling (" + ", ".join(str(x) for x in p.sampling) + ")"
        sig = _with_denoise(lambda k: comfy.samplers.calculate_sigmas(ms, p.schedule[1], k).cpu(), n, float(denoise))
        if p.sampling[0] == "av" or getattr(ms, "audio_shift", None):
            audio = audio_sigmas(sig, ms)
    elif kind == "ltxv":
        from comfy_extras.nodes_lt import LTXVScheduler
        _k, mx, base, stretch, terminal = p.schedule
        sig = _with_denoise(lambda k: _out0(LTXVScheduler.execute(k, mx, base, stretch, terminal)).cpu(), n,
                            float(denoise))
        origin = f"core LTXVScheduler (max_shift {mx}, base_shift {base}, stretch {stretch}, terminal {terminal})"
    elif kind == "flux2":
        from comfy_extras.nodes_flux import Flux2Scheduler
        sig = _with_denoise(lambda k: _out0(Flux2Scheduler.execute(k, int(width), int(height))).cpu(), n,
                            float(denoise))
        origin = f"core Flux2Scheduler at {int(width)} x {int(height)}"
    elif kind == "sdturbo":
        from comfy_extras.nodes_custom_sampler import SDTurboScheduler
        ms = model.get_model_object("model_sampling") if model is not None else model_sampling_for(p.sampling)
        fake = SimpleNamespace(get_model_object=lambda name: ms)
        sig = _out0(SDTurboScheduler.execute(fake, n, float(denoise))).cpu()
        origin = "core SDTurboScheduler"
    elif kind == "manual":
        sig = torch.tensor(p.schedule[1], dtype=torch.float32)
        if int(steps) > 0 and int(steps) != len(sig) - 1:
            origin = (f"the template's fixed sigmas ({len(sig) - 1} steps; a distilled schedule, steps are not "
                      "changed)")
        else:
            origin = "the template's fixed sigmas"
    else:  # pragma: no cover
        raise ValueError(kind)
    return sig.float(), audio, origin


def audio_sigmas(video: torch.Tensor, ms) -> Optional[torch.Tensor]:
    """H3's audio clock for the same steps (core comfy.ldm.minimax.model.time_shift_sigma)."""
    shift_a = getattr(ms, "audio_shift", None)
    shift_v = float(getattr(ms, "shift", 1.0))
    if shift_a is None:
        return None
    try:
        from comfy.ldm.minimax.model import time_shift_sigma
    except Exception:  # noqa: BLE001
        return None
    v = video.clamp(min=0.0)
    out = time_shift_sigma(v.clamp(min=1e-6), shift_v, float(shift_a))
    out = torch.where(v <= 0, torch.zeros_like(out), out)
    return out.float()


def how_to(label: str, sig: torch.Tensor, audio, origin: str, steps: int) -> str:
    p = PRESETS[label]
    n = len(sig) - 1
    lines = [f"{p.label}  [{p.family}]",
             f"  sampler {p.sampler}, {n} steps, {p.guidance}" + (f", cfg {p.cfg:g}" if "cfg" not in p.guidance else ""),
             f"  sigmas: {origin}",
             f"  first / last sigma: {float(sig[0]):.4f} -> {float(sig[-1]):.4f}"]
    if audio is not None:
        lines.append(f"  H3 audio clock: {float(audio[0]):.4f} -> {float(audio[-1]):.4f}")
    if p.lora:
        lines.append(f"  needs: {p.lora}")
    lines.append(f"  source: {p.source} ({SOURCE if 'template' in p.source else 'ComfyUI'})")
    lines += [f"  note: {x}" for x in p.notes]
    lines.append("  wiring: sigmas -> SamplerCustom(Advanced) sigmas, sampler -> its sampler; use cfg on the guider.")
    return "\n".join(lines)
