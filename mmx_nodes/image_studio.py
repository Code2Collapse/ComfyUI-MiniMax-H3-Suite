# SPDX-License-Identifier: Unlicense
# PORTED FROM: ComfyUI-MiniMax-H3-Image-Studio :: nodes.py @ upstream
"""H3 still-image generation nodes — short frame packets, decode, frame pick."""

from __future__ import annotations

import copy
import sys
from pathlib import Path
from typing import Optional

import comfy.model_management
import comfy.model_sampling
import comfy.nested_tensor
import comfy.samplers
import node_helpers
import torch
from comfy_api.latest import io

_PKG = Path(__file__).resolve().parents[1]
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from mmx_utils.h3_constants import CANVAS_MULTIPLE, FPS, AUDIO_LATENT_FPS  # noqa: E402
from mmx_utils.h3_grid import latent_t_for_frame_count  # noqa: E402
from mmx_utils.image_studio import (  # noqa: E402
    ASPECT_RATIOS,
    CUSTOM_SAMPLING_PROFILE,
    FRAME_PRESETS,
    LEGACY_SAMPLING_PROFILES,
    RECOMMENDED_FRAME_PROFILE,
    RESOLUTION_PROFILES,
    SAMPLING_PROFILES,
    SINGLE_IMAGE_FRAME_PROFILE,
    calculate_custom_resolution,
    calculate_preset_resolution,
    collect_reference_images,
    detail_tone_lock,
    first_stable_edit_frame,
    normalize_prompt,
    prompt_warning,
    reference_resize,
    resize_image,
    resolve_frame_count,
    round_to_multiple,
    select_still_frame,
    stable_quality_frame,
)

CATEGORY = "MiniMax H3/Image"


def _empty_h3_av_latent(
    width: int,
    height: int,
    length: int,
    batch_size: int = 1,
    output_frames: Optional[int] = None,
    output_frame_index: int = 0,
    output_strategy: str = "fixed",
):
    internal_frames = max(1, int(length))
    latent_t, natural_frames = latent_t_for_frame_count(internal_frames)
    requested_frames = internal_frames if output_frames is None else max(1, int(output_frames))
    requested_frames = min(requested_frames, natural_frames)
    duration = natural_frames / FPS
    audio_t = max(1, round(duration * AUDIO_LATENT_FPS))
    device = comfy.model_management.intermediate_device()
    video = torch.zeros((batch_size, 24, latent_t, height // 16, width // 16), device=device)
    audio = torch.zeros((batch_size, 32, 2, audio_t), device=device)
    nested = comfy.nested_tensor.NestedTensor((video, audio))
    return {
        "samples": nested,
        "h3_requested_frames": requested_frames,
        "h3_context_frames": internal_frames,
        "h3_natural_frames": natural_frames,
        "h3_output_frame_index": max(0, int(output_frame_index)),
        "h3_output_strategy": str(output_strategy),
    }, requested_frames, natural_frames


def _apply_h3_shift(model, shift_video: float, shift_audio: float):
    m = model.clone()
    av_sampling = getattr(comfy.model_sampling, "ModelSamplingAV", None)
    sampling_base = av_sampling or comfy.model_sampling.ModelSamplingDiscreteFlow

    class ModelSamplingAdvanced(sampling_base, comfy.model_sampling.CONST):
        if av_sampling is None:
            audio_shift = None

            @property
            def audio_scale(self):
                if self.audio_shift is None:
                    return 1.0
                return self.shift / self.audio_shift

    original = m.get_model_object("model_sampling")
    model_sampling = ModelSamplingAdvanced(m.model.model_config)
    multiplier = getattr(original, "multiplier", 1000)
    if av_sampling is not None:
        model_sampling.set_parameters(
            shift=float(shift_video),
            audio_shift=float(shift_audio),
            multiplier=multiplier,
        )
        sampling_backend = "ModelSamplingAV"
    else:
        model_sampling.set_parameters(shift=float(shift_video), multiplier=multiplier)
        model_sampling.audio_shift = float(shift_audio)
        sampling_backend = "ModelSamplingAV compatibility shim"
    if hasattr(original, "noise_scale"):
        model_sampling.set_noise_scale(original.noise_scale)
    m.add_object_patch("model_sampling", model_sampling)

    transformer_options = m.model_options.get("transformer_options", {}).copy()
    transformer_options["minimax_h3_sigma_shift_video"] = float(shift_video)
    transformer_options["minimax_h3_sigma_shift_audio"] = float(shift_audio)
    m.model_options["transformer_options"] = transformer_options
    return m, sampling_backend


def _build_sampling(
    model,
    sampler_name: str,
    scheduler: str,
    steps: int,
    denoise: float,
    shift_video: float,
    shift_audio: float,
    beta_alpha: float,
    beta_beta: float,
) -> tuple:
    shifted_model, sampling_backend = _apply_h3_shift(model, shift_video, shift_audio)
    sampler = comfy.samplers.sampler_object(sampler_name)

    model_sampling = shifted_model.get_model_object("model_sampling")
    steps = max(1, int(steps))
    denoise = max(0.0, min(1.0, float(denoise)))

    if denoise <= 0.0:
        sigmas = torch.FloatTensor([])
        beta_note = f" | beta alpha={beta_alpha:g}, beta={beta_beta:g}" if scheduler == "beta_custom" else ""
        report = (
            f"sampler={sampler_name} | scheduler={scheduler} | steps={steps} | denoise=0 | "
            f"schedule_steps=0 | shift_video={shift_video:g} | shift_audio={shift_audio:g} | "
            f"backend={sampling_backend}{beta_note}"
        )
        return shifted_model, sampler, sigmas, report

    total_steps = steps if denoise >= 1.0 else max(steps, int(steps / denoise))

    if scheduler == "beta_custom":
        sigmas = comfy.samplers.beta_scheduler(
            model_sampling,
            total_steps,
            alpha=float(beta_alpha),
            beta=float(beta_beta),
        ).cpu()
    else:
        sigmas = comfy.samplers.calculate_sigmas(
            model_sampling,
            scheduler,
            total_steps,
        ).cpu()

    sigmas = sigmas[-(steps + 1):]

    beta_note = f" | beta alpha={beta_alpha:g}, beta={beta_beta:g}" if scheduler == "beta_custom" else ""
    report = (
        f"sampler={sampler_name} | scheduler={scheduler} | steps={steps} | denoise={denoise:g} | "
        f"schedule_steps={total_steps} | shift_video={shift_video:g} | shift_audio={shift_audio:g} | "
        f"backend={sampling_backend}{beta_note}"
    )
    return shifted_model, sampler, sigmas, report


def _prepare_image_common(
    clip,
    mode: str,
    prompt: str,
    width: int,
    height: int,
    frame_preset: str,
    optimize_prompt: bool,
    preserve_strength: float,
    source_fit: str,
    reference_size: str,
    vae=None,
    source_image: Optional[torch.Tensor] = None,
    reference_image_2: Optional[torch.Tensor] = None,
    reference_image_3: Optional[torch.Tensor] = None,
    reference_image_4: Optional[torch.Tensor] = None,
    reference_image_5: Optional[torch.Tensor] = None,
    reference_image_6: Optional[torch.Tensor] = None,
    reference_image_7: Optional[torch.Tensor] = None,
    reference_image_8: Optional[torch.Tensor] = None,
    reference_image_9: Optional[torch.Tensor] = None,
    reference_transport: str = "native",
):
    width = round_to_multiple(width, CANVAS_MULTIPLE)
    height = round_to_multiple(height, CANVAS_MULTIPLE)
    internal_frames = resolve_frame_count(frame_preset)
    single_frame_i2i = mode == "image_to_image (FL2VA)" and internal_frames == 1
    conditioning_mode = "reference_edit (REF2VA)" if single_frame_i2i else mode
    if reference_transport not in ("native", "semantic (experimental)"):
        raise ValueError("Unknown reference transport. Choose native or semantic (experimental).")

    output_frames = internal_frames
    output_frame_index = 0
    if mode == "image_to_image (FL2VA)" and internal_frames > 1:
        output_strategy = "first_stable_edit"
    else:
        output_strategy = "stable_quality"
    latent, requested_frames, natural_frames = _empty_h3_av_latent(
        width,
        height,
        internal_frames,
        output_frames=output_frames,
        output_frame_index=output_frame_index,
        output_strategy=output_strategy,
    )

    additional_references = (
        reference_image_2, reference_image_3, reference_image_4, reference_image_5,
        reference_image_6, reference_image_7, reference_image_8, reference_image_9,
    )
    ignored_notes = []
    if mode != "reference_edit (REF2VA)" and any(image is not None for image in additional_references):
        ignored_notes.append("Additional reference_image_2..9 inputs are connected but ignored outside REF2VA mode.")
    if mode == "text_to_image (FL2VA)" and source_image is not None:
        ignored_notes.append("source_image is connected but ignored in Text to Image mode.")
    active_additional_references = (
        additional_references if mode == "reference_edit (REF2VA)" else (None,) * len(additional_references)
    )
    if conditioning_mode == "reference_edit (REF2VA)":
        connected_references = (source_image, *active_additional_references)
        batched_inputs = [
            index for index, image in enumerate(connected_references, start=1)
            if isinstance(image, torch.Tensor) and image.ndim == 4 and image.shape[0] > 1
        ]
        if batched_inputs:
            ignored_notes.append(
                "REF2VA uses the first image from each reference socket; extra batch images were ignored for "
                f"Picture input(s) {', '.join(map(str, batched_inputs))}."
            )

    references = (
        collect_reference_images(source_image, active_additional_references)
        if conditioning_mode == "reference_edit (REF2VA)" and source_image is not None
        else []
    )
    final_prompt = normalize_prompt(
        conditioning_mode, prompt, optimize_prompt, preserve_strength, max(1, len(references))
    )
    if conditioning_mode == "reference_edit (REF2VA)" and len(references) > 1:
        missing_tags = [
            f"<Picture {index}>" for index in range(1, len(references) + 1)
            if f"<picture {index}>" not in (prompt or "").lower()
        ]
        if missing_tags:
            ignored_notes.append(
                "Target instructions do not explicitly assign " + ", ".join(missing_tags) +
                "; reference adherence is more reliable when every connected picture has a stated role."
            )

    black = torch.zeros((1, height, width, 3), dtype=torch.float32)
    fitted_source = black

    if mode == "text_to_image (FL2VA)":
        tokens = clip.tokenize(final_prompt, images=[])
        cond = clip.encode_from_tokens_scheduled(tokens)
        checkpoint_note = "Use an FL2VA checkpoint."

    elif mode == "image_to_image (FL2VA)" and not single_frame_i2i:
        if source_image is None:
            raise ValueError("Image to Image mode requires source_image.")
        if vae is None:
            raise ValueError("Image to Image mode requires a VAE to encode source_image.")
        fitted_source = resize_image(source_image[:1], width, height, source_fit)
        tokens = clip.tokenize(final_prompt, images=[fitted_source])
        cond = clip.encode_from_tokens_scheduled(tokens)
        keyframe_latent = vae.encode(fitted_source)
        cond = node_helpers.conditioning_set_values(cond, {
            "minimax_keyframes": [{"resolved_frame_index": 0, "latent": keyframe_latent}],
            "minimax_frame_count": natural_frames,
        })
        checkpoint_note = "Use an FL2VA checkpoint; frame 0 is the exact source anchor."

    else:
        if source_image is None:
            raise ValueError("Reference Edit mode requires source_image as <Picture 1>.")
        if vae is None and reference_transport == "native":
            raise ValueError("Reference Edit mode requires a VAE to encode the reference image(s).")
        fitted_source = resize_image(references[0], width, height, source_fit)
        ref_mode = "max_identity_2048" if reference_size == "max_identity_2048" else "match_generation_area"
        ref_items = []
        ref_blocks = []
        reference_sizes = []
        for reference_image in references:
            reference, tw, th = reference_resize(reference_image, width, height, ref_mode)
            ref_items.append({"type": "image", "data": reference})
            if reference_transport == "native":
                ref_blocks.append({
                    "kind": "image",
                    "latent_h": th // 16,
                    "latent_w": tw // 16,
                    "latent": vae.encode(reference),
                })
            reference_sizes.append(f"{tw}x{th}")
        tokens = clip.tokenize(final_prompt, minimax_ref_items=ref_items)
        cond = clip.encode_from_tokens_scheduled(tokens)
        if ref_blocks:
            cond = node_helpers.conditioning_set_values(cond, {"minimax_refs": ref_blocks})
        if single_frame_i2i:
            checkpoint_note = (
                "Use a hybrid or REF2VA checkpoint; one-frame I2I uses Picture 1 reference conditioning instead "
                "of an FL2VA frame-0 keyframe so the only output frame remains editable. "
                f"The source was encoded as {reference_sizes[0]}."
            )
        else:
            checkpoint_note = (
                f"Use a REF2VA checkpoint; {len(references)} ordered reference image(s) encoded "
                f"as {', '.join(reference_sizes)} and exposed as <Picture 1> through <Picture {len(references)}>."
            )
        if reference_transport == "semantic (experimental)":
            checkpoint_note += (
                " Semantic references use the vision encoder only, without VAE reference latents. "
                "Identity and fine details may change."
            )

    if natural_frames > 362:
        trained_note = "beyond the documented 124-362-frame training range"
    elif natural_frames >= 124:
        trained_note = "inside the documented 124-362-frame training range"
    else:
        trained_note = "short experimental temporal packet chosen to reduce image-mode compute"
    decode_note = (
        f"exact {requested_frames}-frame batch"
        if requested_frames == natural_frames
        else (
            f"temporal latent naturally decodes {natural_frames} frames; "
            f"H3 Exact Frame Decode keeps the requested {requested_frames}"
        )
    )
    ignored_text = f" {' '.join(ignored_notes)}" if ignored_notes else ""
    image_vae_note = (
        " The one-frame profile requires minimax_h3_t1_image_vae_step1597.safetensors for the intended sharp "
        "single-image decode; the standard video VAE can look soft."
        if internal_frames == 1 else ""
    )
    report = (
        f"Mode: {mode} | temporal profile: {internal_frames} frames | canvas {width}×{height} | "
        f"internal packet {natural_frames} frames | decoded profile {requested_frames} | {decode_note} | "
        f"{trained_note}. {checkpoint_note} Decode only the video latent; the audio VAE is unnecessary for image output. "
        f"Preferred output strategy: {output_strategy}; Single Image Output receives the full decoded profile and "
        f"normally emits one selected frame unless emit_candidate_batch is enabled.{ignored_text}{prompt_warning(prompt)}"
        f"{image_vae_note}"
    )
    return cond, latent, fitted_source, requested_frames, final_prompt, report


class MiniMaxH3_ImageResolution(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3_ImageResolution",
            display_name="H3 Image Resolution",
            category=CATEGORY,
            description=(
                "Calculate a custom H3 canvas on the 32-pixel grid with an optional native-area cap. "
                "Oversize canvases cost VRAM without guaranteed detail gain — H3-Base was trained near 768p."
            ),
            inputs=[
                io.Combo.Input(
                    "aspect_ratio",
                    options=["source image"] + list(ASPECT_RATIOS.keys()) + ["custom dimensions"],
                    default="16:9 landscape",
                    tooltip="Wrong aspect for your subject leaves letterboxing or a cropped head. "
                            "'source image' copies a connected reference ratio.",
                ),
                io.Float.Input(
                    "megapixels", default=1.00, min=0.10, max=64.00, step=0.10,
                    tooltip="Target area in ComfyUI megapixels (1 MP = 1024²). "
                            "Above ~1 MP you are outside H3-Base's native training envelope.",
                ),
                io.Combo.Input("multiple", options=[32, 64], default=32,
                               tooltip="Both axes snap to this grid. H3 requires multiples of 32."),
                io.Boolean.Input(
                    "native_area_cap", default=True,
                    tooltip="Cap total pixels near 768×1344. Turn off only if you accept "
                            "VRAM spikes and soft detail from out-of-distribution resolution.",
                ),
                io.Int.Input("custom_width", default=2048, min=32, max=16384, step=32,
                             tooltip='Only when aspect_ratio is "custom dimensions".'),
                io.Int.Input("custom_height", default=2048, min=32, max=16384, step=32,
                             tooltip='Only when aspect_ratio is "custom dimensions".'),
                io.Image.Input("source_image", optional=True,
                               tooltip='Required only for aspect_ratio "source image".'),
            ],
            outputs=[
                io.Int.Output("width"),
                io.Int.Output("height"),
                io.String.Output("report"),
            ],
        )

    @classmethod
    def execute(cls, aspect_ratio, megapixels, multiple, native_area_cap,
                custom_width, custom_height, source_image=None) -> io.NodeOutput:
        width, height, report = calculate_custom_resolution(
            aspect_ratio, megapixels, int(multiple), bool(native_area_cap),
            custom_width, custom_height, source_image,
        )
        return io.NodeOutput(width, height, report)


class MiniMaxH3_ImageResolutionPreset(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3_ImageResolutionPreset",
            display_name="H3 Image Resolution Preset",
            category=CATEGORY,
            description="Preset megapixel targets for still generation on H3's 32-pixel grid.",
            inputs=[
                io.Combo.Input(
                    "aspect_ratio",
                    options=["source image"] + list(ASPECT_RATIOS.keys()),
                    default="16:9 landscape",
                    tooltip="Portrait vs landscape changes where H3 puts detail — pick the ratio "
                            "your subject actually fills.",
                ),
                io.Combo.Input(
                    "resolution_profile",
                    options=list(RESOLUTION_PROFILES.keys()),
                    default="native detail | 0.98 MP",
                    tooltip="Higher MP costs decode and attention; H3 detail does not scale linearly. "
                            "Start at native detail unless you have VRAM headroom.",
                ),
                io.Image.Input("source_image", optional=True),
                io.Float.Input("custom_megapixels", default=2.0, min=0.10, max=64.0, step=0.10, optional=True,
                               tooltip='Only when resolution_profile is "custom megapixels".'),
                io.Boolean.Input(
                    "limit_to_native_area", default=False, optional=True,
                    tooltip="Hard-cap near 768×1344 even when the profile name suggests oversize.",
                ),
            ],
            outputs=[
                io.Int.Output("width"),
                io.Int.Output("height"),
                io.String.Output("report"),
            ],
        )

    @classmethod
    def execute(cls, aspect_ratio, resolution_profile, source_image=None,
                custom_megapixels=2.0, limit_to_native_area=False) -> io.NodeOutput:
        width, height, report = calculate_preset_resolution(
            aspect_ratio, resolution_profile, source_image, custom_megapixels, limit_to_native_area,
        )
        return io.NodeOutput(width, height, report)


class MiniMaxH3_ImagePrepare(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3_ImagePrepare",
            display_name="H3 Image Prepare (combined)",
            category=CATEGORY,
            description=(
                "Prepare T2I, I2I, or REF2VA conditioning plus an empty H3 AV latent for still-image sampling."
            ),
            inputs=[
                io.Clip.Input("clip"),
                io.Combo.Input(
                    "mode",
                    options=[
                        "text_to_image (FL2VA)",
                        "image_to_image (FL2VA)",
                        "reference_edit (REF2VA)",
                    ],
                    default="text_to_image (FL2VA)",
                    tooltip="FL2VA paths anchor or encode sources differently. REF2VA needs explicit "
                            "<Picture N> tags in the prompt or references drift.",
                ),
                io.String.Input("prompt", default="", multiline=True,
                                tooltip="Describe ONE final still. Video phrasing ('pan left', 'over 3 seconds') "
                                        "makes the temporal packet wander between frames."),
                io.Int.Input("width", default=1344, min=32, max=16384, step=32),
                io.Int.Input("height", default=768, min=32, max=16384, step=32),
                io.Combo.Input("frame_preset", options=list(FRAME_PRESETS.keys()),
                               default=RECOMMENDED_FRAME_PROFILE,
                               tooltip="H3 denoises the whole packet jointly. More frames add temporal context "
                                       "but raise cost; one frame needs the experimental image VAE."),
                io.Boolean.Input("optimize_prompt", default=True,
                                 tooltip="Wraps the prompt for a locked-camera still. Does not change steps or CFG."),
                io.Float.Input("preserve_strength", default=0.60, min=0.0, max=1.0, step=0.05,
                               tooltip="Prompt-language preservation for edits — NOT diffusion denoise strength."),
                io.Combo.Input("source_fit", options=["crop_center", "contain_pad", "stretch"],
                               default="crop_center",
                               tooltip="How sources map to the generation canvas before VAE encode."),
                io.Combo.Input("reference_size", options=["match_generation_area", "max_identity_2048"],
                               default="max_identity_2048",
                               tooltip="REF2VA encode resolution. max_identity_2048 keeps more identity detail."),
                io.Vae.Input("vae", optional=True),
                io.Image.Input("source_image", optional=True),
                io.Image.Input("reference_image_2", optional=True),
                io.Image.Input("reference_image_3", optional=True),
                io.Image.Input("reference_image_4", optional=True),
                io.Image.Input("reference_image_5", optional=True),
                io.Image.Input("reference_image_6", optional=True),
                io.Image.Input("reference_image_7", optional=True),
                io.Image.Input("reference_image_8", optional=True),
                io.Image.Input("reference_image_9", optional=True),
            ],
            outputs=[
                io.Conditioning.Output("positive"),
                io.Latent.Output("h3_latent"),
                io.Image.Output("fitted_source"),
                io.Int.Output("requested_frames"),
                io.String.Output("optimized_prompt"),
                io.String.Output("report"),
            ],
        )

    @classmethod
    def execute(cls, clip, mode, prompt, width, height, frame_preset, optimize_prompt,
                preserve_strength, source_fit, reference_size, vae=None, source_image=None,
                reference_image_2=None, reference_image_3=None, reference_image_4=None,
                reference_image_5=None, reference_image_6=None, reference_image_7=None,
                reference_image_8=None, reference_image_9=None) -> io.NodeOutput:
        out = _prepare_image_common(
            clip, mode, prompt, width, height, frame_preset, optimize_prompt, preserve_strength,
            source_fit, reference_size, vae, source_image,
            reference_image_2, reference_image_3, reference_image_4, reference_image_5,
            reference_image_6, reference_image_7, reference_image_8, reference_image_9,
        )
        return io.NodeOutput(*out)


class MiniMaxH3_TextToImagePrepare(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3_TextToImagePrepare",
            display_name="H3 Text to Image Prepare",
            category=CATEGORY,
            description="FL2VA text-to-image conditioning and empty H3 latent for still generation.",
            inputs=[
                io.Clip.Input("clip"),
                io.String.Input("prompt", default="", multiline=True,
                                tooltip="One finished still — not a shot list or storyboard."),
                io.Int.Input("width", default=1344, min=32, max=16384, step=32),
                io.Int.Input("height", default=768, min=32, max=16384, step=32),
                io.Combo.Input("quality_profile", options=list(FRAME_PRESETS.keys()),
                               default=RECOMMENDED_FRAME_PROFILE),
                io.Boolean.Input("optimize_for_still", default=True),
            ],
            outputs=[
                io.Conditioning.Output("positive"),
                io.Latent.Output("h3_latent"),
                io.Int.Output("requested_frames"),
                io.String.Output("image_prompt"),
                io.String.Output("report"),
            ],
        )

    @classmethod
    def execute(cls, clip, prompt, width, height, quality_profile, optimize_for_still) -> io.NodeOutput:
        cond, latent, _src, frames, image_prompt, report = _prepare_image_common(
            clip=clip, vae=None, mode="text_to_image (FL2VA)", prompt=prompt,
            width=width, height=height, frame_preset=quality_profile,
            optimize_prompt=optimize_for_still, preserve_strength=0.75,
            source_fit="crop_center", reference_size="match_generation_area", source_image=None,
        )
        return io.NodeOutput(cond, latent, frames, image_prompt, report)


class MiniMaxH3_ImageToImagePrepare(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3_ImageToImagePrepare",
            display_name="H3 Image to Image Prepare",
            category=CATEGORY,
            description="Multi-frame FL2VA anchor or one-frame REF2VA-conditioned image editing.",
            inputs=[
                io.Clip.Input("clip"),
                io.Vae.Input("vae"),
                io.Image.Input("source_image"),
                io.String.Input("edit_instruction", default="", multiline=True),
                io.Int.Input("width", default=1344, min=32, max=16384, step=32),
                io.Int.Input("height", default=768, min=32, max=16384, step=32),
                io.Combo.Input("quality_profile", options=list(FRAME_PRESETS.keys()),
                               default=RECOMMENDED_FRAME_PROFILE,
                               tooltip="One frame switches to REF2VA-style conditioning and needs the image VAE."),
                io.Float.Input("source_fidelity", default=0.75, min=0.0, max=1.0, step=0.05,
                               tooltip="Prompt preservation strength — not sampler denoise."),
                io.Combo.Input("source_fit", options=["crop_center", "contain_pad", "stretch"],
                               default="crop_center"),
                io.Boolean.Input("optimize_for_still", default=True),
            ],
            outputs=[
                io.Conditioning.Output("positive"),
                io.Latent.Output("h3_latent"),
                io.Image.Output("fitted_source"),
                io.Int.Output("requested_frames"),
                io.String.Output("image_prompt"),
                io.String.Output("report"),
            ],
        )

    @classmethod
    def execute(cls, clip, vae, source_image, edit_instruction, width, height, quality_profile,
                source_fidelity, source_fit, optimize_for_still) -> io.NodeOutput:
        out = _prepare_image_common(
            clip=clip, vae=vae, mode="image_to_image (FL2VA)", prompt=edit_instruction,
            width=width, height=height, frame_preset=quality_profile,
            optimize_prompt=optimize_for_still, preserve_strength=source_fidelity,
            source_fit=source_fit,
            reference_size=(
                "max_identity_2048" if quality_profile == SINGLE_IMAGE_FRAME_PROFILE
                else "match_generation_area"
            ),
            source_image=source_image,
        )
        return io.NodeOutput(*out)


class MiniMaxH3_ReferenceEditPrepare(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3_ReferenceEditPrepare",
            display_name="H3 Reference Edit Prepare",
            category=CATEGORY,
            description="REF2VA editing with up to nine ordered references.",
            inputs=[
                io.Clip.Input("clip"),
                io.Vae.Input("vae"),
                io.Image.Input("source_image"),
                io.String.Input("edit_instruction", default="", multiline=True,
                                tooltip="Refer to inputs as <Picture 1>, <Picture 2>, …"),
                io.Int.Input("width", default=1344, min=32, max=16384, step=32),
                io.Int.Input("height", default=768, min=32, max=16384, step=32),
                io.Combo.Input("quality_profile", options=list(FRAME_PRESETS.keys()),
                               default=RECOMMENDED_FRAME_PROFILE),
                io.Float.Input("source_fidelity", default=0.60, min=0.0, max=1.0, step=0.05),
                io.Combo.Input("source_fit", options=["crop_center", "contain_pad", "stretch"],
                               default="crop_center"),
                io.Combo.Input("reference_detail", options=["match_generation_area", "max_identity_2048"],
                               default="match_generation_area"),
                io.Boolean.Input("optimize_for_still", default=True),
                io.Image.Input("reference_image_2", optional=True),
                io.Image.Input("reference_image_3", optional=True),
                io.Image.Input("reference_image_4", optional=True),
                io.Image.Input("reference_image_5", optional=True),
                io.Image.Input("reference_image_6", optional=True),
                io.Image.Input("reference_image_7", optional=True),
                io.Image.Input("reference_image_8", optional=True),
                io.Image.Input("reference_image_9", optional=True),
                io.Combo.Input("reference_transport", options=["native", "semantic (experimental)"],
                               default="native", optional=True),
            ],
            outputs=[
                io.Conditioning.Output("positive"),
                io.Latent.Output("h3_latent"),
                io.Image.Output("fitted_source"),
                io.Int.Output("requested_frames"),
                io.String.Output("image_prompt"),
                io.String.Output("report"),
            ],
        )

    @classmethod
    def execute(cls, clip, vae, source_image, edit_instruction, width, height, quality_profile,
                source_fidelity, source_fit, reference_detail, optimize_for_still,
                reference_image_2=None, reference_image_3=None, reference_image_4=None,
                reference_image_5=None, reference_image_6=None, reference_image_7=None,
                reference_image_8=None, reference_image_9=None,
                reference_transport="native") -> io.NodeOutput:
        out = _prepare_image_common(
            clip=clip, vae=vae, mode="reference_edit (REF2VA)", prompt=edit_instruction,
            width=width, height=height, frame_preset=quality_profile,
            optimize_prompt=optimize_for_still, preserve_strength=source_fidelity,
            source_fit=source_fit, reference_size=reference_detail, source_image=source_image,
            reference_image_2=reference_image_2, reference_image_3=reference_image_3,
            reference_image_4=reference_image_4, reference_image_5=reference_image_5,
            reference_image_6=reference_image_6, reference_image_7=reference_image_7,
            reference_image_8=reference_image_8, reference_image_9=reference_image_9,
            reference_transport=reference_transport,
        )
        return io.NodeOutput(*out)


class MiniMaxH3_ImageDecode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3_ImageDecode",
            display_name="H3 Image Exact Frame Decode",
            category=CATEGORY,
            description="Decode the requested temporal frame profile from a sampled H3 latent.",
            inputs=[
                io.Latent.Input("samples"),
                io.Vae.Input("vae"),
                io.Combo.Input("decode_mode", options=["temporal", "single_latent_slice"],
                               default="temporal", optional=True),
                io.Int.Input("latent_index", default=0, min=0, max=4096, optional=True),
                io.Combo.Input("spatial_decode", options=["native", "full_image (experimental)"],
                               default="native", optional=True),
            ],
            outputs=[
                io.Image.Output("frames"),
                io.Int.Output("decoded_frames"),
                io.String.Output("report"),
                io.Int.Output("recommended_index"),
            ],
        )

    @classmethod
    def execute(cls, samples, vae, decode_mode="temporal", latent_index=0,
                spatial_decode="native") -> io.NodeOutput:
        latent = samples["samples"]
        if latent.is_nested:
            latent = latent.unbind()[0]

        if decode_mode == "single_latent_slice":
            if latent.ndim != 5 or not 0 <= latent_index < latent.shape[2]:
                raise ValueError("latent_index must select an existing slice of an H3 video latent.")
            decode_vae = vae
            first_stage = getattr(vae, "first_stage_model", None)
            if spatial_decode not in ("native", "full_image (experimental)"):
                raise ValueError("Unknown spatial decode mode.")
            if spatial_decode == "full_image (experimental)" and first_stage is not None and hasattr(first_stage, "tiling"):
                decode_vae = copy.copy(vae)
                decode_vae.first_stage_model = copy.copy(first_stage)
                decode_vae.first_stage_model.tiling = False
                if hasattr(vae, "memory_used_decode"):
                    original_estimate = vae.memory_used_decode
                    decode_vae.memory_used_decode = lambda shape, dtype: max(
                        original_estimate(shape, dtype),
                        shape[-2] * shape[-1] * 2048 * 96 * comfy.model_management.dtype_size(dtype),
                    )
            image = decode_vae.decode(latent[:, :, latent_index:latent_index + 1].clone())
            if image.ndim == 5:
                image = image.reshape(-1, *image.shape[-3:])
            if image.ndim != 4 or image.shape[0] != latent.shape[0]:
                raise ValueError("Single-slice decoding requires a compatible H3 VAE returning one image per batch item.")
            report = f"Independently decoded latent slice {latent_index}, spatial mode {spatial_decode}; one image per batch item."
            return io.NodeOutput(image, int(image.shape[0]), report, 0)

        if decode_mode != "temporal":
            raise ValueError("Unknown decode mode.")

        latent_batch = int(latent.shape[0]) if hasattr(latent, "shape") and len(latent.shape) > 0 else 1
        images = vae.decode(latent)

        profile_frames = max(
            1,
            int(samples.get("h3_context_frames", samples.get("h3_requested_frames", 1))),
        )
        output_strategy = str(samples.get("h3_output_strategy", "fixed"))

        if images.ndim == 5:
            batched = images
        elif images.ndim == 4 and latent_batch > 1 and int(images.shape[0]) % latent_batch == 0:
            frames_per_item = int(images.shape[0]) // latent_batch
            batched = images.reshape(latent_batch, frames_per_item, *images.shape[-3:])
        else:
            batched = images.unsqueeze(0)

        batch_size = int(batched.shape[0])
        natural_frames = int(batched.shape[1])
        kept_frames = min(profile_frames, natural_frames)
        kept = batched[:, :kept_frames]

        preferred_indices = []
        recommendation_scores = []
        fixed_index = max(0, int(samples.get("h3_output_frame_index", 0)))
        for batch_index in range(batch_size):
            item = kept[batch_index]
            if output_strategy == "first_stable_edit":
                preferred_index, recommendation_score = first_stable_edit_frame(item)
            elif output_strategy == "stable_quality":
                preferred_index, recommendation_score = stable_quality_frame(item)
            else:
                preferred_index, recommendation_score = fixed_index, 1.0
            preferred_index = min(max(0, int(preferred_index)), kept_frames - 1)
            preferred_indices.append(preferred_index)
            recommendation_scores.append(float(recommendation_score))

        images_out = kept.reshape(-1, *kept.shape[-3:]).clone()
        decoded_frames = int(images_out.shape[0])

        if natural_frames == kept_frames:
            packet_note = f"Decoded the complete natural {natural_frames}-frame packet per batch item."
        else:
            packet_note = (
                f"The temporal latent naturally decoded {natural_frames} frames per batch item; kept the requested "
                f"{kept_frames}-frame profile for each item."
            )

        preferred_text = ", ".join(
            f"b{index}:frame {preferred_indices[index]} (score {recommendation_scores[index]:.4f})"
            for index in range(batch_size)
        )
        report = (
            f"{packet_note} Batch items={batch_size}; emitted images={decoded_frames}. "
            f"Preferred still(s) via {output_strategy}: {preferred_text}. "
            "No requested profile frames were discarded before Single Image Output."
        )
        return io.NodeOutput(images_out, decoded_frames, report, preferred_indices[0])


class MiniMaxH3_ImageFrameSelector(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3_ImageFrameSelector",
            display_name="H3 Image Single Output",
            category=CATEGORY,
            description="Select one decoded still by index or quality metric.",
            inputs=[
                io.Image.Input("frames"),
                io.Combo.Input(
                    "strategy",
                    options=[
                        "decode_recommended", "first", "stable_quality", "balanced_edit",
                        "best_quality", "most_similar_to_source", "sharpest", "middle", "last", "manual_index",
                    ],
                    default="decode_recommended",
                    tooltip="sharpest picks edge energy — motion blur mid-transition scores high and "
                            "looks wrong. stable_quality penalises neighbour disagreement.",
                ),
                io.Int.Input("manual_index", default=0, min=0, max=4096),
                io.Int.Input("skip_first_frames", default=0, min=0, max=128),
                io.Float.Input("candidate_start", default=0.0, min=0.0, max=1.0, step=0.05),
                io.Float.Input("candidate_end", default=1.0, min=0.0, max=1.0, step=0.05),
                io.Float.Input("similarity_weight", default=0.60, min=0.0, max=1.0, step=0.05),
                io.Int.Input("top_k", default=4, min=1, max=16),
                io.Image.Input("source_image", optional=True),
                io.Boolean.Input("emit_candidate_batch", default=False, optional=True),
                io.Int.Input("recommended_index", optional=True, force_input=True),
            ],
            outputs=[
                io.Image.Output("selected_image"),
                io.Image.Output("candidate_batch_debug"),
                io.Int.Output("selected_index"),
                io.Float.Output("selected_score"),
                io.String.Output("report"),
            ],
        )

    @classmethod
    def execute(cls, frames, strategy, manual_index, skip_first_frames, candidate_start,
                candidate_end, similarity_weight, top_k, source_image=None,
                emit_candidate_batch=False, recommended_index=None) -> io.NodeOutput:
        primary, debug, index, score, report = select_still_frame(
            frames, strategy, manual_index, skip_first_frames, candidate_start, candidate_end,
            similarity_weight, top_k, source_image, emit_candidate_batch, recommended_index,
        )
        return io.NodeOutput(primary, debug, index, score, report)


class MiniMaxH3_ImageSamplingPreset(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        scheduler_options = list(comfy.samplers.SCHEDULER_NAMES)
        if "beta_custom" not in scheduler_options:
            scheduler_options.append("beta_custom")
        return io.Schema(
            node_id="MiniMaxH3_ImageSamplingPreset",
            display_name="H3 Image Sampling Preset",
            category=CATEGORY,
            description=(
                "Documented still-image sampling recipes. Turbo profiles must match their LoRA; "
                "wrong sampler/step pairs look washed or crunchy."
            ),
            inputs=[
                io.Model.Input("model"),
                io.Combo.Input(
                    "sampling_profile",
                    options=list(SAMPLING_PROFILES.keys()) + [CUSTOM_SAMPLING_PROFILE] + list(LEGACY_SAMPLING_PROFILES.keys()),
                    default="base quality | RES 20 steps",
                ),
                io.Combo.Input("custom_sampler", options=list(comfy.samplers.SAMPLER_NAMES),
                               default="res_multistep", optional=True),
                io.Combo.Input("custom_scheduler", options=scheduler_options, default="simple", optional=True),
                io.Int.Input("custom_steps", default=20, min=1, max=10000, optional=True),
                io.Float.Input("custom_denoise", default=1.0, min=0.0, max=1.0, step=0.01, optional=True),
                io.Float.Input("custom_shift_video", default=12.0, min=0.01, max=100.0, step=0.01, optional=True),
                io.Float.Input("custom_shift_audio", default=3.0, min=0.01, max=100.0, step=0.01, optional=True),
                io.Float.Input("custom_beta_alpha", default=0.6, min=0.01, max=50.0, step=0.01, optional=True),
                io.Float.Input("custom_beta_beta", default=0.6, min=0.01, max=50.0, step=0.01, optional=True),
            ],
            outputs=[
                io.Model.Output("model"),
                io.Sampler.Output("sampler"),
                io.Sigmas.Output("sigmas"),
                io.String.Output("report"),
            ],
        )

    @classmethod
    def execute(cls, model, sampling_profile, custom_sampler="res_multistep", custom_scheduler="simple",
                custom_steps=20, custom_denoise=1.0, custom_shift_video=12.0, custom_shift_audio=3.0,
                custom_beta_alpha=0.6, custom_beta_beta=0.6) -> io.NodeOutput:
        if sampling_profile == CUSTOM_SAMPLING_PROFILE:
            sampler_name = custom_sampler
            scheduler = custom_scheduler
            steps = custom_steps
            denoise = custom_denoise
            shift_video = custom_shift_video
            shift_audio = custom_shift_audio
            beta_alpha = custom_beta_alpha
            beta_beta = custom_beta_beta
        else:
            profiles = {**LEGACY_SAMPLING_PROFILES, **SAMPLING_PROFILES}
            if sampling_profile not in profiles:
                raise ValueError(f"Unknown H3 sampling profile: {sampling_profile}")
            sampler_name, scheduler, steps, shift_video, shift_audio = profiles[sampling_profile]
            denoise = 1.0
            beta_alpha = 0.6
            beta_beta = 0.6
        shifted_model, sampler, sigmas, info = _build_sampling(
            model, sampler_name, scheduler, steps, denoise,
            shift_video, shift_audio, beta_alpha, beta_beta,
        )
        return io.NodeOutput(shifted_model, sampler, sigmas, f"profile={sampling_profile} | {info}")


class MiniMaxH3_DetailToneLock(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3_DetailToneLock",
            display_name="H3 Detail Tone Lock",
            category=CATEGORY,
            description=(
                "After a detail-refinement pass, restore the source's broad lighting and colour "
                "while keeping sharpened micro-detail."
            ),
            inputs=[
                io.Image.Input("source_image",
                               tooltip="Authoritative tone and dimensions."),
                io.Image.Input("refined_image",
                               tooltip="Sharpened or upscaled result that may have drifted in global colour."),
                io.Float.Input("tone_lock", default=0.85, min=0.0, max=1.0, step=0.05,
                               tooltip="How much source low-frequency colour returns. Too low and the still "
                                       "keeps the refiner's colour cast."),
                io.Float.Input("refinement_strength", default=0.45, min=0.0, max=1.0, step=0.05,
                               tooltip="How much refined detail is mixed in. Too high drifts identity."),
                io.Int.Input("detail_radius", default=16, min=2, max=64, step=2,
                             tooltip="Separates broad tone from fine detail in pixels."),
            ],
            outputs=[
                io.Image.Output("image"),
                io.String.Output("report"),
            ],
        )

    @classmethod
    def execute(cls, source_image, refined_image, tone_lock, refinement_strength,
                detail_radius) -> io.NodeOutput:
        out = detail_tone_lock(source_image, refined_image, tone_lock, refinement_strength, detail_radius)
        report = (
            f"Detail tone lock: source {int(source_image.shape[2])}×{int(source_image.shape[1])}, "
            f"tone_lock={float(tone_lock):g}, refinement_strength={float(refinement_strength):g}, "
            f"detail_radius={int(detail_radius)} px."
        )
        return io.NodeOutput(out, report)


NODE_LIST = [
    MiniMaxH3_ImageResolution,
    MiniMaxH3_ImageResolutionPreset,
    MiniMaxH3_ImagePrepare,
    MiniMaxH3_TextToImagePrepare,
    MiniMaxH3_ImageToImagePrepare,
    MiniMaxH3_ReferenceEditPrepare,
    MiniMaxH3_ImageDecode,
    MiniMaxH3_ImageFrameSelector,
    MiniMaxH3_ImageSamplingPreset,
    MiniMaxH3_DetailToneLock,
]
