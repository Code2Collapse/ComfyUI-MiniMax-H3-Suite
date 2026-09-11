# PORTED FROM: 1038lab/ComfyUI-Minimax-H3-Promptor
# Comfyui-Minimax-H3-Promptor — author 1038lab, GPL-3.0
# V3 node wrappers: MiniMaxH3_Promptor, MiniMaxH3_Vision, MiniMaxH3_PromptEditor, MiniMaxH3_PromptComposer

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

import torch

from comfy_api.latest import io

_PKG = Path(__file__).resolve().parents[1]
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from mmx_utils.promptor import PROMPTOR_ROOT
from mmx_utils.promptor.config_manager import get_config_manager
from mmx_utils.promptor.pipeline_engine import execute_director_pipeline, execute_vision_pipeline
from mmx_utils.promptor.task_detector import TASK_TYPE_OPTIONS
from mmx_utils.promptor.utils import log_error

try:
    import comfy.model_management as model_management
except ImportError:
    model_management = None

_PROMPT_CATEGORY = "MiniMax H3/Prompt"

with open(PROMPTOR_ROOT / "vision_prompts.json", "r", encoding="utf-8") as _f:
    _PRESETS = json.load(_f)

IMAGE_MODES = list(_PRESETS.get("image_prompts", {}).keys())
VIDEO_MODES = list(_PRESETS.get("video_prompts", {}).keys())

COMPOSER_MODES = [
    "T2VA (Text to Video & Audio)",
    "I2VA (Image to Video & Audio)",
    "FL2VA (First & Last Frame)",
    "Ref2VA (Omni / Reference)",
    "V2VA (Video to Video)",
    "L2VA (Live Action / Extended)",
    "A2V (Audio to Video)",
    "Custom / Blank",
]


def _provider_choices(default_key: str) -> tuple[list[str], str]:
    try:
        config = get_config_manager().load()
        active_providers: list[str] = []
        default_uuid = config.get("defaults", {}).get(default_key, "")
        default_choice = ""
        for k, v in config.get("providers", {}).items():
            if v.get("enabled", True) is not False:
                model_name = v.get("model", "").strip()
                raw_name = v.get("name", k)
                clean_raw = re.sub(r"\s*\([^)]*\)\s*$", "", raw_name).strip()
                name = f"{clean_raw} ({model_name})" if model_name else clean_raw
                active_providers.append(name)
                if k == default_uuid:
                    default_choice = name
        active_providers.sort()
        if not default_choice and active_providers:
            default_choice = active_providers[0]
        if not active_providers:
            active_providers = ["No Provider Configured"]
            default_choice = active_providers[0]
        return active_providers, default_choice
    except Exception:
        return ["Error Loading Providers"], "Error Loading Providers"


class MiniMaxH3_Promptor(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        active_providers, default_choice = _provider_choices("promptor_provider")
        return io.Schema(
            node_id="MiniMaxH3_Promptor",
            display_name="MiniMax H3 Promptor",
            category=_PROMPT_CATEGORY,
            description=(
                "Two-stage director pipeline for MiniMax H3 structured prompts. "
                "Ported from Comfyui-Minimax-H3-Promptor (1038lab, GPL-3.0)."
            ),
            inputs=[
                io.Combo.Input(
                    "task_type",
                    options=TASK_TYPE_OPTIONS,
                    default=TASK_TYPE_OPTIONS[0],
                    tooltip="Task format for MiniMax H3 or Auto detection from vision context.",
                ),
                io.String.Input(
                    "scene_direction",
                    multiline=True,
                    default="",
                    tooltip="Director instructions and scene plot.",
                ),
                io.Float.Input("duration", default=5.0, min=4.0, max=15.0, step=0.5),
                io.String.Input(
                    "vision_context",
                    multiline=True,
                    force_input=True,
                    default="",
                    optional=True,
                    tooltip="Connect VISION_CONTEXT from MiniMax H3 Vision.",
                ),
                io.Combo.Input(
                    "reference_images",
                    options=["Auto", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
                    default="Auto",
                    optional=True,
                ),
                io.Combo.Input(
                    "reference_videos",
                    options=["Auto", "1", "2", "3"],
                    default="Auto",
                    optional=True,
                ),
                io.Combo.Input(
                    "reference_audios",
                    options=["Auto", "1", "2", "3"],
                    default="Auto",
                    optional=True,
                ),
                io.Combo.Input("output_language", options=["English", "Chinese"], default="English", optional=True),
                io.Combo.Input("provider", options=active_providers, default=default_choice, optional=True),
                io.Float.Input("temperature", default=0.7, min=0.0, max=1.0, step=0.05, optional=True),
                io.Int.Input("max_tokens", default=4096, min=256, max=8192, step=256, optional=True),
            ],
            outputs=[
                io.String.Output("prompt", display_name="PROMPT"),
                io.Float.Output("duration", display_name="DURATION"),
                io.Int.Output("length", display_name="LENGTH"),
            ],
        )

    @classmethod
    def execute(
        cls,
        task_type: str,
        duration: float,
        scene_direction: str = "",
        vision_context: str = "",
        reference_images: str = "Auto",
        reference_videos: str = "Auto",
        reference_audios: str = "Auto",
        output_language: str = "English",
        provider: str = "",
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs,
    ) -> io.NodeOutput:
        user_prompt = scene_direction or kwargs.get("scene_direction", "") or kwargs.get("description", "")
        result = execute_director_pipeline(
            task_type=task_type,
            description=user_prompt,
            duration=duration,
            vision_context=vision_context,
            reference_images=reference_images,
            reference_videos=reference_videos,
            reference_audios=reference_audios,
            output_language=output_language,
            provider=provider,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return io.NodeOutput(result["final_prompt"], float(result["duration"]), int(result["length"]))


class MiniMaxH3_Vision(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        active_providers, default_choice = _provider_choices("vision_provider")
        return io.Schema(
            node_id="MiniMaxH3_Vision",
            display_name="MiniMax H3 Vision",
            category=_PROMPT_CATEGORY,
            description=(
                "Multimodal vision analysis for H3 prompt pipelines. "
                "Ported from Comfyui-Minimax-H3-Promptor (1038lab, GPL-3.0). No custom JS."
            ),
            is_output_node=True,
            inputs=[
                io.Combo.Input("global_image_mode", options=IMAGE_MODES, default="Subject / Identity"),
                io.Combo.Input("global_video_mode", options=VIDEO_MODES, default="Comprehensive"),
                io.Combo.Input("output_language", options=["English", "Chinese"], default="English", optional=True),
                io.Combo.Input("provider", options=active_providers, default=default_choice, optional=True),
                io.Float.Input("temperature", default=0.2, min=0.0, max=1.0, step=0.05, optional=True),
                io.Int.Input("max_tokens", default=2048, min=256, max=8192, step=256, optional=True),
                io.Int.Input("seed", default=0, min=0, max=0xFFFFFFFFFFFFFFFF, optional=True),
                io.Autogrow.Input(
                    "ref_images",
                    optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Image.Input("image"),
                        prefix="image_",
                        min=0,
                        max=9,
                    ),
                ),
                io.Autogrow.Input(
                    "ref_videos",
                    optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Video.Input("video"),
                        prefix="video_",
                        min=0,
                        max=3,
                    ),
                ),
                io.Autogrow.Input(
                    "ref_audios",
                    optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Audio.Input("audio"),
                        prefix="audio_",
                        min=0,
                        max=3,
                    ),
                ),
                io.String.Input("custom_prompt_override", multiline=True, default="", optional=True),
            ],
            outputs=[
                io.String.Output("vision_context", display_name="VISION_CONTEXT"),
                io.Image.Output("ref_images_out", display_name="REF_IMAGES"),
            ],
        )

    @classmethod
    def execute(
        cls,
        global_image_mode: str,
        global_video_mode: str,
        output_language: str = "English",
        provider: str = "",
        custom_prompt_override: str = "",
        temperature: float = 0.2,
        max_tokens: int = 2048,
        seed: int = 0,
        ref_images: io.Autogrow.Type = None,
        ref_videos: io.Autogrow.Type = None,
        ref_audios: io.Autogrow.Type = None,
        **kwargs,
    ) -> io.NodeOutput:
        del seed
        try:
            ref_images_dict: dict[str, Any] = {}
            ref_videos_dict: dict[str, Any] = {}
            ref_audios_dict: dict[str, Any] = {}

            if ref_images is not None:
                if isinstance(ref_images, dict):
                    for key in sorted(ref_images.keys()):
                        if ref_images[key] is not None:
                            k_name = key if key.startswith("image_") else f"image_{key}"
                            ref_images_dict[k_name if k_name.startswith("image_") else f"image_{k_name}"] = ref_images[key]
                elif isinstance(ref_images, (list, tuple)):
                    for idx, img in enumerate(ref_images):
                        if img is not None:
                            ref_images_dict[f"image_{idx}"] = img
                else:
                    ref_images_dict["image_0"] = ref_images

            if ref_videos is not None:
                if isinstance(ref_videos, dict):
                    for key in sorted(ref_videos.keys()):
                        if ref_videos[key] is not None:
                            k_name = key if key.startswith("video_") else f"video_{key}"
                            ref_videos_dict[k_name] = ref_videos[key]
                elif isinstance(ref_videos, (list, tuple)):
                    for idx, vid in enumerate(ref_videos):
                        if vid is not None:
                            ref_videos_dict[f"video_{idx}"] = vid
                else:
                    ref_videos_dict["video_0"] = ref_videos

            if ref_audios is not None:
                if isinstance(ref_audios, dict):
                    for key in sorted(ref_audios.keys()):
                        if ref_audios[key] is not None:
                            k_name = key if key.startswith("audio_") else f"audio_{key}"
                            ref_audios_dict[k_name] = ref_audios[key]
                elif isinstance(ref_audios, (list, tuple)):
                    for idx, aud in enumerate(ref_audios):
                        if aud is not None:
                            ref_audios_dict[f"audio_{idx}"] = aud
                else:
                    ref_audios_dict["audio_0"] = ref_audios

            for k, v in kwargs.items():
                if v is None:
                    continue
                if k.startswith("image_"):
                    ref_images_dict[k] = v
                elif k.startswith("video_"):
                    ref_videos_dict[k] = v
                elif k.startswith("audio_"):
                    ref_audios_dict[k] = v

            config_manager = get_config_manager()
            provider_key = config_manager.find_provider_by_display_name(provider)
            provider_config = config_manager.get_provider_config(provider_key)
            batch_size = provider_config.get("batch_size")
            if batch_size is None:
                batch_size = 1 if provider_config.get("batch_vision") is False else 4

            if provider_key == "ollama" and model_management:
                model_management.unload_all_models()
                model_management.soft_empty_cache()

            final_dict, _media_keys = execute_vision_pipeline(
                provider_name_or_key=provider_key,
                ref_images=ref_images_dict or None,
                ref_videos=ref_videos_dict or None,
                ref_audios=ref_audios_dict or None,
                global_image_mode=global_image_mode,
                global_video_mode=global_video_mode,
                output_language=output_language,
                temperature=temperature,
                max_tokens=max_tokens,
                custom_prompt_override=custom_prompt_override,
                batch_size=batch_size,
            )

            out_images = list(ref_images_dict.values()) if ref_images_dict else []
            if not final_dict:
                return io.NodeOutput("{}", out_images)

            final_output = json.dumps(final_dict, indent=4, ensure_ascii=False)
            if provider_key == "ollama" and model_management:
                model_management.soft_empty_cache()
            return io.NodeOutput(final_output, out_images)
        except Exception as e:
            log_error(str(e))
            fallback = [torch.zeros((1, 64, 64, 3), dtype=torch.float32)]
            return io.NodeOutput(f"[Analyzer Exception]: {e}", fallback)


class MiniMaxH3_PromptEditor(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MiniMaxH3_PromptEditor",
            display_name="MiniMax H3 Prompt Preview & Edit",
            category=_PROMPT_CATEGORY,
            description="Pass-through prompt editor widget (no custom JS). Prefers wired PROMPT over stored text.",
            is_output_node=True,
            inputs=[
                io.String.Input("prompt", multiline=True, force_input=True, default="", optional=True),
                io.String.Input("_stored_prompt", multiline=True, default="", tooltip="Edited prompt storage."),
            ],
            outputs=[io.String.Output("prompt", display_name="PROMPT")],
        )

    @classmethod
    def fingerprint_inputs(cls, prompt: str = "", _stored_prompt: str = "", **kwargs):
        incoming = str(prompt).strip() if prompt else ""
        stored = str(_stored_prompt).strip() if _stored_prompt else ""
        blob = incoming if incoming else stored
        return hashlib.md5(blob.encode("utf-8")).hexdigest()

    @classmethod
    def execute(cls, prompt: str = "", _stored_prompt: str = "", **kwargs) -> io.NodeOutput:
        incoming = str(prompt).strip() if prompt and str(prompt).strip() else ""
        stored = str(_stored_prompt).strip() if _stored_prompt else ""
        value = incoming if incoming else stored
        return io.NodeOutput(value, ui={"text": [value], "prompt": [value]})


class MiniMaxH3_PromptComposer(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MiniMaxH3_PromptComposer",
            display_name="MiniMax H3 Prompt Composer",
            category=_PROMPT_CATEGORY,
            description="Manual prompt composer scaffold (no custom JS). Output is stored composer text.",
            is_output_node=True,
            inputs=[
                io.Combo.Input("mode", options=COMPOSER_MODES, default=COMPOSER_MODES[0]),
                io.String.Input("_composer_prompt", multiline=True, default=""),
            ],
            outputs=[io.String.Output("prompt", display_name="PROMPT")],
        )

    @classmethod
    def fingerprint_inputs(cls, mode: str = COMPOSER_MODES[0], _composer_prompt: str = "", **kwargs):
        return hashlib.md5(f"{mode}|{_composer_prompt}".encode("utf-8")).hexdigest()

    @classmethod
    def execute(cls, mode: str = COMPOSER_MODES[0], _composer_prompt: str = "", **kwargs) -> io.NodeOutput:
        del mode
        val = str(_composer_prompt).strip() if _composer_prompt else ""
        return io.NodeOutput(val, ui={"text": [val], "prompt": [val]})
