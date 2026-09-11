# PORTED FROM: 1038lab/ComfyUI-Minimax-H3-Promptor :: py/vision_orchestrator.py
# Comfyui-Minimax-H3-Promptor — author 1038lab, GPL-3.0

from __future__ import annotations

from .utils import log_error, log_info, tensor_to_base64


def _get_iterable(media_input):
    if not media_input:
        return []
    if isinstance(media_input, dict):
        return media_input.values()
    if isinstance(media_input, (list, tuple)):
        return media_input
    return [media_input]


class VisionOrchestrator:
    def __init__(
        self,
        llm,
        prompt_builder,
        response_parser_cls,
        system_prompt: str,
        vibe_system_prompt: str,
        temperature: float,
        max_tokens: int,
        model_override: str,
    ):
        self.llm = llm
        self.pb = prompt_builder
        self.parser = response_parser_cls
        self.system_prompt = system_prompt
        self.vibe_system_prompt = vibe_system_prompt
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.model = model_override

    def analyze_all(
        self,
        ref_images,
        ref_videos,
        ref_audios,
        presets: dict,
        overrides: dict,
        global_image_mode: str,
        global_video_mode: str,
        output_language: str,
        batch_size: int,
        provider_label: str,
    ) -> tuple[dict, list]:
        media_keys: list[str] = []
        final_dict: dict = {}
        img_list = [img for img in _get_iterable(ref_images) if img is not None]
        vid_iterable = [vid for vid in _get_iterable(ref_videos) if vid is not None]
        if img_list:
            if batch_size > 1:
                self._process_images_batch(
                    img_list,
                    batch_size,
                    presets,
                    overrides,
                    global_image_mode,
                    output_language,
                    provider_label,
                    media_keys,
                    final_dict,
                )
            else:
                self._process_images_sequential(
                    img_list,
                    presets,
                    overrides,
                    global_image_mode,
                    output_language,
                    provider_label,
                    media_keys,
                    final_dict,
                )
        if vid_iterable:
            self._process_videos(
                vid_iterable,
                presets,
                overrides,
                global_video_mode,
                output_language,
                provider_label,
                media_keys,
                final_dict,
            )
        aud_iterable = [aud for aud in _get_iterable(ref_audios) if aud is not None]
        if aud_iterable:
            aud_index = 1
            for _aud in aud_iterable:
                target_key = f"<Audio {aud_index}>"
                media_keys.append(target_key)
                final_dict[target_key] = "[Audio file identified. LLM analysis not supported.]"
                aud_index += 1
        final_dict["_media_keys"] = media_keys
        return final_dict, media_keys

    @staticmethod
    def _get_prompt_str(presets: dict, mode: str, is_video: bool = False) -> str:
        dict_key = "video_prompts" if is_video else "image_prompts"
        return presets.get(dict_key, {}).get(mode, "Analyze visually.")

    def _call_llm(self, system_prompt, user_message, payload):
        return self.llm.chat(
            system_prompt=system_prompt,
            user_message=user_message,
            base64_images=payload,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            model=self.model,
        )

    def _process_images_batch(
        self, img_list, batch_size, presets, overrides, global_image_mode,
        output_language, provider_label, media_keys, final_dict,
    ):
        sub_batches = [img_list[i : i + batch_size] for i in range(0, len(img_list), batch_size)]
        img_index = 1
        for batch_idx, sub_batch in enumerate(sub_batches):
            chunk_keys = []
            instructions = []
            all_frames = []
            pos = 1
            for img in sub_batch:
                target_key = f"<Picture {img_index}>"
                media_keys.append(target_key)
                chunk_keys.append(target_key)
                frames = tensor_to_base64(img, max_frames=1)
                active_prompt = overrides.get(target_key, self._get_prompt_str(presets, global_image_mode))
                instructions.append(
                    self.pb.build_vision_request(output_language, target_key, active_prompt, pos=pos)
                )
                all_frames.extend(frames)
                img_index += 1
                pos += 1
            mega_prompt = "Order of attached media:\n" + "\n".join(instructions)
            payload = [{"text": mega_prompt}] + [{"image": f} for f in all_frames]
            log_info(f"Analyzer calling {provider_label} for IMAGES batch {batch_idx + 1}...")
            response = self._call_llm(self.system_prompt, "", payload)
            if response.success:
                final_dict.update(self.parser.parse_vision_response(response.content, chunk_keys))
            else:
                log_error(f"API Error in Image Batch {batch_idx + 1}: {response.error}")
                for key in chunk_keys:
                    final_dict[key] = f"API Error: {response.error}"

    def _process_images_sequential(
        self, img_list, presets, overrides, global_image_mode,
        output_language, provider_label, media_keys, final_dict,
    ):
        log_info(f"Analyzer calling {provider_label} for SEQUENTIAL image processing...")
        img_index = 1
        for img in img_list:
            target_key = f"<Picture {img_index}>"
            media_keys.append(target_key)
            frames = tensor_to_base64(img, max_frames=1)
            active_prompt = overrides.get(target_key, self._get_prompt_str(presets, global_image_mode))
            single_prompt = self.pb.build_vision_request(output_language, target_key, active_prompt, pos=1)
            payload = [{"text": single_prompt}] + [{"image": f} for f in frames]
            response = self._call_llm(self.system_prompt, "", payload)
            if response.success:
                final_dict.update(self.parser.parse_vision_response(response.content, [target_key]))
            else:
                log_error(f"API Error in Image {img_index}: {response.error}")
                final_dict[target_key] = f"API Error: {response.error}"
            img_index += 1

    def _process_videos(
        self, vid_iterable, presets, overrides, global_video_mode,
        output_language, provider_label, media_keys, final_dict,
    ):
        vid_index = 1
        for vid in vid_iterable:
            target_key = f"<Video {vid_index}>"
            media_keys.append(target_key)
            frames = tensor_to_base64(vid, max_frames=4)
            active_prompt = overrides.get(target_key, self._get_prompt_str(presets, global_video_mode, is_video=True))
            mega_prompt = self.pb.build_vision_request(
                output_language, target_key, active_prompt, is_video=True, num_frames=len(frames)
            )
            payload = [{"text": mega_prompt}] + [{"image": f} for f in frames]
            log_info(f"Analyzer calling {provider_label} for {target_key} ({len(frames)} frames)...")
            response = self._call_llm(self.system_prompt, "", payload)
            if response.success:
                final_dict.update(self.parser.parse_vision_response(response.content, [target_key]))
            else:
                log_error(f"API Error in Video {vid_index}: {response.error}")
                final_dict[target_key] = f"API Error: {response.error}"
            vid_index += 1
