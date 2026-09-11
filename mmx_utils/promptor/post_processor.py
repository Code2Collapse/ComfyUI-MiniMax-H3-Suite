# PORTED FROM: 1038lab/ComfyUI-Minimax-H3-Promptor :: py/post_processor.py
# Comfyui-Minimax-H3-Promptor — author 1038lab, GPL-3.0

from __future__ import annotations

import re

from .utils import log_warning, sanitize_llm_output

H3_MAX_CHARS = 7000
SOUND_PREFIXES = ("audio:", "sound:", "soundscape:", "sfx:")
MUSIC_PREFIXES = ("music:", "score:", "soundtrack:")
BASE_MODES = ("T2V", "I2V", "I2VA", "FL2VA", "L2VA", "A2V")


class PostProcessor:
    @staticmethod
    def split_audio_music(text: str) -> tuple[str, str, str]:
        if not text:
            return "", "", ""
        kept, sound, music = [], [], []
        current = None
        for line in text.splitlines():
            stripped = line.strip()
            low = stripped.lower()
            if low.startswith("overall_soundscape:"):
                current = sound
                val = stripped.split(":", 1)[1].strip()
                if val:
                    sound.append(val)
                continue
            if low.startswith("non_diegetic_music:"):
                current = music
                val = stripped.split(":", 1)[1].strip()
                if val:
                    music.append(val)
                continue
            is_sound = any(low.startswith(p) for p in SOUND_PREFIXES)
            if is_sound:
                current = sound
                val = stripped.split(":", 1)[1].strip()
                if val:
                    sound.append(val)
                continue
            is_music = any(low.startswith(p) for p in MUSIC_PREFIXES)
            if is_music:
                current = music
                val = stripped.split(":", 1)[1].strip()
                if val:
                    music.append(val)
                continue
            if current is not None:
                if stripped:
                    current.append(stripped)
                else:
                    current = None
            else:
                kept.append(line)
        body = "\n".join(kept).strip()
        body = re.sub(
            r"^(?:integrated_multimodal_description|detailed_description):\s*\n?",
            "",
            body,
            flags=re.IGNORECASE,
        ).strip()
        return body, " ".join(sound).strip(), " ".join(music).strip()

    @staticmethod
    def extract_shot_appearances(body: str) -> dict[str, list[str]]:
        shots = re.split(r"(?=\[Shot\s+\d+)", body, flags=re.IGNORECASE)
        appearances: dict[str, list[str]] = {}
        for shot in shots:
            shot_match = re.search(r"\[Shot\s+(\d+)", shot, re.IGNORECASE)
            if not shot_match:
                continue
            shot_tag = f"[Shot {shot_match.group(1)}]"
            for tag in set(re.findall(r"<(?:Subject|Picture|Video|Audio)\s+\d+>", shot)):
                appearances.setdefault(tag, []).append(shot_tag)
        return appearances

    @staticmethod
    def construct_ref_blocks(
        task_type: str, subject_definitions: str, body_text: str, duration: float = 5.0
    ) -> tuple[str, str]:
        if not subject_definitions.strip():
            return "", ""
        declared_subjects = re.findall(r"<Subject\s+\d+>", subject_definitions)
        declared_pictures = re.findall(r"<Picture\s+\d+>", subject_definitions)
        has_video = "<Video 1>" in subject_definitions
        has_audio = "<Audio 1>" in subject_definitions
        prefix = "video editing" if task_type in ["V2V", "V2VA"] else "reference generation"
        summary = (
            f"summary:\n[{prefix}] A {duration:0.1f}-second live-action sequence executing the "
            f"requested visual narrative across shots using the defined references."
        )
        appearances = PostProcessor.extract_shot_appearances(body_text)
        retention = ["retention_analysis:"]
        for subj in declared_subjects:
            shot_list = appearances.get(subj, [])
            if not shot_list:
                s_idx = re.search(r"\d+", subj)
                if s_idx:
                    shot_list = appearances.get(f"<Picture {s_idx.group(0)}>", [])
            trait_match = re.search(rf"{subj}.*?with (.*?)\.", subject_definitions, re.IGNORECASE)
            specific_trait = (
                trait_match.group(1).strip() + " maintained."
                if trait_match and trait_match.group(1).strip()
                else "visual appearance and features maintained."
            )
            if shot_list:
                retention.append(f"{subj} (appears in {', '.join(shot_list)}): fully_preserved - {specific_trait}")
            else:
                retention.append(f"{subj} (appears in target video): fully_preserved - {specific_trait}")
        for pic in set(declared_pictures):
            p_num_match = re.search(r"\d+", pic)
            p_num = p_num_match.group(0) if p_num_match else ""
            if pic in appearances and not any(f"<Subject {p_num}>" in s for s in declared_subjects):
                retention.append(
                    f"{pic}: reference - environment setting and visual tone followed in {', '.join(appearances[pic])}."
                )
        if has_video:
            v_shots = appearances.get("<Video 1>", [])
            if v_shots:
                retention.append(
                    f"<Video 1> (camera and motion): weak_reference - motion timing and camera work followed in {', '.join(v_shots)}."
                )
            else:
                retention.append("<Video 1> (camera and motion): weak_reference - motion timing and camera work followed.")
        if has_audio:
            retention.append("<Audio 1>: reference - voice timbre and audio synchronization maintained.")
        return summary, "\n".join(retention)

    @staticmethod
    def compile_final_prompt(
        creative_text: str,
        task_type: str,
        subject_definitions: str = "",
        alignment_instructions: str = "",
        duration: float = 5.0,
        user_description: str = "",
        output_language: str = "English",
    ) -> str:
        prompt_text = sanitize_llm_output(creative_text)
        prompt_text = re.sub(
            r"(?<![\w/\\])[\w .()\-\u4e00-\u9fff]+\.(?:png|jpe?g|webp|bmp|gif|mp4|mov|webm|mkv|avi|mp3|wav|flac|m4a|ogg|aac)(?!\w)",
            "",
            prompt_text,
            flags=re.IGNORECASE,
        ).strip()
        prompt_text = re.sub(
            r"<(?:Environment|Setting|Location|Scene)\s+(\d+)>",
            lambda m: f"<Subject {m.group(1)}>",
            prompt_text,
            flags=re.IGNORECASE,
        )
        body, soundscape, music = PostProcessor.split_audio_music(prompt_text)
        detailed_match = re.search(r"detailed_description:\s*\n?(.*)", body, flags=re.DOTALL | re.IGNORECASE)
        if detailed_match:
            body = detailed_match.group(1).strip()
        if body and not re.search(r"^\s*\[Shot\s+1\b", body, re.IGNORECASE):
            first_shot = re.search(r"\[Shot\s+\d+\b", body, re.IGNORECASE)
            body = body[first_shot.start() :].strip() if first_shot else f"[Shot 1] {body.strip()}"
        if not soundscape:
            soundscape = "Ambient room tone and subtle environmental sound effects matching the scene actions."
        if not music:
            music = "N/A"
        parts = []
        if task_type in BASE_MODES:
            if alignment_instructions.strip():
                parts.append(alignment_instructions.strip())
            parts.append("integrated_multimodal_description:\n" + body.strip())
            parts.append("overall_soundscape:\n" + soundscape)
            parts.append("non_diegetic_music:\n" + music)
        else:
            if subject_definitions.strip():
                parts.append("subject_definitions:\n" + subject_definitions.strip())
                summary_block, retention_block = PostProcessor.construct_ref_blocks(
                    task_type, subject_definitions, body, duration=duration
                )
                if summary_block:
                    parts.append(summary_block)
                if retention_block:
                    parts.append(retention_block)
            parts.append("detailed_description:\n" + body.strip())
            parts.append("overall_soundscape:\n" + soundscape)
            parts.append("non_diegetic_music:\n" + music)
        return "\n\n".join(parts).strip()

    @staticmethod
    def clean(
        raw_output: str,
        task_type: str = "T2V",
        full_task_desc: str = "",
        subject_defs: str = "",
        alignment_inst: str = "",
        duration: float = 5.0,
        user_description: str = "",
        output_language: str = "English",
    ) -> str:
        if not raw_output:
            return ""
        final_compiled_prompt = PostProcessor.compile_final_prompt(
            raw_output,
            task_type,
            subject_defs,
            alignment_inst,
            duration=duration,
            user_description=user_description,
            output_language=output_language,
        )
        if len(final_compiled_prompt) > H3_MAX_CHARS:
            log_warning(f"Prompt exceeds H3 limit ({len(final_compiled_prompt)}/{H3_MAX_CHARS} chars).")
            final_compiled_prompt = final_compiled_prompt[:H3_MAX_CHARS]
        return final_compiled_prompt
