# PORTED FROM: 1038lab/ComfyUI-Minimax-H3-Promptor :: py/task_detector.py
# Comfyui-Minimax-H3-Promptor — author 1038lab, GPL-3.0
"""Detect the appropriate H3 task type from node inputs."""

TASK_TYPES = ["T2V", "I2V", "I2VA", "V2V", "V2VA", "A2V", "FL2VA", "Ref2VA", "L2VA"]

TASK_DESCRIPTIONS = {
    "T2V": "Text-to-Video (T2V)",
    "I2V": "Image-to-Video (I2V)",
    "I2VA": "Image-to-Video-Audio (I2VA)",
    "V2V": "Video-to-Video (V2V)",
    "V2VA": "Video-to-Video-Audio (V2VA)",
    "A2V": "Audio-to-Video (A2V)",
    "FL2VA": "First-and-Last-Frame-to-Video (FL2VA)",
    "Ref2VA": "Reference-to-Video-Audio (Ref2VA)",
    "L2VA": "Last-Frame-to-Video-Audio (L2VA)",
}

TASK_TYPE_OPTIONS = ["Auto"] + list(TASK_DESCRIPTIONS.values())


class TaskDetector:
    """Detect the appropriate H3 task type from node inputs."""

    @staticmethod
    def detect(
        image_count: int = 0,
        has_video: bool = False,
        has_audio: bool = False,
        user_override: str = "Auto",
    ) -> str:
        if user_override != "Auto":
            for short_code, desc in TASK_DESCRIPTIONS.items():
                if user_override == desc:
                    return short_code
            if user_override in TASK_TYPES:
                return user_override

        if has_video and image_count > 0:
            return "Ref2VA"
        if image_count >= 3:
            return "Ref2VA"
        if has_video and has_audio:
            return "V2VA"
        if has_video:
            return "V2V"
        if image_count == 2:
            return "FL2VA"
        if image_count == 1 and has_audio:
            return "I2VA"
        if image_count == 1:
            return "I2V"
        if has_audio:
            return "A2V"
        return "T2V"

    @staticmethod
    def get_task_description(task_type: str) -> str:
        descriptions = {
            "T2V": "Text-to-Video (no references)",
            "I2V": "Image-to-Video (single image)",
            "I2VA": "Image-to-Audio-Video (image + audio reference)",
            "V2V": "Video-to-Video (video reference)",
            "V2VA": "Video-to-Video-Audio (video + audio reference)",
            "A2V": "Audio-to-Video (audio reference)",
            "FL2VA": "First/Last Frame (two boundary images)",
            "Ref2VA": "Omni Reference (multi-modal references)",
            "L2VA": "Last Frame Anchor (end frame constraints)",
        }
        return descriptions.get(task_type, f"Unknown ({task_type})")
