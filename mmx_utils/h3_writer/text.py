# PORTED FROM: third_party/ComfyUI-MiniMaxH3-Prompt-Writer/backend/text_normalization.py (MIT, H3 Prompt Writer contributors)
# PORTED FROM: third_party/ComfyUI-MiniMaxH3-Prompt-Writer/backend/context.py (MIT, H3 Prompt Writer contributors)
"""Text helpers shared by the H3 Prompt Writer modules."""

from __future__ import annotations

import math
import re

REPLACEMENT_CHARACTER = "�"
CONTEXT_SAFETY_TOKENS = 512
CHAT_TEMPLATE_OVERHEAD_TOKENS = 384
ESTIMATED_VISUAL_TOKENS = 280


def normalize_unicode_text(value: str) -> str:
    """Replace invalid UTF-16 surrogate code units without changing valid Unicode text."""
    if not any(0xD800 <= ord(character) <= 0xDFFF for character in value):
        return value

    normalized: list[str] = []
    index = 0
    while index < len(value):
        codepoint = ord(value[index])
        if 0xD800 <= codepoint <= 0xDBFF:
            if index + 1 < len(value):
                low = ord(value[index + 1])
                if 0xDC00 <= low <= 0xDFFF:
                    normalized.append(chr(0x10000 + ((codepoint - 0xD800) << 10) + (low - 0xDC00)))
                    index += 2
                    continue
            normalized.append(REPLACEMENT_CHARACTER)
        elif 0xDC00 <= codepoint <= 0xDFFF:
            normalized.append(REPLACEMENT_CHARACTER)
        else:
            normalized.append(value[index])
        index += 1
    return "".join(normalized)


def estimate_text_tokens(text: str) -> int:
    """Conservative fallback used when an exact lightweight tokenizer is unavailable."""
    encoded = text.encode("utf-8")
    return math.ceil(max(len(text) / 3.0, len(encoded) / 3.0))


_THINK = re.compile(r"(?is)<think>.*?</think>")


def strip_thinking(text: str) -> str:
    """Drop reasoning blocks some local models emit before the answer (ours, not ported)."""
    text = _THINK.sub("", text or "")
    if "</think>" in text:          # an opening tag consumed by the chat template
        text = text.split("</think>", 1)[1]
    return text.strip()
