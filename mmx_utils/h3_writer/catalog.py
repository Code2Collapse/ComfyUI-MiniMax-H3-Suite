"""The curated model list behind the node's model_preset dropdown (models.json next to this file).

The GGUF entries come from the H3 Prompt Writer catalog (third_party/ComfyUI-MiniMaxH3-Prompt-Writer/models.json, MIT,
H3 Prompt Writer contributors); the Hugging Face entries were added here. The list is a convenience: model_id takes any
Hugging Face repo id, local folder or .gguf file.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from typing import Any

CUSTOM = "custom (use model_id)"


@lru_cache(maxsize=1)
def models() -> tuple[dict[str, Any], ...]:
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models.json")
    try:
        with open(path, encoding="utf-8") as fh:
            return tuple(json.load(fh).get("models", []))
    except Exception:  # noqa: BLE001 - a broken list must not cost the node; custom model_id still works
        return ()


def preset_labels() -> tuple[str, ...]:
    return (CUSTOM,) + tuple(m["display_name"] for m in models())


def lookup(label: str) -> dict[str, Any] | None:
    if not label or label == CUSTOM:
        return None
    return next((m for m in models() if m.get("display_name") == label), None)
