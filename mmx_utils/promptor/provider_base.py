# PORTED FROM: 1038lab/ComfyUI-Minimax-H3-Promptor :: py/provider_base.py
# Comfyui-Minimax-H3-Promptor — author 1038lab, GPL-3.0

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class LLMResponse:
    content: str = ""
    model: str = ""
    usage: dict = field(default_factory=dict)
    error: str | None = None

    @property
    def success(self) -> bool:
        return self.error is None and bool(self.content)


class LLMProvider(ABC):
    def __init__(self, api_base: str, api_key: str = "", model: str = ""):
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.model = model

    @abstractmethod
    def chat(
        self,
        system_prompt: str,
        user_message: str,
        base64_images: list[str] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        model: str | None = None,
    ) -> LLMResponse:
        ...

    @abstractmethod
    def is_available(self) -> bool:
        ...

    def get_model(self, override: str | None = None) -> str:
        return override or self.model
