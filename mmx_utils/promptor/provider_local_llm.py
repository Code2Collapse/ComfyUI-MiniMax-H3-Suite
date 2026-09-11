# PORTED FROM: 1038lab/ComfyUI-Minimax-H3-Promptor :: py/provider_local_llm.py
# Comfyui-Minimax-H3-Promptor — author 1038lab, GPL-3.0

from __future__ import annotations

from .provider_base import LLMProvider, LLMResponse


class LocalLLMProvider(LLMProvider):
    """Local Multimodal / Text LLM Provider (ComfyUI-QwenVL bridge)."""

    def __init__(
        self,
        api_base: str = "",
        api_key: str = "",
        model: str = "local-model",
        disable_thinking: bool = True,
        **kwargs,
    ):
        super().__init__(api_base=api_base, api_key=api_key, model=model)
        self.type = "qwenvl"
        self.disable_thinking = disable_thinking
        self.extra_kwargs = kwargs

    def chat(
        self,
        system_prompt: str,
        user_message: str,
        base64_images=None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        model: str | None = None,
    ) -> LLMResponse:
        target_model = model or self.model
        return LLMResponse(
            error=(
                "[Local LLM] ComfyUI-QwenVL bridge not available in this environment. "
                f"Configure a cloud provider in {self.__class__.__name__} config.json instead."
            ),
            model=target_model,
        )

    def is_available(self) -> bool:
        return False

    def get_available_models(self) -> list[dict]:
        return []


QwenVLProvider = LocalLLMProvider
