# PORTED FROM: 1038lab/ComfyUI-Minimax-H3-Promptor :: py/provider_gemini.py
# Comfyui-Minimax-H3-Promptor — author 1038lab, GPL-3.0

from __future__ import annotations

import time

import requests

from .provider_base import LLMProvider, LLMResponse

REQUEST_TIMEOUT = 120
MAX_RETRIES = 1
RETRY_DELAY = 2.0


class GeminiProvider(LLMProvider):
    def __init__(self, api_base: str, api_key: str = "", model: str = "", disable_thinking: bool = True):
        super().__init__(api_base=api_base, api_key=api_key, model=model)
        self.disable_thinking = disable_thinking

    def chat(
        self,
        system_prompt: str,
        user_message: str,
        base64_images: list[str] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        model: str | None = None,
    ) -> LLMResponse:
        model_name = self.get_model(model)
        url = f"{self.api_base}/models/{model_name}:generateContent?key={self.api_key}"
        parts = []
        if base64_images and len(base64_images) > 0:
            if isinstance(base64_images[0], dict):
                for item in base64_images:
                    if "text" in item:
                        parts.append({"text": item["text"]})
                    elif "image" in item:
                        parts.append({"inline_data": {"mime_type": "image/jpeg", "data": item["image"]}})
            else:
                parts.append({"text": user_message})
                for img_b64 in base64_images:
                    parts.append({"inline_data": {"mime_type": "image/jpeg", "data": img_b64}})
        else:
            parts.append({"text": user_message})

        generation_config = {"temperature": temperature, "maxOutputTokens": max_tokens}
        if self.disable_thinking and any(k in str(model_name).lower() for k in ("thinking", "2.5", "3.", "flash")):
            generation_config["thinkingConfig"] = {"thinkingBudget": 0}

        payload = {
            "system_instruction": {"parts": [{"text": system_prompt}]},
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": generation_config,
        }

        for attempt in range(MAX_RETRIES + 1):
            try:
                response = requests.post(url, headers={"Content-Type": "application/json"}, json=payload, timeout=REQUEST_TIMEOUT)
                if response.status_code in (401, 403):
                    return LLMResponse(error="Gemini authentication failed.", model=model_name)
                if not response.ok:
                    return LLMResponse(error=f"HTTP {response.status_code}: {response.text[:200]}", model=model_name)
                data = response.json()
                candidates = data.get("candidates", [])
                if not candidates:
                    return LLMResponse(error="No candidates in Gemini response.", model=model_name)
                content_parts = candidates[0].get("content", {}).get("parts", [])
                content = "".join(p.get("text", "") for p in content_parts)
                return LLMResponse(content=content, model=model_name)
            except requests.exceptions.Timeout:
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_DELAY)
                    continue
                return LLMResponse(error="Gemini request timed out.", model=model_name)
            except Exception as e:
                return LLMResponse(error=f"Unexpected Gemini error: {e}", model=model_name)
        return LLMResponse(error="Max retries exceeded.", model=model_name)

    def is_available(self) -> bool:
        try:
            return requests.get(f"{self.api_base}/models?key={self.api_key}", timeout=10).ok
        except Exception:
            return False
