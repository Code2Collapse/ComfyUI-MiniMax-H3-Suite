# PORTED FROM: 1038lab/ComfyUI-Minimax-H3-Promptor :: py/provider_claude.py
# Comfyui-Minimax-H3-Promptor — author 1038lab, GPL-3.0

from __future__ import annotations

import time

import requests

from .provider_base import LLMProvider, LLMResponse

REQUEST_TIMEOUT = 120
MAX_RETRIES = 1
RETRY_DELAY = 2.0
ANTHROPIC_VERSION = "2023-06-01"


class ClaudeProvider(LLMProvider):
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
        url = f"{self.api_base}/messages"
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": ANTHROPIC_VERSION,
        }
        if base64_images and len(base64_images) > 0:
            user_content = []
            if isinstance(base64_images[0], dict):
                for item in base64_images:
                    if "text" in item:
                        user_content.append({"type": "text", "text": item["text"]})
                    elif "image" in item:
                        user_content.append(
                            {
                                "type": "image",
                                "source": {"type": "base64", "media_type": "image/jpeg", "data": item["image"]},
                            }
                        )
            else:
                for img_b64 in base64_images:
                    user_content.append(
                        {
                            "type": "image",
                            "source": {"type": "base64", "media_type": "image/jpeg", "data": img_b64},
                        }
                    )
                user_content.append({"type": "text", "text": user_message})
        else:
            user_content = user_message

        payload = {
            "model": model_name,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_content}],
        }

        for attempt in range(MAX_RETRIES + 1):
            try:
                response = requests.post(url, headers=headers, json=payload, timeout=REQUEST_TIMEOUT)
                if not response.ok:
                    return LLMResponse(error=f"HTTP {response.status_code}: {response.text[:200]}", model=model_name)
                data = response.json()
                content = "".join(
                    block.get("text", "") for block in data.get("content", []) if block.get("type") == "text"
                )
                return LLMResponse(content=content, model=data.get("model", model_name))
            except requests.exceptions.Timeout:
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_DELAY)
                    continue
                return LLMResponse(error="Claude request timed out.", model=model_name)
            except Exception as e:
                return LLMResponse(error=f"Unexpected Claude error: {e}", model=model_name)
        return LLMResponse(error="Max retries exceeded.", model=model_name)

    def is_available(self) -> bool:
        try:
            return requests.get(self.api_base.replace("/v1", ""), timeout=10).status_code < 500
        except Exception:
            return False
