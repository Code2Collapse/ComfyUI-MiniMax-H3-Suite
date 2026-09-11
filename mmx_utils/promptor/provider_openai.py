# PORTED FROM: 1038lab/ComfyUI-Minimax-H3-Promptor :: py/provider_openai.py
# Comfyui-Minimax-H3-Promptor — author 1038lab, GPL-3.0

from __future__ import annotations

import time

import requests

from .provider_base import LLMProvider, LLMResponse
from .utils import log_debug, log_warning

REQUEST_TIMEOUT = 120
MAX_RETRIES = 1
RETRY_DELAY = 2.0


class OpenAIProvider(LLMProvider):
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
        url = f"{self.api_base}/chat/completions"
        headers = {"Content-Type": "application/json", "Connection": "close"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        if base64_images and len(base64_images) > 0:
            if isinstance(base64_images[0], dict):
                user_content = []
                for item in base64_images:
                    if "text" in item:
                        user_content.append({"type": "text", "text": item["text"]})
                    elif "image" in item:
                        user_content.append(
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{item['image']}"},
                            }
                        )
            else:
                user_content = [{"type": "text", "text": user_message}]
                for img_b64 in base64_images:
                    user_content.append(
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
                    )
        else:
            user_content = user_message

        payload = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        api_host = self.api_base.lower()
        model_lower = str(model_name or "").lower()
        is_thinking_model = any(
            marker in model_lower
            for marker in ("qwen3", "deepseek", "glm-4.5", "glm-4.6", "glm-4.7", "glm-5", "hunyuan", "r1", "reasoner")
        )
        if self.disable_thinking and ("siliconflow" in api_host or is_thinking_model):
            payload["enable_thinking"] = False

        for attempt in range(MAX_RETRIES + 1):
            try:
                response = requests.post(url, headers=headers, json=payload, timeout=REQUEST_TIMEOUT)
                if response.status_code in (401, 403):
                    return LLMResponse(
                        error=f"Authentication failed (HTTP {response.status_code}). Check API key in config.json.",
                        model=model_name,
                    )
                if response.status_code == 429:
                    return LLMResponse(error="Rate limited.", model=model_name)
                if response.status_code >= 500:
                    if attempt < MAX_RETRIES:
                        time.sleep(RETRY_DELAY)
                        continue
                    return LLMResponse(error=f"Server error HTTP {response.status_code}", model=model_name)
                if not response.ok:
                    return LLMResponse(error=f"HTTP {response.status_code}: {response.text[:200]}", model=model_name)
                data = response.json()
                choices = data.get("choices", [])
                if not choices:
                    return LLMResponse(error="No choices in API response.", model=model_name)
                content = choices[0].get("message", {}).get("content", "")
                return LLMResponse(content=content, model=data.get("model", model_name), usage=data.get("usage", {}))
            except requests.exceptions.Timeout:
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_DELAY)
                    continue
                return LLMResponse(error=f"Request timed out after {REQUEST_TIMEOUT}s.", model=model_name)
            except requests.exceptions.ConnectionError:
                return LLMResponse(error=f"Cannot connect to {self.api_base}.", model=model_name)
            except Exception as e:
                return LLMResponse(error=f"Unexpected error: {e}", model=model_name)
        return LLMResponse(error="Max retries exceeded.", model=model_name)

    def is_available(self) -> bool:
        try:
            headers = {}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            return requests.get(f"{self.api_base}/models", headers=headers, timeout=10).ok
        except Exception:
            return False
