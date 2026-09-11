# PORTED FROM: 1038lab/ComfyUI-Minimax-H3-Promptor :: py/provider_ollama.py
# Comfyui-Minimax-H3-Promptor — author 1038lab, GPL-3.0

from __future__ import annotations

import time

import requests

from .provider_base import LLMProvider, LLMResponse

REQUEST_TIMEOUT = 300
MAX_RETRIES = 1
RETRY_DELAY = 3.0


class OllamaProvider(LLMProvider):
    def __init__(self, api_base: str = "http://localhost:11434", **kwargs):
        super().__init__(api_base=api_base, api_key="", **kwargs)

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
        url = f"{self.api_base}/api/chat"
        messages = []
        if base64_images and len(base64_images) > 0:
            user_payload = {"role": "user", "content": ""}
            if isinstance(base64_images[0], dict):
                text_parts, image_parts = [], []
                for item in base64_images:
                    if "text" in item:
                        text_parts.append(item["text"])
                    elif "image" in item:
                        image_parts.append(item["image"])
                user_payload["content"] = "\n\n".join(text_parts)
                if image_parts:
                    user_payload["images"] = image_parts
            else:
                user_payload["content"] = user_message
                user_payload["images"] = base64_images
            if system_prompt:
                user_payload["content"] = f"{system_prompt}\n\n{user_payload['content']}"
            messages.append(user_payload)
        else:
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": user_message})

        payload = {
            "model": model_name,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens, "num_ctx": max(max_tokens * 2, 8192)},
        }

        for attempt in range(MAX_RETRIES + 1):
            try:
                response = requests.post(url, json=payload, timeout=REQUEST_TIMEOUT)
                if response.status_code == 404:
                    return LLMResponse(error=f"Model '{model_name}' not found in Ollama.", model=model_name)
                if not response.ok:
                    return LLMResponse(error=f"Ollama HTTP {response.status_code}: {response.text[:200]}", model=model_name)
                data = response.json()
                msg = data.get("message", {})
                content = msg.get("content", "") or msg.get("thinking", "") or msg.get("reasoning_content", "")
                if not content.strip():
                    return LLMResponse(error=f"Model '{model_name}' returned an empty response.", model=model_name)
                return LLMResponse(content=content, model=data.get("model", model_name))
            except requests.exceptions.Timeout:
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_DELAY)
                    continue
                return LLMResponse(error=f"Ollama timed out after {REQUEST_TIMEOUT}s.", model=model_name)
            except requests.exceptions.ConnectionError:
                return LLMResponse(error=f"Cannot connect to Ollama at {self.api_base}.", model=model_name)
            except Exception as e:
                return LLMResponse(error=f"Unexpected Ollama error: {e}", model=model_name)
        return LLMResponse(error="Max retries exceeded.", model=model_name)

    def is_available(self) -> bool:
        try:
            return requests.get(f"{self.api_base}/api/tags", timeout=5).ok
        except Exception:
            return False
