"""Prompt-model providers behind one small interface (ours).

    p = open_provider(settings, progress)      # may download / load a local model, or preflight a server
    p.capabilities() -> {"images": bool, "audio": bool}
    p.complete(messages, temperature, top_p, max_tokens, seed, check_interrupt)
        -> {"text", "finish_reason", "usage": {"prompt_tokens", "completion_tokens"}}
    p.close(unload)

The five HTTP providers reuse mmx_utils/llm_http for everything that touches a key: preflight, the L9.288 host rule
(an environment key only goes to loopback, the key's own service or MMX_PROMPT_WRITER_HOSTS), headers and scrubbing.
"""

from __future__ import annotations

import base64
import time
from dataclasses import dataclass
from typing import Any, Callable

from mmx_utils import llm_http as L

from .modes import WriterError
from .text import strip_thinking

TEMPLATE = "template (no LLM)"
HF_LOCAL = "huggingface transformers (local)"
GGUF_LOCAL = "gguf llama.cpp (local)"
PROVIDERS: tuple[str, ...] = (TEMPLATE, HF_LOCAL, GGUF_LOCAL, L.OLLAMA, L.OPENAI_COMPAT, L.OPENAI, L.ANTHROPIC,
                              L.GEMINI)
HTTP_PROVIDERS: tuple[str, ...] = (L.OLLAMA, L.OPENAI_COMPAT, L.OPENAI, L.ANTHROPIC, L.GEMINI)
LOCAL_PROVIDERS: tuple[str, ...] = (HF_LOCAL, GGUF_LOCAL)


@dataclass
class Settings:
    provider: str
    model_id: str = ""
    preset: dict | None = None
    base_url: str = ""
    api_key_env: str = ""
    timeout_s: float = 180.0
    retries: int = 1
    device: str = "auto"
    dtype: str = "auto"
    quantization: str = "none"
    trust_remote_code: bool = False
    gguf_gpu_layers: int = -1
    context_tokens: int = 16384
    downloads: str = "download if missing"


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def _text_of(parts: list[dict[str, Any]]) -> str:
    return "\n\n".join(p["text"] for p in parts if p["type"] == "text")


class HttpProvider:
    def __init__(self, settings: Settings, progress: Callable[[str], None], sleep: Callable[[float], None] = time.sleep):
        self.backend = settings.provider
        self.retries = max(0, int(settings.retries))
        self.sleep = sleep
        cfg = L.BackendConfig(self.backend, settings.base_url.strip(), settings.model_id.strip(),
                              settings.api_key_env.strip(), float(settings.timeout_s))
        try:
            self.cfg = L.preflight(cfg)
        except L.BackendError as e:
            raise WriterError(f"{self.backend}: {L.scrub(e.message, cfg.key)}") from None
        self.model = self.cfg.model
        self.name = f"{self.backend} ({self.model} @ {L._host(self.cfg.base_url) or self.cfg.base_url})"
        self._caps: dict[str, bool] | None = None
        progress(f"server: {self.name}")
        if self.cfg.api_key_env:
            progress(f"key: from ${self.cfg.api_key_env} ({'set' if self.cfg.key else 'not set'})")

    @property
    def log(self) -> list[str]:
        # cfg.key may append to cfg.log (a refused host), so read it once, then copy the list and drop repeats
        secret = self.cfg.key
        return [L.scrub(line, secret) for line in dict.fromkeys(list(self.cfg.log))]

    def capabilities(self) -> dict[str, bool]:
        if self._caps is not None:
            return self._caps
        b, m = self.backend, (self.model or "").lower()
        if b == L.OLLAMA:
            caps = {"images": False, "audio": False}
            try:
                info = L._request("POST", f"{self.cfg.base_url}/api/show", {}, {"model": self.model}, L.PREFLIGHT_TIMEOUT,
                                  self.cfg.key)
                caps["images"] = "vision" in [str(c).lower() for c in (info or {}).get("capabilities", [])]
            except L.BackendError:
                pass
        elif b == L.OPENAI_COMPAT:
            caps = {"images": True, "audio": True}          # unknown server: try, and explain a rejection in words
        elif b == L.OPENAI:
            caps = {"images": True, "audio": "audio" in m}
        elif b == L.ANTHROPIC:
            caps = {"images": True, "audio": False}
        else:                                               # gemini
            caps = {"images": True, "audio": True}
        self._caps = caps
        return caps

    # -- payloads ---------------------------------------------------------------------------------------------------
    def _payload(self, messages: list[dict[str, Any]], temperature: float, top_p: float, max_tokens: int,
                 seed: int | None) -> tuple[str, dict[str, Any]]:
        b, base = self.backend, self.cfg.base_url
        system = "\n\n".join(_text_of(m["parts"]) for m in messages if m["role"] == "system")
        convo = [m for m in messages if m["role"] != "system"]
        if b == L.OLLAMA:
            msgs = [{"role": "system", "content": system}] if system else []
            for m in convo:
                msgs.append({"role": m["role"], "content": _text_of(m["parts"]),
                             **({"images": imgs} if (imgs := [_b64(p["data"]) for p in m["parts"]
                                                              if p["type"] == "image"]) else {})})
            opts: dict[str, Any] = {"temperature": temperature, "top_p": top_p, "num_predict": max_tokens}
            if seed is not None:
                opts["seed"] = int(seed)
            return f"{base}/api/chat", {"model": self.model, "stream": False, "messages": msgs, "options": opts}
        if b in (L.OPENAI_COMPAT, L.OPENAI):
            msgs = [{"role": "system", "content": system}] if system else []
            for m in convo:
                content = []
                for p in m["parts"]:
                    if p["type"] == "text":
                        content.append({"type": "text", "text": p["text"]})
                    elif p["type"] == "image":
                        content.append({"type": "image_url",
                                        "image_url": {"url": f"data:{p['mime']};base64,{_b64(p['data'])}"}})
                    elif p["type"] == "audio":
                        content.append({"type": "input_audio", "input_audio": {"data": _b64(p["data"]), "format": "wav"}})
                if m["role"] == "assistant":
                    content = _text_of(m["parts"])
                msgs.append({"role": m["role"], "content": content})
            body: dict[str, Any] = {"model": self.model, "messages": msgs, "temperature": temperature, "top_p": top_p,
                                    "max_tokens": max_tokens}
            if seed is not None:
                body["seed"] = int(seed)
            return f"{base}/chat/completions", body
        if b == L.ANTHROPIC:
            msgs = []
            for m in convo:
                content = []
                for p in m["parts"]:
                    if p["type"] == "text":
                        content.append({"type": "text", "text": p["text"]})
                    elif p["type"] == "image":
                        content.append({"type": "image", "source": {"type": "base64", "media_type": p["mime"],
                                                                    "data": _b64(p["data"])}})
                msgs.append({"role": m["role"], "content": content})
            return f"{base}/messages", {"model": self.model, "max_tokens": max_tokens, "temperature": min(1.0, temperature),
                                        "system": system, "messages": msgs}
        if b == L.GEMINI:
            contents = []
            for m in convo:
                parts = []
                for p in m["parts"]:
                    if p["type"] == "text":
                        parts.append({"text": p["text"]})
                    else:
                        parts.append({"inline_data": {"mime_type": p["mime"], "data": _b64(p["data"])}})
                contents.append({"role": "model" if m["role"] == "assistant" else "user", "parts": parts})
            gen: dict[str, Any] = {"temperature": temperature, "topP": top_p, "maxOutputTokens": max_tokens}
            if seed is not None:
                gen["seed"] = int(seed)
            return f"{base}/models/{self.model}:generateContent", {
                "systemInstruction": {"parts": [{"text": system}]}, "contents": contents, "generationConfig": gen}
        raise WriterError(f"unknown provider {b!r}")

    def _parse(self, r: Any) -> dict[str, Any]:
        b, r = self.backend, r or {}
        if b == L.OLLAMA:
            text = str((r.get("message") or {}).get("content") or "")
            finish = "length" if r.get("done_reason") == "length" else "stop"
            usage = {"prompt_tokens": int(r.get("prompt_eval_count") or 0),
                     "completion_tokens": int(r.get("eval_count") or 0)}
        elif b in (L.OPENAI_COMPAT, L.OPENAI):
            choice = (r.get("choices") or [{}])[0] or {}
            msg = choice.get("message") or {}
            content = msg.get("content") or ""
            if isinstance(content, list):
                content = "".join(c.get("text", "") for c in content if isinstance(c, dict))
            text, finish = str(content), str(choice.get("finish_reason") or "stop")
            u = r.get("usage") or {}
            usage = {"prompt_tokens": int(u.get("prompt_tokens") or 0),
                     "completion_tokens": int(u.get("completion_tokens") or 0)}
        elif b == L.ANTHROPIC:
            text = "".join(c.get("text", "") for c in r.get("content", []) if c.get("type") == "text")
            finish = "length" if r.get("stop_reason") == "max_tokens" else "stop"
            u = r.get("usage") or {}
            usage = {"prompt_tokens": int(u.get("input_tokens") or 0), "completion_tokens": int(u.get("output_tokens") or 0)}
        else:
            cand = (r.get("candidates") or [{}])[0] or {}
            text = "".join(p.get("text", "") for p in (cand.get("content") or {}).get("parts", []))
            finish = "length" if cand.get("finishReason") == "MAX_TOKENS" else "stop"
            u = r.get("usageMetadata") or {}
            usage = {"prompt_tokens": int(u.get("promptTokenCount") or 0),
                     "completion_tokens": int(u.get("candidatesTokenCount") or 0)}
        return {"text": strip_thinking(text), "finish_reason": finish, "usage": usage}

    def complete(self, messages: list[dict[str, Any]], temperature: float = 1.0, top_p: float = 0.95,
                 max_tokens: int = 2048, seed: int | None = None,
                 check_interrupt: Callable[[], None] | None = None) -> dict[str, Any]:
        url, body = self._payload(messages, temperature, top_p, max_tokens, seed)
        has_media = any(p["type"] != "text" for m in messages for p in m["parts"])
        key = self.cfg.key
        timeout = max(float(self.cfg.timeout_s), 5.0)
        last: L.BackendError | None = None
        for attempt in range(self.retries + 1):
            if check_interrupt is not None:
                check_interrupt()
            try:
                return self._parse(L._request("POST", url, L._headers(self.cfg), body, timeout, key))
            except L.BackendError as e:
                last = e
                self.cfg.log.append(f"attempt {attempt + 1}: {e.message}")
                if not e.retry or attempt == self.retries:
                    break
                self.sleep(float(attempt + 1))
        msg = L.scrub(last.message if last else "no reply", key)
        if has_media and msg.startswith("HTTP 4"):
            msg += (f". The server rejected the request with attached media: {self.model} may not accept images or "
                    f"audio. Pick a vision model, or set media_to_model to 'labels only (send no media)'")
        raise WriterError(f"{self.backend}: {msg}")

    def close(self, unload: bool = True) -> None:
        return None


def open_provider(settings: Settings, progress: Callable[[str], None]):
    if settings.provider in HTTP_PROVIDERS:
        return HttpProvider(settings, progress)
    if settings.provider == HF_LOCAL:
        from .hf_local import HFProvider
        return HFProvider(settings, progress)
    if settings.provider == GGUF_LOCAL:
        from .gguf_local import GGUFProvider
        return GGUFProvider(settings, progress)
    raise WriterError(f"Unknown provider {settings.provider!r}. Pick one of: {', '.join(PROVIDERS)}.")
