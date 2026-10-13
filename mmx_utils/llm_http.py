"""Out-of-process LLM backends for H3 Prompt Writer v2 (C2C L8.24): Ollama, any OpenAI-compatible server (llama-server,
LM Studio, vLLM) and the OpenAI / Anthropic / Gemini APIs, over stdlib HTTP.

Rules this module keeps (design: C2C docs/research/l824_prompt_writer_v2.md):
- no native inference runtime in the ComfyUI process - a crashing server only fails the request;
- an API key travels only in a request header and is scrubbed from every error string;
- a short preflight (reachable, model present, key set) before any long request;
- bounded time: timeout per request, a fixed number of retries on 429 / 5xx / network errors.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable

TEMPLATE = "template (no LLM)"
OLLAMA = "ollama"
OPENAI_COMPAT = "openai-compatible server"
ANTHROPIC = "anthropic api"
GEMINI = "gemini api"
OPENAI = "openai api"
BACKENDS: tuple[str, ...] = (TEMPLATE, OLLAMA, OPENAI_COMPAT, ANTHROPIC, GEMINI, OPENAI)

# (default base URL, default model, default key variable). A blank model means: the server's own (Ollama /
# OpenAI-compatible list their models) or "you must name one" (cloud APIs - model IDs change too often to pin here,
# except Anthropic's current flagship).
DEFAULTS: dict[str, tuple[str, str, str]] = {
    OLLAMA: ("http://127.0.0.1:11434", "", ""),
    OPENAI_COMPAT: ("http://127.0.0.1:8080/v1", "", ""),
    ANTHROPIC: ("https://api.anthropic.com/v1", "claude-opus-5-5", "ANTHROPIC_API_KEY"),
    GEMINI: ("https://generativelanguage.googleapis.com/v1beta", "", "GEMINI_API_KEY"),
    OPENAI: ("https://api.openai.com/v1", "", "OPENAI_API_KEY"),
}
OFFLINE_ENV = "MMX_PROMPT_WRITER_OFFLINE"
PREFLIGHT_TIMEOUT = 3.0


class BackendError(Exception):
    """A failure the node can explain in one sentence. `retry` marks transient ones."""

    def __init__(self, message: str, retry: bool = False):
        super().__init__(message)
        self.message = message
        self.retry = retry


@dataclass
class BackendConfig:
    backend: str
    base_url: str = ""
    model: str = ""
    api_key_env: str = ""
    timeout_s: float = 90.0
    temperature: float = 0.4
    max_tokens: int = 1200
    log: list[str] = field(default_factory=list)

    def resolved(self) -> "BackendConfig":
        url, model, key_env = DEFAULTS.get(self.backend, ("", "", ""))
        return BackendConfig(self.backend, (self.base_url or url).rstrip("/"), self.model or model,
                             self.api_key_env or key_env, self.timeout_s, self.temperature, self.max_tokens, self.log)

    @property
    def key(self) -> str:
        return os.environ.get(self.api_key_env, "") if self.api_key_env else ""


def offline() -> bool:
    return os.environ.get(OFFLINE_ENV, "").strip().lower() in ("1", "true", "yes", "on")


def scrub(text: str, *secrets: str) -> str:
    text = str(text)
    for s in secrets:
        if s and len(s) >= 4:
            text = text.replace(s, "***")
    return text


def _request(method: str, url: str, headers: dict[str, str], body: Any, timeout: float, secret: str) -> Any:
    data = None if body is None else json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, method=method, headers={"Content-Type": "application/json",
                                                                          "Accept": "application/json", **headers})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read()
    except urllib.error.HTTPError as e:
        detail = ""
        try:
            detail = e.read().decode("utf-8", "replace")[:300]
        except Exception:  # noqa: BLE001 - the status is what matters
            pass
        transient = e.code == 429 or e.code >= 500
        hint = {401: "the key was rejected", 403: "the key has no access", 404: "not found - check the URL / model"}
        raise BackendError(scrub(f"HTTP {e.code} from {url}: {hint.get(e.code, detail or e.reason)}", secret),
                           retry=transient) from None
    except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
        reason = getattr(e, "reason", e)
        raise BackendError(scrub(f"cannot reach {url} ({reason})", secret), retry=True) from None
    try:
        return json.loads(raw.decode("utf-8"))
    except ValueError:
        raise BackendError(f"{url} did not answer with JSON") from None


def _headers(cfg: BackendConfig) -> dict[str, str]:
    key = cfg.key
    if cfg.backend == ANTHROPIC:
        return {"x-api-key": key, "anthropic-version": "2023-06-01"}
    if cfg.backend == GEMINI:
        return {"x-goog-api-key": key}
    return {"Authorization": f"Bearer {key}"} if key else {}


def preflight(cfg: BackendConfig) -> BackendConfig:
    """Fast checks before a long request. Returns the config with the model filled in; raises BackendError."""
    cfg = cfg.resolved()
    if cfg.backend not in BACKENDS or cfg.backend == TEMPLATE:
        raise BackendError(f"unknown backend {cfg.backend!r}")
    if not cfg.base_url:
        raise BackendError("no base URL")
    secret = cfg.key
    if cfg.backend in (ANTHROPIC, GEMINI, OPENAI):
        if not cfg.api_key_env:
            raise BackendError("name the environment variable that holds the API key (api_key_env)")
        if not secret:
            raise BackendError(f"the environment variable {cfg.api_key_env} is not set - set it where ComfyUI starts")
        if not cfg.model:
            raise BackendError(f"name the {cfg.backend} model to use (model)")
    if cfg.backend == OLLAMA:
        tags = _request("GET", f"{cfg.base_url}/api/tags", {}, None, PREFLIGHT_TIMEOUT, secret)
        names = [m.get("name", "") for m in (tags or {}).get("models", [])]
        if not names:
            raise BackendError(f"Ollama at {cfg.base_url} has no models - `ollama pull <model>` first")
        if not cfg.model:
            cfg.model = names[0]
        elif cfg.model not in names and f"{cfg.model}:latest" not in names:
            raise BackendError(f"Ollama has no model {cfg.model!r}; it has: {', '.join(names[:8])}")
    elif cfg.backend in (OPENAI_COMPAT, OPENAI):
        listing = _request("GET", f"{cfg.base_url}/models", _headers(cfg), None, PREFLIGHT_TIMEOUT, secret)
        ids = [m.get("id", "") for m in (listing or {}).get("data", [])]
        if not cfg.model:
            if not ids:
                raise BackendError(f"{cfg.base_url} lists no models - name one (model)")
            cfg.model = ids[0]
        elif cfg.backend == OPENAI_COMPAT and ids and cfg.model not in ids:
            raise BackendError(f"the server has no model {cfg.model!r}; it has: {', '.join(ids[:8])}")
    elif cfg.backend == ANTHROPIC:
        _request("GET", f"{cfg.base_url}/models?limit=1", _headers(cfg), None, PREFLIGHT_TIMEOUT, secret)
    elif cfg.backend == GEMINI:
        _request("GET", f"{cfg.base_url}/models?pageSize=1", _headers(cfg), None, PREFLIGHT_TIMEOUT, secret)
    return cfg


def _chat_once(cfg: BackendConfig, system: str, user: str) -> str:
    secret, h, t = cfg.key, _headers(cfg), cfg.timeout_s
    if cfg.backend == OLLAMA:
        r = _request("POST", f"{cfg.base_url}/api/chat", h, {
            "model": cfg.model, "stream": False, "format": "json",
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "options": {"temperature": cfg.temperature, "num_predict": cfg.max_tokens}}, t, secret)
        return str(((r or {}).get("message") or {}).get("content") or "")
    if cfg.backend in (OPENAI_COMPAT, OPENAI):
        r = _request("POST", f"{cfg.base_url}/chat/completions", h, {
            "model": cfg.model, "temperature": cfg.temperature, "max_tokens": cfg.max_tokens,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}, t, secret)
        msg = (((r or {}).get("choices") or [{}])[0] or {}).get("message") or {}
        return str(msg.get("content") or "")
    if cfg.backend == ANTHROPIC:
        r = _request("POST", f"{cfg.base_url}/messages", h, {
            "model": cfg.model, "max_tokens": cfg.max_tokens, "temperature": cfg.temperature, "system": system,
            "messages": [{"role": "user", "content": user}]}, t, secret)
        return "".join(b.get("text", "") for b in (r or {}).get("content", []) if b.get("type") == "text")
    if cfg.backend == GEMINI:
        r = _request("POST", f"{cfg.base_url}/models/{cfg.model}:generateContent", h, {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": {"temperature": cfg.temperature, "maxOutputTokens": cfg.max_tokens,
                                 "responseMimeType": "application/json"}}, t, secret)
        cand = ((r or {}).get("candidates") or [{}])[0] or {}
        return "".join(p.get("text", "") for p in (cand.get("content") or {}).get("parts", []))
    raise BackendError(f"unknown backend {cfg.backend!r}")


def chat(cfg: BackendConfig, system: str, user: str, retries: int = 1,
         check_interrupt: Callable[[], None] | None = None, sleep: Callable[[float], None] = time.sleep) -> str:
    """One reply. Retries transient failures `retries` times (1 s, 2 s, ...); `check_interrupt` may raise."""
    last: BackendError | None = None
    for attempt in range(max(0, int(retries)) + 1):
        if check_interrupt is not None:
            check_interrupt()
        try:
            text = _chat_once(cfg, system, user)
            if not text.strip():
                raise BackendError("the model returned an empty reply", retry=True)
            return text
        except BackendError as e:
            last = e
            cfg.log.append(f"attempt {attempt + 1}: {e.message}")
            if not e.retry or attempt == retries:
                break
            sleep(float(attempt + 1))
    raise last or BackendError("no reply")
