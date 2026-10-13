"""mmx_utils/llm_http.py - the out-of-process backends of H3 Prompt Writer v2 (C2C L8.24).

CPU-only, no network: a stdlib HTTP server on 127.0.0.1 plays Ollama / an OpenAI-compatible server / the cloud APIs.
"""

from __future__ import annotations

import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

PACK = Path(__file__).resolve().parents[1]
if str(PACK) not in sys.path:
    sys.path.insert(0, str(PACK))

from mmx_utils import llm_http as L  # noqa: E402

SECRET = "sk-test-SECRET-1234567890"


class _Fake:
    """Scripted server: routes -> list of (status, json) answered in order (last one repeats)."""

    def __init__(self):
        self.routes: dict[tuple[str, str], list[tuple[int, object]]] = {}
        self.seen: list[dict] = []
        fake = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _answer(self, method):
                n = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(n) or b"null") if n else None
                path = self.path.split("?")[0]
                fake.seen.append({"method": method, "path": self.path, "headers": dict(self.headers), "body": body})
                script = fake.routes.get((method, path)) or [(404, {"error": "no route"})]
                status, payload = script.pop(0) if len(script) > 1 else script[0]
                data = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                self._answer("GET")

            def do_POST(self):
                self._answer("POST")

        self.server = HTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.server.server_port}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()


@pytest.fixture()
def fake():
    f = _Fake()
    yield f
    f.close()


def test_ollama_preflight_picks_the_first_model_and_chat_sends_json_format(fake):
    # INVARIANT: blank model = the server's first; the chat asks Ollama for JSON and returns message.content.
    fake.routes[("GET", "/api/tags")] = [(200, {"models": [{"name": "qwen3:8b"}, {"name": "llama3.2:latest"}]})]
    fake.routes[("POST", "/api/chat")] = [(200, {"message": {"content": '{"scene_style": "dusk"}'}})]
    cfg = L.preflight(L.BackendConfig(L.OLLAMA, base_url=fake.url))
    assert cfg.model == "qwen3:8b"
    assert L.chat(cfg, "sys", "user") == '{"scene_style": "dusk"}'
    body = fake.seen[-1]["body"]
    assert body["format"] == "json" and body["stream"] is False and body["model"] == "qwen3:8b"


def test_ollama_missing_model_names_what_is_there(fake):
    # INVARIANT: a model the server lacks fails the PREFLIGHT, naming the models it does have.
    fake.routes[("GET", "/api/tags")] = [(200, {"models": [{"name": "qwen3:8b"}]})]
    with pytest.raises(L.BackendError, match="qwen3:8b"):
        L.preflight(L.BackendConfig(L.OLLAMA, base_url=fake.url, model="nope"))


def test_unreachable_server_fails_fast_with_a_plain_reason():
    # INVARIANT: nothing listening -> BackendError naming the URL, within the preflight budget.
    with pytest.raises(L.BackendError, match="cannot reach"):
        L.preflight(L.BackendConfig(L.OLLAMA, base_url="http://127.0.0.1:9"))


def test_openai_compatible_retries_a_503_then_succeeds(fake, monkeypatch):
    # INVARIANT: 5xx is transient - retried (`retries` times, 1 s apart here stubbed), then the reply comes through.
    fake.routes[("GET", "/v1/models")] = [(200, {"data": [{"id": "local-gguf"}]})]
    fake.routes[("POST", "/v1/chat/completions")] = [
        (503, {"error": "loading"}), (200, {"choices": [{"message": {"content": "{}"}}]})]
    slept = []
    cfg = L.preflight(L.BackendConfig(L.OPENAI_COMPAT, base_url=fake.url + "/v1"))
    assert cfg.model == "local-gguf"
    assert L.chat(cfg, "s", "u", retries=1, sleep=slept.append) == "{}"
    assert slept == [1.0] and any("HTTP 503" in line for line in cfg.log)


def test_a_400_is_not_retried(fake):
    # INVARIANT: a client error is final - no retry, the reason surfaces.
    fake.routes[("GET", "/v1/models")] = [(200, {"data": [{"id": "m"}]})]
    fake.routes[("POST", "/v1/chat/completions")] = [(400, {"error": "bad"}), (200, {"choices": []})]
    cfg = L.preflight(L.BackendConfig(L.OPENAI_COMPAT, base_url=fake.url + "/v1"))
    with pytest.raises(L.BackendError, match="HTTP 400"):
        L.chat(cfg, "s", "u", retries=3, sleep=lambda s: None)
    assert sum(1 for r in fake.seen if r["method"] == "POST") == 1


def test_interrupt_is_honoured_between_attempts(fake):
    # INVARIANT: ComfyUI's interrupt check runs before every attempt and may stop the loop.
    fake.routes[("POST", "/v1/chat/completions")] = [(503, {})]
    cfg = L.BackendConfig(L.OPENAI_COMPAT, base_url=fake.url + "/v1", model="m").resolved()
    calls = []

    def check():
        calls.append(1)
        if len(calls) > 1:
            raise InterruptedError("stop")

    with pytest.raises(InterruptedError):
        L.chat(cfg, "s", "u", retries=3, check_interrupt=check, sleep=lambda s: None)


@pytest.mark.parametrize("backend,route,answer,header", [
    (L.ANTHROPIC, "/v1/messages", {"content": [{"type": "text", "text": "{}"}]}, "x-api-key"),
    (L.GEMINI, "/v1beta/models/g:generateContent", {"candidates": [{"content": {"parts": [{"text": "{}"}]}}]},
     "x-goog-api-key"),
    (L.OPENAI, "/v1/chat/completions", {"choices": [{"message": {"content": "{}"}}]}, "Authorization"),
])
def test_cloud_keys_travel_only_in_headers(fake, monkeypatch, backend, route, answer, header):
    # INVARIANT: the key is in a header, never in a URL or body (Gemini's ?key= form leaks into error text).
    monkeypatch.setenv("C2C_TEST_KEY", SECRET)
    base = fake.url + ("/v1beta" if backend == L.GEMINI else "/v1")
    for path in ("/v1/models", "/v1beta/models"):
        fake.routes[("GET", path)] = [(200, {"data": [{"id": "g"}], "models": []})]
    fake.routes[("POST", route)] = [(200, answer)]
    cfg = L.preflight(L.BackendConfig(backend, base_url=base, model="g", api_key_env="C2C_TEST_KEY"))
    assert L.chat(cfg, "s", "u") == "{}"
    for r in fake.seen:
        assert SECRET not in r["path"] and SECRET not in json.dumps(r["body"])
        hdr = {k.lower(): v for k, v in r["headers"].items()}
        assert SECRET in hdr.get(header.lower(), "")


def test_a_rejected_key_never_appears_in_the_error(fake, monkeypatch):
    # INVARIANT: even when the server echoes the key back, the BackendError text has it scrubbed.
    monkeypatch.setenv("C2C_TEST_KEY", SECRET)
    fake.routes[("GET", "/v1/models")] = [(401, {"error": f"invalid key {SECRET}"})]
    with pytest.raises(L.BackendError) as ei:
        L.preflight(L.BackendConfig(L.OPENAI, base_url=fake.url + "/v1", model="m", api_key_env="C2C_TEST_KEY"))
    assert SECRET not in ei.value.message and "rejected" in ei.value.message


def test_cloud_backend_without_its_key_says_which_variable(monkeypatch):
    # INVARIANT: an unset key fails before any request, naming the variable - not a value.
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(L.BackendError, match="ANTHROPIC_API_KEY"):
        L.preflight(L.BackendConfig(L.ANTHROPIC))


def test_no_native_runtime_is_imported():
    # INVARIANT: the writer's modules never import llama.cpp (or any in-process model runtime): the reported crash is
    # an illegal instruction inside the ComfyUI process, which no Python code can catch.
    import importlib

    for mod in ("mmx_utils.llm_http", "mmx_utils.prompt_writer"):
        src = (PACK / (mod.replace(".", "/") + ".py")).read_text(encoding="utf-8")
        assert "llama_cpp" not in src.replace("llama.cpp", "") and "import llama" not in src
        importlib.import_module(mod)
    assert "llama_cpp" not in sys.modules
