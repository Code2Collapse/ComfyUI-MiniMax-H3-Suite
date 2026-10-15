"""L8.31: the Promptor's Gemini provider (mmx_utils/promptor/provider_gemini.py) sends the API key in the
x-goog-api-key header, never in the URL, and no error text it returns can carry the key."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PACK = Path(__file__).resolve().parents[1]
if str(PACK) not in sys.path:
    sys.path.insert(0, str(PACK))

requests = pytest.importorskip("requests")

from mmx_utils.promptor import provider_gemini as pg  # noqa: E402

KEY = "AIzaSy-test-key-0123456789"
BASE = "https://generativelanguage.googleapis.com/v1beta"


class _Resp:
    def __init__(self, status, body=None, text=""):
        self.status_code = status
        self.ok = 200 <= status < 300
        self._body = body or {}
        self.text = text

    def json(self):
        return self._body


def test_key_goes_in_the_header_not_the_url(monkeypatch):
    seen = {}

    def fake_post(url, headers=None, json=None, timeout=None):
        seen.update(url=url, headers=dict(headers or {}))
        return _Resp(200, {"candidates": [{"content": {"parts": [{"text": "ok"}]}}]})

    monkeypatch.setattr(pg.requests, "post", fake_post)
    r = pg.GeminiProvider(BASE, api_key=KEY, model="gemini-2.5-flash").chat("sys", "hi")
    assert r.content == "ok"
    assert KEY not in seen["url"] and "key=" not in seen["url"]
    assert seen["headers"]["x-goog-api-key"] == KEY


def test_connection_errors_never_carry_the_key(monkeypatch):
    def boom(url, headers=None, json=None, timeout=None):
        # requests puts the URL in its exception text; a key would come along if it were in the URL or echoed back
        raise requests.exceptions.ConnectionError(f"HTTPSConnectionPool: Max retries exceeded with url: {url}?key={KEY}")

    monkeypatch.setattr(pg.requests, "post", boom)
    r = pg.GeminiProvider(BASE, api_key=KEY, model="gemini-2.5-flash").chat("sys", "hi")
    assert r.error and KEY not in r.error and "***" in r.error


def test_http_error_body_is_scrubbed(monkeypatch):
    monkeypatch.setattr(pg.requests, "post",
                        lambda *a, **k: _Resp(400, text=f'{{"error": "API key {KEY} not valid"}}'))
    r = pg.GeminiProvider(BASE, api_key=KEY, model="gemini-2.5-flash").chat("sys", "hi")
    assert r.error.startswith("HTTP 400") and KEY not in r.error


def test_availability_check_uses_the_header(monkeypatch):
    seen = {}

    def fake_get(url, headers=None, timeout=None):
        seen.update(url=url, headers=dict(headers or {}))
        return _Resp(200)

    monkeypatch.setattr(pg.requests, "get", fake_get)
    assert pg.GeminiProvider(BASE, api_key=KEY).is_available() is True
    assert KEY not in seen["url"] and seen["headers"]["x-goog-api-key"] == KEY
