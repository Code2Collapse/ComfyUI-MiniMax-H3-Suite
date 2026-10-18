"""The Directors' cache key sees the media files their timeline names (ledger L2.05).

A timeline refers to uploaded clips/images by NAME. Re-uploading different media under the same
name (the chunked upload route writes in place) changed nothing ComfyUI hashes, so the Director
returned the old render. media_fingerprint adds (name, size, mtime_ns) of every referenced file.
"""
import json
import os
import time

import pytest

from mmx_nodes.director import minimax_media as media


@pytest.fixture
def input_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(media.folder_paths, "get_input_directory", lambda: str(tmp_path))
    return tmp_path


def _timeline(*names):
    return json.dumps({"tracks": [{"segments": [{"imageFile": n, "start": 0, "length": 24} for n in names]}],
                       "audio": [{"audioFile": "music.wav"}]})


def _write(path, data):
    path.write_bytes(data)
    st = path.stat()
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000))   # distinct mtime even on coarse clocks


def test_fingerprint_changes_when_a_named_file_is_replaced(input_dir):
    _write(input_dir / "clip.mp4", b"A" * 100)
    tl = _timeline("clip.mp4")
    first = media.media_fingerprint(tl)
    assert first == media.media_fingerprint(tl)                 # stable while nothing changes
    _write(input_dir / "clip.mp4", b"B" * 120)                  # re-upload, same name
    assert media.media_fingerprint(tl) != first


def test_same_size_rewrite_is_caught_by_mtime(input_dir):
    _write(input_dir / "clip.mp4", b"A" * 100)
    tl = _timeline("clip.mp4")
    first = media.media_fingerprint(tl)
    time.sleep(0.01)
    _write(input_dir / "clip.mp4", b"C" * 100)
    assert media.media_fingerprint(tl) != first


def test_every_reference_is_found_and_missing_files_are_stable(input_dir):
    _write(input_dir / "a.png", b"x")
    fp = media.media_fingerprint(_timeline("a.png", "gone.mp4"))
    refs = {r[0]: r for r in fp}
    assert set(refs) == {"a.png", "gone.mp4", "music.wav"}
    assert refs["gone.mp4"][1:] == (-1, -1)
    assert refs["a.png"][1] == 1


def test_workspace_subdir_upload_is_resolved(input_dir):
    sub = input_dir / media.WORKSPACE_SUBDIR
    sub.mkdir()
    _write(sub / "up.mp4", b"z" * 7)
    [(ref, size, _mt)] = [r for r in media.media_fingerprint(_timeline("up.mp4")) if r[0] == "up.mp4"]
    assert size == 7


def test_empty_and_bad_timelines_do_not_raise():
    assert media.media_fingerprint("") == ()
    assert media.media_fingerprint("{not json") == ()
    assert media.media_fingerprint({}) == ()


def test_directors_use_it_as_their_fingerprint(input_dir):
    from mmx_nodes.director.minimax_chain import MiniMaxH3DirectorChain
    from mmx_nodes.director.minimax_director import MiniMaxH3Director

    _write(input_dir / "clip.mp4", b"A" * 10)
    tl = _timeline("clip.mp4")
    for node in (MiniMaxH3Director, MiniMaxH3DirectorChain):
        assert node.fingerprint_inputs(timeline_data=tl, model=None) == media.media_fingerprint(tl)


def test_v1_adapter_forwards_is_changed_to_fingerprint_inputs(input_dir):
    """The All-in-One Director is a classic node wrapped for V3; core only calls the V3 hook."""
    from mmx_nodes.director.allinone import MuseMinimaxDirector, _resolve_path
    from mmx_nodes.director.v1_adapter import adapt

    adapted = adapt(MuseMinimaxDirector, node_id="T_AllInOne", display_name="t", category="t")
    _write(input_dir / "ref.png", b"q" * 5)
    tl = json.dumps({"characters": [{"file": "ref.png"}]})
    assert adapted.fingerprint_inputs(timeline_data=tl) == media.media_fingerprint(tl, resolve=_resolve_path)
    assert adapted.fingerprint_inputs(timeline_data=tl)[0][1] == 5


# -- seed reaches the model (A15 Phase 1.1: Enhance Prompt's seed only busted the cache) ---------------------------

class _FakeResp:
    status = 200

    def __init__(self, body):
        self._body = body

    async def json(self):
        return self._body

    async def text(self):
        return ""

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False


def _fake_session(sent):
    class _Session:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        def post(self, url, json=None, timeout=None):
            sent.append((url, json))
            if url.endswith("/api/generate"):
                return _FakeResp({"response": "ok"})
            return _FakeResp({"choices": [{"message": {"content": "ok"}}]})
    return _Session


@pytest.mark.parametrize("provider,url", [("ollama", "http://127.0.0.1:11434"), ("lmstudio", "http://127.0.0.1:1234")])
def test_vlm_generate_sends_the_seed(monkeypatch, provider, url):
    import asyncio

    import aiohttp
    sent = []
    monkeypatch.setattr(aiohttp, "ClientSession", _fake_session(sent))
    out = asyncio.run(media.vlm_generate([], "hi", provider, url, "m", max_tokens=10, seed=1234))
    assert out == "ok"
    body = sent[-1][1]
    got = body["options"]["seed"] if provider == "ollama" else body["seed"]
    assert got == 1234
    sent.clear()
    asyncio.run(media.vlm_generate([], "hi", provider, url, "m", max_tokens=10))          # no seed: nothing sent
    body = sent[-1][1]
    assert "seed" not in body and "seed" not in (body.get("options") or {})
