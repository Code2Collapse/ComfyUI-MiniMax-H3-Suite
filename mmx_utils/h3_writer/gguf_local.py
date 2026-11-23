"""Local GGUF provider through llama-cpp-python, when it is installed (ours).

llama.cpp runs in a CHILD process (gguf_worker.py), never inside ComfyUI: a prebuilt llama-cpp-python wheel with
AVX-512 code dies with an illegal instruction on CPUs without it, which in-process would take ComfyUI down (the
reason Prompt Writer v2 never loads a model in-process; tools/llama_cpp_probe.py). Here the child dies and the node
reports it in words. The model is loaded per run, so unload_after has nothing to keep.

Any GGUF chat model works for text. Pictures and video sheets need an mmproj file beside the model AND a matching
chat handler in the installed llama-cpp-python (Qwen2.5-VL, LLaVA 1.5 / 1.6, MiniCPM-V 2.6). Newer vision
architectures (Gemma 4, Qwen 3.x) need a newer llama.cpp: run llama-server and pick 'openai-compatible server'.
A repo file that is not on disk is downloaded into <models>/LLM/<org>--<repo>/ only when the run may download.
"""

from __future__ import annotations

import base64
import json
import os
import subprocess
import sys
import time
from typing import Any, Callable

from . import downloads as D
from .modes import WriterError
from .text import strip_thinking

WORKER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gguf_worker.py")
PROBE_TIMEOUT = 60.0
_HANDLERS = (                      # (filename fragments, handler class name)
    (("qwen2.5-vl", "qwen2_5_vl", "qwen25vl", "qwen2.5vl"), "Qwen25VLChatHandler"),
    (("minicpm",), "MiniCPMv26ChatHandler"),
    (("llava-v1.6", "llava16", "llava-1.6"), "Llava16ChatHandler"),
    (("llava",), "Llava15ChatHandler"),
)


def _find(root: str, name: str) -> str | None:
    if not os.path.isdir(root):
        return None
    for dirpath, _dirs, files in os.walk(root):
        if name in files:
            return os.path.join(dirpath, name)
    return None


def resolve_model(model_id: str, preset: dict | None, root: str) -> tuple[str, str | None, str | None, str, str | None]:
    """(path of the .gguf, repo or None, revision, filename, mmproj filename from the preset)."""
    mid = (model_id or "").strip().strip('"')
    if not mid and preset:
        files = list(preset.get("files") or [])
        if not files:
            raise WriterError(f"The preset {preset.get('display_name')} lists no GGUF file.")
        repo = preset["repo"]
        target = os.path.join(root, repo.replace("/", "--"), files[0])
        return target, repo, preset.get("revision"), files[0], (files[1] if len(files) > 1 else None)
    if not mid:
        raise WriterError("Name the GGUF: a .gguf path, a file name in the LLM folder, 'org/repo::file.gguf', or a "
                          "preset.")
    if "::" in mid:
        repo, filename = (x.strip() for x in mid.split("::", 1))
        return os.path.join(root, repo.replace("/", "--"), filename), repo, None, filename, None
    if os.path.isabs(mid):
        if not os.path.isfile(mid):
            raise WriterError(f"The file {mid} does not exist.")
        return mid, None, None, os.path.basename(mid), None
    found = _find(root, os.path.basename(mid)) if mid.lower().endswith(".gguf") else None
    if found:
        return found, None, None, os.path.basename(found), None
    raise WriterError(f"{mid!r} was not found in {root}. Give a .gguf path, a file in that folder, or "
                      f"'org/repo::file.gguf' to download one.")


def _mmproj_beside(path: str, preset_name: str | None) -> str | None:
    folder = os.path.dirname(path)
    if preset_name and os.path.isfile(os.path.join(folder, preset_name)):
        return os.path.join(folder, preset_name)
    try:
        for f in sorted(os.listdir(folder)):
            if f.lower().startswith("mmproj") and f.lower().endswith(".gguf"):
                return os.path.join(folder, f)
    except OSError:
        pass
    return None


def handler_name(filename: str) -> str | None:
    low = filename.lower()
    for frags, name in _HANDLERS:
        if any(f in low for f in frags):
            return name
    return None


def explain_exit(code: int) -> str:
    if code in (-4, 132, 3221225501, -1073741795):
        return ("the llama.cpp child died with an illegal instruction: this llama-cpp-python build uses CPU "
                "instructions (often AVX-512) this processor lacks. Run llama-server and use 'openai-compatible "
                "server', or install a llama-cpp-python build for this CPU (check with tools/llama_cpp_probe.py)")
    if code in (-11, 139, 3221225477, -1073741819):
        return "the llama.cpp child crashed (access violation); the model file may be damaged or unsupported"
    return f"the llama.cpp child exited with code {code}"


def run_worker(request: dict[str, Any], timeout: float, check_interrupt: Callable[[], None] | None = None,
               popen: Callable[..., Any] = subprocess.Popen) -> dict[str, Any]:
    """Send one request to a fresh child; poll for interrupt; return its JSON reply. Raises WriterError."""
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if sys.platform == "win32" else 0
    try:
        proc = popen([sys.executable, "-I", WORKER], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                     stderr=subprocess.PIPE, env=env, creationflags=flags)
    except OSError as e:
        raise WriterError(f"Could not start the llama.cpp child process ({e}).") from None
    payload = json.dumps(request).encode("utf-8")
    t0 = time.monotonic()
    out = err = b""
    try:
        try:
            out, err = proc.communicate(payload, timeout=1.0)
        except subprocess.TimeoutExpired:
            while True:
                if check_interrupt is not None:
                    check_interrupt()
                if time.monotonic() - t0 > timeout:
                    raise WriterError(f"llama.cpp did not answer within {timeout:.0f} s; raise timeout_s.")
                try:
                    out, err = proc.communicate(timeout=1.0)
                    break
                except subprocess.TimeoutExpired:
                    continue
    except BaseException:
        proc.kill()
        proc.communicate()
        raise
    if proc.returncode != 0:
        tail = [ln for ln in (err or b"").decode("utf-8", "replace").splitlines() if ln.strip()][-1:]
        raise WriterError(explain_exit(int(proc.returncode)) + (f" ({tail[0][:200]})" if tail else "") + ".")
    lines = [ln for ln in out.decode("utf-8", "replace").splitlines() if ln.strip()]
    try:
        reply = json.loads(lines[-1])
    except (IndexError, ValueError):
        tail = err.decode("utf-8", "replace").strip().splitlines()[-1:] or ["no output"]
        raise WriterError(f"llama.cpp child gave no answer ({tail[0][:200]}).") from None
    if not reply.get("ok"):
        err = str(reply.get("error") or "")
        if "context" in err.lower():
            raise WriterError("The prompt does not fit the llama.cpp context; raise context_tokens or shorten the brief.")
        raise WriterError(f"llama.cpp: {err}")
    return reply


class GGUFProvider:
    def __init__(self, settings, progress: Callable[..., None], runner: Callable[..., dict] = run_worker):
        self.settings = settings
        self.progress = progress
        self.runner = runner
        root = D.llm_dir()
        path, repo, revision, filename, mmproj_name = resolve_model(settings.model_id, settings.preset, root)
        if not os.path.isfile(path):
            if repo is None:
                raise WriterError(f"{path} does not exist.")
            if not D.downloads_allowed(settings.downloads):
                raise WriterError(f"{filename} from {repo} is not in {root} and downloads are off "
                                  f"({D.why_not(settings.downloads)}). Put it at {path}, or allow downloads.")
            self._download(repo, revision, [f for f in (filename, mmproj_name) if f],
                           os.path.join(root, repo.replace("/", "--")))
        self.path = path
        self.model = os.path.basename(path)
        self.name = f"gguf llama.cpp ({self.model})"
        self.mmproj = _mmproj_beside(path, mmproj_name)
        probe = self.runner({"action": "probe"}, PROBE_TIMEOUT)
        self.version = probe.get("version", "?")
        hname = handler_name(self.model)
        self.handler = hname if (self.mmproj and hname and hname in probe.get("handlers", [])) else None
        progress(f"model: {self.path} (llama-cpp-python {self.version}, in a child process"
                 + (f", vision via {self.handler})" if self.handler else ", text only)"))

    def _download(self, repo: str, revision: str | None, names: list[str], target: str) -> None:
        try:
            import huggingface_hub as hub
        except Exception as e:  # noqa: BLE001
            raise WriterError(f"huggingface_hub could not be imported ({e}); download the GGUF by hand.") from None
        for i, name in enumerate(names, 1):
            self.progress(f"download {i}/{len(names)}: {repo}/{name} to {target}")
            t0 = time.perf_counter()
            try:
                hub.hf_hub_download(repo, name, revision=revision, local_dir=target)
            except Exception as e:  # noqa: BLE001
                raise WriterError(f"Download of {name} from {repo} failed ({type(e).__name__}: {e}).") from None
            self.progress(f"downloaded {name} in {time.perf_counter() - t0:.1f} s")

    def capabilities(self) -> dict[str, bool]:
        return {"images": self.handler is not None, "audio": False}

    def why_no_images(self) -> str:
        if not self.mmproj:
            return f"no mmproj*.gguf beside {self.model}"
        return (f"this GGUF ({self.model}) has no vision handler in llama-cpp-python {self.version}; run llama-server "
                f"with its mmproj and use 'openai-compatible server', or pick a Qwen2.5-VL / LLaVA / MiniCPM-V GGUF")

    def complete(self, messages: list[dict[str, Any]], temperature: float = 1.0, top_p: float = 0.95,
                 max_tokens: int = 2048, seed: int | None = None,
                 check_interrupt: Callable[[], None] | None = None) -> dict[str, Any]:
        msgs = []
        for m in messages:
            if self.handler is not None and m["role"] == "user":
                content = []
                for p in m["parts"]:
                    if p["type"] == "text":
                        content.append({"type": "text", "text": p["text"]})
                    elif p["type"] == "image":
                        url = f"data:{p['mime']};base64,{base64.b64encode(p['data']).decode('ascii')}"
                        content.append({"type": "image_url", "image_url": {"url": url}})
                msgs.append({"role": "user", "content": content})
            else:
                msgs.append({"role": m["role"], "content": "\n\n".join(p["text"] for p in m["parts"]
                                                                       if p["type"] == "text")})
        s = self.settings
        reply = self.runner({"action": "chat", "model_path": self.path, "mmproj": self.mmproj, "handler": self.handler,
                             "n_ctx": int(s.context_tokens), "n_gpu_layers": int(s.gguf_gpu_layers),
                             "messages": msgs, "temperature": float(temperature), "top_p": float(top_p),
                             "max_tokens": int(max_tokens), "seed": seed},
                            max(float(s.timeout_s), 30.0), check_interrupt)
        u = reply.get("usage") or {}
        return {"text": strip_thinking(reply.get("text", "")), "finish_reason": reply.get("finish_reason", "stop"),
                "usage": {"prompt_tokens": int(u.get("prompt_tokens") or 0),
                          "completion_tokens": int(u.get("completion_tokens") or 0)}}

    def close(self, unload: bool = True) -> None:
        return None                      # nothing stays loaded: each run uses a fresh child
