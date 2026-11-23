"""Child process for the GGUF provider (ours). Standalone: stdlib + llama_cpp only, never imports the pack.

Reads one JSON request on stdin, writes one JSON line on stdout. Runs in its own process because a llama-cpp-python
wheel built with AVX-512 dies with an illegal instruction on CPUs without it (see tools/llama_cpp_probe.py): here that
kills this child and fails one request, never ComfyUI.

    {"action": "probe"}                      -> {"ok": true, "version": "...", "handlers": [...]}
    {"action": "chat", "model_path", "mmproj", "handler", "n_ctx", "n_gpu_layers", "messages", "temperature",
     "top_p", "max_tokens", "seed"}         -> {"ok": true, "text", "finish_reason", "usage"} | {"ok": false, "error"}
"""

from __future__ import annotations

import json
import sys


def main() -> int:
    req = json.loads(sys.stdin.read() or "{}")
    try:
        import llama_cpp
        from llama_cpp import llama_chat_format
    except Exception as e:  # noqa: BLE001
        print(json.dumps({"ok": False, "error": f"llama-cpp-python is not importable ({type(e).__name__}: {e})"}))
        return 0
    if req.get("action") == "probe":
        handlers = [n for n in dir(llama_chat_format) if n.endswith("ChatHandler")]
        print(json.dumps({"ok": True, "version": getattr(llama_cpp, "__version__", "?"), "handlers": handlers}))
        return 0
    kwargs = {"model_path": req["model_path"], "n_ctx": int(req["n_ctx"]), "n_gpu_layers": int(req["n_gpu_layers"]),
              "verbose": False}
    if req.get("handler") and req.get("mmproj"):
        kwargs["chat_handler"] = getattr(llama_chat_format, req["handler"])(clip_model_path=req["mmproj"], verbose=False)
    try:
        llm = llama_cpp.Llama(**kwargs)
        kw = {"messages": req["messages"], "temperature": float(req["temperature"]), "top_p": float(req["top_p"]),
              "max_tokens": int(req["max_tokens"])}
        if req.get("seed") is not None:
            kw["seed"] = int(req["seed"])
        r = llm.create_chat_completion(**kw)
    except Exception as e:  # noqa: BLE001
        print(json.dumps({"ok": False, "error": f"{type(e).__name__}: {e}"}))
        return 0
    choice = (r.get("choices") or [{}])[0]
    print(json.dumps({"ok": True, "text": str((choice.get("message") or {}).get("content") or ""),
                      "finish_reason": str(choice.get("finish_reason") or "stop"), "usage": r.get("usage") or {}}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
