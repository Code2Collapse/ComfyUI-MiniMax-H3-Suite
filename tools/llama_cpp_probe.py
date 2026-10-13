"""Safe probe for the llama.cpp crash behind the old H3 Prompt Writer (C2C L8.24).

    python tools/llama_cpp_probe.py                      # CPU flags + is llama-cpp-python usable at all
    python tools/llama_cpp_probe.py --model x.gguf       # also load that GGUF and generate a few tokens
    python tools/llama_cpp_probe.py --model x.gguf --gpu-layers 0 --json out.json

Run it with the SAME Python that runs ComfyUI (e.g. ComfyUI's venv / python_embeded). Every risky step - importing
llama_cpp, loading the model, generating - runs in a CHILD process, so an illegal instruction (SIGILL, Windows
0xC000001D) or a segfault kills only the child and is reported here instead of taking anything else down.

Why: llama-cpp-python's prebuilt CUDA wheels can contain AVX-512 code; on a CPU without AVX-512 (e.g. AMD Zen 2 /
Zen 3, Intel 12th-14th gen consumer parts) loading a model dies with an illegal instruction (upstream reports
duckyshell/ComfyUI-MiniMaxH3-Prompt-Writer #13, #22). Inside ComfyUI that is the whole ComfyUI process. MiniMax H3
Prompt Writer v2 never loads a model in-process; for llama.cpp run `llama-server` and use its
'openai-compatible server' backend.

Stdlib only. Prints a plain report; --json also writes it as JSON. Nothing is downloaded or installed.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import textwrap

FLAGS = ("sse4_2", "avx", "avx2", "fma", "f16c", "avx512f", "avx512bw", "avx512vl", "avx512_vnni", "avx512_bf16",
         "avx_vnni", "amx_tile")


def cpu_info() -> dict:
    info: dict = {"machine": platform.machine(), "processor": platform.processor()}
    if sys.platform.startswith("linux"):
        try:
            text = open("/proc/cpuinfo", encoding="utf-8", errors="replace").read()
        except OSError as e:
            info["error"] = str(e)
            return info
        model = next((ln.split(":", 1)[1].strip() for ln in text.splitlines() if ln.startswith("model name")), "")
        flags = next((ln.split(":", 1)[1].split() for ln in text.splitlines() if ln.startswith("flags")), [])
        info.update(model=model, flags={f: f in flags for f in FLAGS})
    elif sys.platform == "win32":
        import ctypes
        k = ctypes.windll.kernel32
        # PF_* constants (winnt.h): 40 AVX2, 41 AVX512F
        info["flags"] = {"avx2": bool(k.IsProcessorFeaturePresent(40)), "avx512f": bool(k.IsProcessorFeaturePresent(41))}
        try:
            out = subprocess.run(["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"],
                                 capture_output=True, text=True, timeout=20)
            info["model"] = out.stdout.strip()
        except Exception as e:  # noqa: BLE001
            info["model_error"] = str(e)
    return info


def explain(code: int | None) -> str:
    if code is None:
        return "timed out"
    if code == 0:
        return "ok"
    if code in (-4, 132, 0xC000001D, -1073741795):
        return "ILLEGAL INSTRUCTION - this llama-cpp-python build uses CPU instructions this processor lacks"
    if code in (-11, 139, 0xC0000005, -1073741819):
        return "segmentation fault / access violation inside the native library"
    if code in (-6, 134, 0xC0000409, -1073740791):
        return "native abort"
    return f"exited with code {code}"


def child(code: str, timeout: float) -> dict:
    try:
        p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"exit": None, "meaning": explain(None), "stdout": "", "stderr": ""}
    return {"exit": p.returncode, "meaning": explain(p.returncode), "stdout": p.stdout[-4000:],
            "stderr": p.stderr[-4000:]}


IMPORT = textwrap.dedent("""
    import json, importlib.metadata as m
    import llama_cpp
    info = {"version": m.version("llama-cpp-python")}
    try:
        info["system_info"] = llama_cpp.llama_print_system_info().decode("utf-8", "replace")
    except Exception as e:
        info["system_info_error"] = repr(e)
    try:
        info["gpu_offload"] = bool(llama_cpp.llama_supports_gpu_offload())
    except Exception as e:
        info["gpu_offload_error"] = repr(e)
    print("PROBE_JSON " + json.dumps(info))
""")

LOAD = textwrap.dedent("""
    import json, sys, time
    from llama_cpp import Llama
    t = time.time()
    llm = Llama(model_path=sys.argv[1] if len(sys.argv) > 1 else {path!r}, n_ctx=512, n_gpu_layers={layers},
                verbose=False)
    loaded = time.time() - t
    out = llm.create_completion("Hello", max_tokens=8)
    print("PROBE_JSON " + json.dumps({{"load_s": round(loaded, 2), "text": out["choices"][0]["text"]}}))
""")


def parse(res: dict) -> dict:
    for line in res.get("stdout", "").splitlines():
        if line.startswith("PROBE_JSON "):
            return json.loads(line[len("PROBE_JSON "):])
    return {}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", help="a GGUF file to load (optional)")
    ap.add_argument("--gpu-layers", type=int, default=-1, help="n_gpu_layers for the load test (default -1 = all)")
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--json", help="also write the report here")
    a = ap.parse_args()

    rep: dict = {"python": sys.executable, "python_version": platform.python_version(), "os": platform.platform(),
                 "cpu": cpu_info(), "env": {k: os.environ[k] for k in ("CUDA_VISIBLE_DEVICES", "GGML_CUDA_NO_PINNED")
                                            if k in os.environ}}
    imp = child(IMPORT, a.timeout)
    rep["import"] = {**{k: imp[k] for k in ("exit", "meaning")}, **parse(imp)}
    if imp["exit"] != 0:
        rep["import"]["stderr_tail"] = imp["stderr"][-1500:]
    if a.model and imp["exit"] == 0:
        load = child(LOAD.format(path=os.path.abspath(a.model), layers=int(a.gpu_layers)), a.timeout)
        rep["load_and_generate"] = {**{k: load[k] for k in ("exit", "meaning")}, **parse(load)}
        if load["exit"] != 0:
            rep["load_and_generate"]["stderr_tail"] = load["stderr"][-1500:]

    flags = rep["cpu"].get("flags", {})
    print(f"CPU: {rep['cpu'].get('model') or rep['cpu'].get('processor')}  AVX2={flags.get('avx2')}  "
          f"AVX-512F={flags.get('avx512f')}")
    print(f"llama-cpp-python import: {rep['import']['meaning']}  version={rep['import'].get('version', '-')}")
    if rep["import"].get("system_info"):
        print("  build/system info:", rep["import"]["system_info"].strip())
    if "load_and_generate" in rep:
        print(f"load + generate: {rep['load_and_generate']['meaning']}")
    verdict = []
    lg = rep.get("load_and_generate", {})
    if "ILLEGAL" in rep["import"]["meaning"] or "ILLEGAL" in lg.get("meaning", ""):
        verdict.append("This is the crash: the wheel needs instructions this CPU does not have. Inside ComfyUI it "
                       "would kill ComfyUI. Use llama-server / Ollama built for this CPU and Prompt Writer v2's "
                       "'openai-compatible server' or 'ollama' backend.")
    elif imp["exit"] != 0:
        verdict.append("llama-cpp-python is not usable here (see stderr_tail) - Prompt Writer v2 does not need it.")
    elif not a.model:
        verdict.append("Import is fine; pass --model some.gguf to test a real load (that is where the crash happens).")
    else:
        verdict.append("No crash on this machine with this wheel and model.")
    rep["verdict"] = verdict
    print("verdict:", " ".join(verdict))
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(rep, f, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
