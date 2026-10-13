"""tools/llama_cpp_probe.py - the safe Kubuntu / Windows probe for the llama.cpp crash (C2C L8.24). CPU-only."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

PROBE = Path(__file__).resolve().parents[1] / "tools" / "llama_cpp_probe.py"


def _probe():
    spec = importlib.util.spec_from_file_location("llama_cpp_probe", PROBE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_illegal_instruction_codes_are_named_on_both_oses():
    # INVARIANT: Linux SIGILL (-4 / shell 132) and Windows 0xC000001D both read as the AVX-512 crash.
    p = _probe()
    for code in (-4, 132, 0xC000001D):
        assert "ILLEGAL INSTRUCTION" in p.explain(code)
    assert p.explain(0) == "ok" and p.explain(None) == "timed out"


def test_a_native_crash_kills_only_the_child():
    # INVARIANT: the risky step runs in a child; a native abort there is reported, the probe itself lives on.
    p = _probe()
    res = p.child("import os; os.abort()", 60)
    assert res["exit"] not in (0, None)
    assert res["meaning"] in ("native abort", f"exited with code {res['exit']}")


def test_probe_imports_nothing_native_itself():
    # INVARIANT: loading the probe module never imports llama_cpp into the calling process.
    _probe()
    assert "llama_cpp" not in sys.modules
