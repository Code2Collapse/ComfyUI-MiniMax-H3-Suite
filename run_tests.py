#!/usr/bin/env python3
"""Offline test runner — writes results next to this file."""
from __future__ import annotations

import io
import sys
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parent

import importlib.util


sys.path.insert(0, str(ROOT))

# Prefer ComfyUI on disk for comfy_api / comfy_extras imports.
for candidate in (
    ROOT.parent.parent / "ComfyUI_windows_portable" / "ComfyUI",
    ROOT.parent / "third_party" / "ComfyUI",
):
    if (candidate / "comfy_api").is_dir():
        sys.path.insert(0, str(candidate))
        break

import pytest

LOG_PATH = ROOT / "_pytest_last_run.txt"


if __name__ == "__main__":
    buf = io.StringIO()
    with redirect_stdout(buf), redirect_stderr(buf):
        code = pytest.main(["-q", str(ROOT / "tests")])
    text = buf.getvalue()
    sys.stdout.write(text)
    sys.stdout.flush()
    LOG_PATH.write_text(f"{text}\nEXIT_CODE={code}\n", encoding="utf-8")
    raise SystemExit(code)
