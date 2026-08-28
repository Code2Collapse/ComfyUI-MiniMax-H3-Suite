"""JS/Python parity for the sigma-shift math used by the W3 live plot.

`timeShiftSigma` in web/w3_sigma_plot.js is a second implementation of
`mmx_utils.sigma_schedule.time_shift_sigma`, written in JS so the plot can
recompute the curve client-side when the shift widgets move, without
re-queueing the prompt. Two implementations of one formula drift.

This test extracts the function SOURCE OUT OF THE REAL WIDGET FILE and runs
it. It deliberately does not paste a copy of the JS into this file: a copy
would stay green while the shipped widget drifted, which is the exact failure
mode the test exists to prevent.

The widget module cannot simply be imported — it pulls in ./shared.js, which
imports ../../scripts/app.js and only resolves inside a running ComfyUI. So
the single function is lifted by brace balancing and evaluated on its own.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from mmx_utils.sigma_schedule import time_shift_sigma

ROOT = Path(__file__).resolve().parents[1]
WIDGET_JS = ROOT / "web" / "w3_sigma_plot.js"
FUNC_NAME = "timeShiftSigma"


def _extract_js_function(source: str, name: str) -> str:
    """Return the full `function <name>(...) { ... }` text, by brace balance."""
    marker = f"function {name}("
    start = source.find(marker)
    assert start != -1, (
        f"{name} not found in {WIDGET_JS.name}. If it was renamed or inlined, this "
        "parity pin is broken and must be repointed — not deleted."
    )
    brace = source.find("{", start)
    assert brace != -1, f"no body found for {name}"
    depth = 0
    for i in range(brace, len(source)):
        if source[i] == "{":
            depth += 1
        elif source[i] == "}":
            depth -= 1
            if depth == 0:
                return source[start : i + 1]
    raise AssertionError(f"unbalanced braces while extracting {name}")


@pytest.fixture(scope="module")
def node_exe():
    exe = shutil.which("node")
    if not exe:
        pytest.skip("node not on PATH")
    return exe


def test_extracted_source_is_the_shipped_one():
    """Guard the extractor itself — a silently empty lift would pass everything."""
    src = _extract_js_function(WIDGET_JS.read_text(encoding="utf-8"), FUNC_NAME)
    assert src.count("{") == src.count("}")
    assert "return" in src
    # the shipped formula, not a stub
    assert "fromShift" in src and "toShift" in src


def test_js_time_shift_sigma_matches_python(node_exe):
    js_func = _extract_js_function(WIDGET_JS.read_text(encoding="utf-8"), FUNC_NAME)
    harness = (
        js_func
        + "\nconst payload = JSON.parse(process.argv[1]);"
        + f"\nconsole.log(JSON.stringify(payload.map(([s, sv, sa]) => {FUNC_NAME}(s, sv, sa))));"
    )

    grid = [
        (sigma, sv, sa)
        for sigma in (0.0, 0.05, 0.25, 0.5, 0.75, 1.0)
        for sv in (1.0, 3.0, 12.0)
        for sa in (1.0, 3.0, 12.0)
    ]
    proc = subprocess.run(
        [node_exe, "-e", harness, json.dumps(grid)],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    js_vals = json.loads(proc.stdout.strip())
    py_vals = [float(time_shift_sigma(s, sv, sa)) for s, sv, sa in grid]

    assert len(js_vals) == len(py_vals) == len(grid)
    for (s, sv, sa), j, p in zip(grid, js_vals, py_vals):
        assert abs(j - p) <= 1e-12, f"sigma={s} shift_v={sv} shift_a={sa}: JS {j} != Python {p}"
