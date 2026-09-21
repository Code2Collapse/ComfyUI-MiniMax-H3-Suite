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
import re
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


def test_parity_check_actually_fails_on_drift(node_exe):
    """Negative control: the comparison must reject a formula that has drifted.

    Without this, a subtly broken extractor (or a harness that silently returns
    the Python values) would keep the parity test green forever while the shipped
    widget diverged. This perturbs the EXTRACTED source by 0.1% and asserts the
    same comparison rejects it — so the test above is known to be able to fail.
    """
    js_func = _extract_js_function(WIDGET_JS.read_text(encoding="utf-8"), FUNC_NAME)
    perturbed = js_func.replace("return (toShift * base)", "return (1.001 * toShift * base)")
    assert perturbed != js_func, "perturbation did not apply — the return shape changed"

    harness = (
        perturbed
        + "\nconst payload = JSON.parse(process.argv[1]);"
        + f"\nconsole.log(JSON.stringify(payload.map(([s, sv, sa]) => {FUNC_NAME}(s, sv, sa))));"
    )
    grid = [(0.5, 12.0, 3.0), (0.25, 3.0, 1.0)]
    proc = subprocess.run(
        [node_exe, "-e", harness, json.dumps(grid)],
        check=True, capture_output=True, text=True, timeout=30,
    )
    js_vals = json.loads(proc.stdout.strip())
    py_vals = [float(time_shift_sigma(s, sv, sa)) for s, sv, sa in grid]

    assert any(abs(j - p) > 1e-12 for j, p in zip(js_vals, py_vals)), (
        "a 0.1% drift in the JS was NOT detected — the parity test cannot fail"
    )


# ── W8 NegPiP term grammar ─────────────────────────────────────────────────
#
# `parseTerms` in web/w8_negpip.js is a second implementation of
# `mmx_utils.negpip.parse_terms`, written in JS so the ledger updates while the
# user types instead of only after a run. The grammar is small but every part
# of it is a place to drift: whether a colon inside a phrase is a weight,
# whether "# " comments count, whether a bare line defaults to 1.0. Those all
# change which rows get flipped, silently.
#
# The two are NOT identical by design: python REFUSES a negative or oversized
# weight (the run must not proceed), while the JS keeps the line and marks it,
# so the ledger can say which line will be rejected before you queue. That
# difference is pinned here too, so neither side can quietly adopt the other's
# behaviour.

NEGPIP_JS = ROOT / "web" / "w8_negpip.js"

_VALID_TERM_CASES = [
    "blurry",
    "blurry : 1.5",
    "blurry:2",
    "close-up: hands",
    "  padded  ",
    "# a comment\nblurry",
    "\n\nblurry\n\nwatermark : 0.5\n",
    "text : 0",
    "three word phrase : 1.25",
    "trailing colon:",
    "a : b : 2",
    "10",
    "1:2:3",
]


def _run_parse_terms(node_exe, lines: list[str]):
    js_func = _extract_js_function(NEGPIP_JS.read_text(encoding="utf-8"), "parseTerms")
    harness = (
        js_func
        + "\nconst payload = JSON.parse(process.argv[1]);"
        + "\nconsole.log(JSON.stringify(payload.map((t) => parseTerms(t))));"
    )
    proc = subprocess.run(
        [node_exe, "-e", harness, json.dumps(lines)],
        check=True, capture_output=True, text=True, timeout=30,
    )
    return json.loads(proc.stdout.strip())


def test_negpip_extracted_source_is_the_shipped_one():
    src = _extract_js_function(NEGPIP_JS.read_text(encoding="utf-8"), "parseTerms")
    assert src.count("{") == src.count("}")
    assert "weight" in src and "startsWith" in src


def test_js_parse_terms_matches_python(node_exe):
    from mmx_utils.negpip import parse_terms

    js = _run_parse_terms(node_exe, _VALID_TERM_CASES)
    assert len(js) == len(_VALID_TERM_CASES)
    for text, got in zip(_VALID_TERM_CASES, js):
        want = [{"phrase": t.phrase, "weight": t.weight} for t in parse_terms(text)]
        assert got == want, f"{text!r}: JS {got} != Python {want}"


def test_js_keeps_the_lines_python_refuses_so_the_ledger_can_name_them(node_exe):
    from mmx_utils.negpip import parse_terms

    refused = ["blurry : -1", "blurry : 99"]
    js = _run_parse_terms(node_exe, refused)
    for text, got in zip(refused, js):
        with pytest.raises(ValueError):
            parse_terms(text)
        assert len(got) == 1, f"{text!r} must survive the JS parse so it can be flagged"


def test_negpip_parity_check_actually_fails_on_drift(node_exe):
    """Negative control, as above: prove this comparison can reject a change."""
    from mmx_utils.negpip import parse_terms

    js_func = _extract_js_function(NEGPIP_JS.read_text(encoding="utf-8"), "parseTerms")
    perturbed = js_func.replace('out.push({ phrase: line, weight: 1.0 })',
                                'out.push({ phrase: line, weight: 1.5 })')
    assert perturbed != js_func, "perturbation did not apply — the default changed"
    harness = (
        perturbed
        + "\nconst payload = JSON.parse(process.argv[1]);"
        + "\nconsole.log(JSON.stringify(payload.map((t) => parseTerms(t))));"
    )
    proc = subprocess.run(
        [node_exe, "-e", harness, json.dumps(["blurry"])],
        check=True, capture_output=True, text=True, timeout=30,
    )
    js = json.loads(proc.stdout.strip())
    want = [{"phrase": t.phrase, "weight": t.weight} for t in parse_terms("blurry")]
    assert js[0] != want, "a changed default weight was NOT detected"


# ── W9 swap-scope diagram ──────────────────────────────────────────────────
#
# The diagram lights the landmark groups a scope drives, in the same colours
# the renderer uses, and greys the jaw. Two copies of that mapping exist - the
# JS one for the panel and the Python one that actually renders - and if they
# drift the picture tells the user something the node does not do. The jaw
# entry is the one that matters: the panel promises it is never sent.

SWAP_JS = ROOT / "web" / "w9_swap_scope.js"


def _js_object(source: str, name: str) -> dict:
    """Lift a flat `const NAME = { ... };` object literal."""
    start = source.index(f"const {name} = {{")
    depth, i = 0, source.index("{", start)
    for j in range(i, len(source)):
        if source[j] == "{":
            depth += 1
        elif source[j] == "}":
            depth -= 1
            if depth == 0:
                body = source[i:j + 1]
                break
    else:
        raise AssertionError(f"unbalanced braces in {name}")
    out = {}
    for m in re.finditer(r"(\w+)\s*:\s*(\[[^\]]*\]|\"[^\"]*\"|'[^']*')", body):
        raw = m.group(2)
        if raw.startswith("["):
            out[m.group(1)] = re.findall(r"[\"']([^\"']+)[\"']", raw)
        else:
            out[m.group(1)] = raw.strip("\"'")
    return out


def test_the_diagram_knows_the_same_scopes_as_the_renderer():
    from mmx_utils.swap_regions import SWAP_SCOPES

    js = _js_object(SWAP_JS.read_text(encoding="utf-8"), "SCOPE_GROUPS")
    assert set(js) == set(SWAP_SCOPES), (
        f"panel scopes {sorted(js)} != renderer scopes {sorted(SWAP_SCOPES)}")


def test_each_scope_lights_exactly_the_groups_it_drives():
    from mmx_utils.swap_regions import SWAP_SCOPES

    js = _js_object(SWAP_JS.read_text(encoding="utf-8"), "SCOPE_GROUPS")
    for scope, spec in SWAP_SCOPES.items():
        assert sorted(js[scope]) == sorted(spec["groups"]), (
            f"{scope}: panel lights {sorted(js[scope])}, renderer drives "
            f"{sorted(spec['groups'])}")


def test_the_panel_lights_the_jaw_for_every_face_scope():
    """The jaw is driven now - it carries head pose and chin drop. A panel that
    still greyed it would be telling the user the opposite of what runs."""
    js = _js_object(SWAP_JS.read_text(encoding="utf-8"), "SCOPE_GROUPS")
    for scope in ("face", "head", "person", "lips"):
        assert "jaw" in js[scope], f"panel does not light the jaw for {scope!r}"


def test_the_panel_lights_the_pupils_wherever_gaze_is_driven():
    """Pupils are the only eye-DIRECTION signal; if the panel omits them a
    user has no way to see that gaze is being transferred at all."""
    from mmx_utils.swap_regions import SWAP_SCOPES

    js = _js_object(SWAP_JS.read_text(encoding="utf-8"), "SCOPE_GROUPS")
    for scope, spec in SWAP_SCOPES.items():
        if "pupils" in spec["groups"]:
            assert "pupils" in js[scope], f"panel omits pupils for {scope!r}"


def test_the_legend_colours_match_the_rendered_ones():
    from mmx_utils.swap_control import GROUP_COLOUR

    js = _js_object(SWAP_JS.read_text(encoding="utf-8"), "GROUP_COLOUR")
    assert set(js) == set(GROUP_COLOUR), "legend and renderer disagree on groups"
    for group, hexv in js.items():
        r, g, b = (int(hexv[i:i + 2], 16) / 255.0 for i in (1, 3, 5))
        want = GROUP_COLOUR[group]
        assert all(abs(a - b_) < 0.01 for a, b_ in zip((r, g, b), want)), (
            f"{group}: panel {hexv} != renderer {want}")


def test_the_panel_honours_drive_mouth():
    """Both levers change what actually drives the swap. A panel that only
    reacts to one of them shows a mouth that is lit while nothing draws it."""
    src = SWAP_JS.read_text(encoding="utf-8")
    assert 'widgetByName(node, "drive_mouth")' in src, (
        "the diagram never reads drive_mouth")
    assert 'active.delete("mouth")' in src, (
        "the diagram reads drive_mouth but still lights the lips")
    for name in ("swap_scope", "drive_jaw", "drive_mouth"):
        assert f'"{name}"' in src, f"{name} does not repaint the diagram"


def test_both_levers_exist_on_the_node_and_in_the_panel():
    """The panel is only honest if the widgets it reads are really there."""
    node_src = (ROOT / "mmx_nodes" / "swap_control.py").read_text(encoding="utf-8")
    for name in ("drive_jaw", "drive_mouth"):
        assert f'"{name}", default=True' in node_src, (
            f"{name} is not a widget on the node, so the panel reads nothing")
