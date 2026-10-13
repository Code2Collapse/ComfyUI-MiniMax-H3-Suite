"""web/kit.js guardWidgetValues - keeps ComfyUI front end 1.53's legacy widget migration off MiniMax H3 saves.

Front end 1.53's node configure() calls migrateWidgetsValues(): when a saved widgets_values has exactly as many
entries as "widget inputs + forceInput inputs", it drops the entries at the forceInput positions (meant for old saves
that stored values for converted widgets). On H3 Subject Uncrop a save of [16, ""] lost the 16, so feather loaded as
"" and the server refused the graph (C2C L8.21). The guard appends one placeholder in exactly that case.

CPU-only. The guard is read out of kit.js and run through node next to core's function.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

KIT = Path(__file__).resolve().parents[1] / "web" / "kit.js"

# Copied verbatim from ComfyUI_frontend 1.53.10 (GPL-3.0, Comfy Org), minified bundle; used only as the reference.
CORE_MIGRATE = (
    "function migrateWidgetsValues(e,t,n){let r=new Set(t.map(e=>e.name)),i=Object.values(e).filter(e=>r.has(e.name)"
    "||e.forceInput).flatMap(e=>e.control_after_generate?[!!e.forceInput,!1]:[!!e.forceInput]);"
    "return i.length===n?.length?n.filter((e,t)=>!i[t]):n}"
)

UNCROP = {  # object_info of MiniMaxH3_SubjectUncrop as the front end sees it
    "defs": {"cropped_images": {"name": "cropped_images"}, "original_images": {"name": "original_images"},
             "bboxes": {"name": "bboxes", "forceInput": True}, "feather": {"name": "feather"},
             "cropped_masks": {"name": "cropped_masks"}},
    "widgets": ["feather", "mmx_status"],
    "values": [16],
}
DETAIL = {  # MiniMaxH3_DetailMatch: a seed widget gets control_after_generate, the def carries no flag
    "defs": {"cropped_images": {"name": "cropped_images"}, "original_images": {"name": "original_images"},
             "bboxes": {"name": "bboxes", "forceInput": True}, "grow_px": {"name": "grow_px"},
             "sharpen": {"name": "sharpen"}, "max_gain": {"name": "max_gain"}, "grain": {"name": "grain"},
             "seed": {"name": "seed"}, "edit_mask": {"name": "edit_mask"}},
    "widgets": ["grow_px", "sharpen", "max_gain", "grain", "seed", "control_after_generate", "mmx_status"],
    "values": [9, 0.7, 2.5, 1.0, 0, "randomize"],
}


def _guard_source() -> str:
    m = re.search(r"^export function guardWidgetValues\(.*?^}", KIT.read_text(encoding="utf-8"), re.S | re.M)
    assert m, "guardWidgetValues not found in web/kit.js"
    return m.group(0).replace("export ", "", 1)


def _run(cases):
    script = _guard_source() + "\n" + CORE_MIGRATE + "\n" + (
        "const cases = " + json.dumps(cases) + ";\n"
        "const out = cases.map(c => {\n"
        "  const ws = c.widgets.map(name => ({ name }));\n"
        "  return { core_alone: migrateWidgetsValues(c.defs, ws, c.values),\n"
        "           guarded: migrateWidgetsValues(c.defs, ws, guardWidgetValues(c.defs, c.widgets, c.values)) };\n"
        "});\nprocess.stdout.write(JSON.stringify(out));\n")
    with tempfile.TemporaryDirectory() as d:
        probe = Path(d) / "probe.js"
        probe.write_text(script, encoding="utf-8")
        p = subprocess.run([shutil.which("node"), str(probe)], capture_output=True, text=True, timeout=60)
    assert p.returncode == 0, p.stderr[:500]
    return json.loads(p.stdout)


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not on PATH")
def test_old_uncrop_save_keeps_feather():
    # INVARIANT: a pre-fix save [16, ""] (status strip serialised) loads feather = 16; core alone drops the 16.
    case = dict(UNCROP, values=[16, ""])
    (r,) = _run([case])
    assert r["core_alone"][0] == ""                     # the bug, reproduced against core's own function
    assert r["guarded"][0] == 16


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not on PATH")
def test_detail_match_save_keeps_its_first_value():
    # INVARIANT: a 6-value Detail Match save (status not serialised) keeps grow_px; core alone shifts every value.
    (r,) = _run([DETAIL])
    assert r["core_alone"][0] == 0.7
    assert r["guarded"][:6] == DETAIL["values"]


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not on PATH")
def test_saves_core_would_not_touch_pass_through_unchanged():
    # INVARIANT: when the lengths differ (no migration) or there is no forceInput, the guard returns the same array.
    no_force = {"defs": {"feather": {"name": "feather"}}, "widgets": ["feather"], "values": [16]}
    r_uncrop, r_plain = _run([UNCROP, no_force])
    assert r_uncrop["core_alone"] == r_uncrop["guarded"] == [16]
    assert r_plain["core_alone"] == r_plain["guarded"] == [16]
