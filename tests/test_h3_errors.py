"""H3 plain-English error table.

These are the failures this pack actually produces - each one was hit while
building it - and the table applies to all 62 nodes through web/kit.js. It is
worth EXECUTING rather than eyeballing, so these tests drive the real JS module
through node.

web/h3_errors.js has no ComfyUI imports, which is what makes that possible.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

PACK = Path(__file__).resolve().parents[1]
RULES_JS = PACK / "web" / "h3_errors.js"
NODE = shutil.which("node")

pytestmark = pytest.mark.skipif(NODE is None, reason="node is not installed")


def _humanise(samples: list[str]) -> list[str]:
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        (tmp / "h3_errors.mjs").write_bytes(RULES_JS.read_bytes())
        driver = tmp / "run.mjs"
        driver.write_text(
            'import { humaniseH3Error } from "./h3_errors.mjs";\n'
            "const inp = JSON.parse(process.argv[2]);\n"
            "console.log(JSON.stringify(inp.map(humaniseH3Error)));\n",
            encoding="utf-8",
        )
        r = subprocess.run([NODE, str(driver), json.dumps(samples)],
                           capture_output=True, text=True, timeout=60)
        assert r.returncode == 0, r.stderr
        return json.loads(r.stdout.strip())


def test_rules_module_stays_free_of_comfyui_imports():
    # INVARIANT: one ComfyUI import here and the entire table silently stops
    # being tested.
    src = RULES_JS.read_text(encoding="utf-8")
    assert "scripts/app.js" not in src and "scripts/api.js" not in src


def test_missing_h3_blames_the_build_not_the_pack():
    # INVARIANT: this exact error cost 6 nodes their registration during the
    # Motion Context port. It means the ComfyUI BUILD lacks H3 - saying
    # "No module named comfy.ldm.minimax" sends the user to reinstall the wrong
    # thing.
    (out,) = _humanise(["ModuleNotFoundError: No module named 'comfy.ldm.minimax'"])
    assert "build" in out.lower()
    assert "comfy.ldm.minimax" not in out or "Update ComfyUI" in out
    assert "installed correctly" in out.lower()


def test_comfy_kitchen_skew_is_named_as_a_version_mismatch():
    # INVARIANT: this raises AttributeError, not ImportError, and reads as
    # nonsense. It is a version skew between ComfyUI and comfy_kitchen.
    (out,) = _humanise(
        ["AttributeError: module 'comfy_kitchen' has no attribute "
         "'int8_attention_is_available'"])
    assert "version" in out.lower()
    assert "comfy_kitchen" in out


def test_illegal_frame_count_states_the_rule_and_the_fix():
    # INVARIANT: THE H3 gotcha. The raw failure happens deep in the layout and
    # never mentions 17, so the message must state the rule and point at the
    # node that applies it.
    (out,) = _humanise(
        ["ValueError: frame count 81 is not on H3's grid"])
    assert "17" in out
    assert "Context Windows" in out


def test_split_sigmas_is_explained_not_echoed():
    # INVARIANT: "schedule stops at 0.42" means nothing; "this is half a split
    # schedule, remove SplitSigmas" is actionable.
    (out,) = _humanise(
        ["RuntimeError: This sigma schedule stops at 0.418000 instead of 0.0"])
    assert "SplitSigmas" in out


def test_oom_suggests_the_pack_specific_remedy():
    # INVARIANT: for THIS pack the first remedy is a shorter window, not a
    # smaller batch - long clips are meant to be tiled.
    (out,) = _humanise(["torch.cuda.OutOfMemoryError: CUDA out of memory."])
    assert "Context Windows" in out or "window" in out.lower()


def test_unconnected_input_is_plain():
    (out,) = _humanise(["AttributeError: 'NoneType' object has no attribute 'shape'"])
    assert "not connected" in out.lower()
    assert "NoneType" not in out


def test_wrong_model_says_which_wire():
    # INVARIANT: these nodes patch H3 only; the useful information is that the
    # MODEL wire is wrong, not that an isinstance failed.
    (out,) = _humanise(
        ["ValueError: MiniMaxH3_BlockCacheT8 requires a native MiniMax H3 "
         "diffusion model"])
    assert "H3 model" in out or "not a MiniMax H3" in out
    assert "loader" in out.lower()


def test_unknown_error_falls_back_to_the_exception_line():
    (out,) = _humanise([
        "Traceback (most recent call last):\n"
        '  File "a.py", line 1, in <module>\n'
        "OddError: something specific went wrong"
    ])
    assert out.startswith("OddError")
    assert "Traceback" not in out


def test_cancel_is_not_dressed_up_as_a_failure():
    (out,) = _humanise(["KeyboardInterrupt: Interrupted"])
    assert out.lower().startswith("cancel")


def test_never_returns_an_empty_string():
    # INVARIANT: a blank status strip reads as "fine".
    for out in _humanise(["", "   "]):
        assert out.strip()
