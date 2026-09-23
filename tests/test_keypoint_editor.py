"""The Pose Puppeteer keypoint editor, and the contract it rests on.

The node takes `keypoints_json` - a skeleton per frame - and shipped it as a
multiline TEXT BOX. Nobody types a skeleton, so the field was effectively
unusable and the node fell back to "keypoints from pose image only" every
time. That fallback is the node's WEAKER path: the whole point of the override
is correcting a joint the detector put in the wrong place.

These tests pin the format agreement between the editor and mmx_utils. If
either side changes shape the edits stop arriving with no error at all - the
node simply behaves as though nothing was edited, which is exactly what it did
before the editor existed.

CPU-only. The JS is read as text and exercised through node where it matters.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

PACK = Path(__file__).resolve().parents[1]
if str(PACK) not in sys.path:
    sys.path.insert(0, str(PACK))

EDITOR = PACK / "web" / "w10_keypoints.js"
SHARED = PACK / "web" / "shared.js"


def source() -> str:
    assert EDITOR.exists(), "the keypoint editor is gone"
    return EDITOR.read_text(encoding="utf-8")


# -- the format agreement ---------------------------------------------------

def test_the_backend_reads_the_shape_the_editor_writes():
    """The editor writes {"frames":[{"keypoints":[{x,y,c}]}]}. The parser has
    to accept exactly that, or every edit is silently discarded."""
    from mmx_utils.puppeteer import _keypoints_json_to_array

    doc = {"frames": [{"keypoints": [{"x": 10, "y": 20, "c": 1.0},
                                     {"x": 30, "y": 40, "c": 0.5}]}]}
    arr = _keypoints_json_to_array(json.dumps(doc))
    assert arr is not None, "the backend rejected the editor's own format"
    assert arr.shape == (1, 2, 3)
    assert arr[0, 0, 0] == pytest.approx(10)
    assert arr[0, 1, 2] == pytest.approx(0.5)


def test_the_backend_also_takes_the_array_form():
    """The editor reads [x,y,c] arrays as well as objects, because the backend
    does. A file that loads in one and not the other would be a trap."""
    from mmx_utils.puppeteer import _keypoints_json_to_array

    arr = _keypoints_json_to_array(json.dumps(
        {"frames": [{"keypoints": [[1, 2, 0.9], [3, 4]]}]}))
    assert arr is not None
    assert arr[0, 0, 2] == pytest.approx(0.9)
    assert arr[0, 1, 2] == pytest.approx(1.0), "a missing confidence is not 1.0"


def test_several_frames_survive():
    from mmx_utils.puppeteer import _keypoints_json_to_array

    doc = {"frames": [{"keypoints": [{"x": 1, "y": 1, "c": 1}]},
                      {"keypoints": [{"x": 2, "y": 2, "c": 1}]}]}
    arr = _keypoints_json_to_array(json.dumps(doc))
    assert arr.shape[0] == 2


def test_nonsense_is_refused_rather_than_half_read():
    from mmx_utils.puppeteer import _keypoints_json_to_array

    assert _keypoints_json_to_array("{not json") is None
    assert _keypoints_json_to_array(json.dumps({"nope": 1})) is None
    assert _keypoints_json_to_array(json.dumps({"frames": []})) is None


def test_the_widget_name_matches():
    """The editor writes keypoints_json by name. Rename it on the node and the
    editor saves into nothing."""
    node_src = (PACK / "mmx_nodes" / "puppeteer.py").read_text(encoding="utf-8")
    assert '"keypoints_json"' in node_src
    assert '"keypoints_json"' in source()


def test_it_targets_the_right_node():
    node_src = (PACK / "mmx_nodes" / "puppeteer.py").read_text(encoding="utf-8")
    assert 'node_id="MiniMaxH3_PosePuppeteer"' in node_src
    assert 'const NODE_ID = "MiniMaxH3_PosePuppeteer"' in source()


# -- round trip through the real JS -----------------------------------------

@pytest.mark.skipif(shutil.which("node") is None, reason="node is not on PATH")
def test_a_round_trip_through_the_real_editor_is_lossless():
    """readDoc then writeDoc must return the same skeleton - otherwise every
    open-and-close of the node drifts the pose.

    The two functions are lifted out of the shipped file rather than
    reimplemented, so this tests what actually runs.
    """
    src = source()
    body = []
    for fn in ("function emptyDoc", "function readDoc", "function writeDoc"):
        i = src.index(fn)
        # to the closing brace at column 0
        j = src.index("\n}\n", i) + 3
        body.append(src[i:j])
    probe = PACK / "tests" / "_kp_probe.mjs"
    doc = {"frames": [{"keypoints": [{"x": 10.5, "y": 20.25, "c": 0.5},
                                     {"x": 30, "y": 40, "c": 1}]},
                      {"keypoints": [{"x": 1, "y": 2, "c": 0.125}]}]}
    probe.write_text(
        "const parseJsonSafe = (t) => { try { return JSON.parse(t); } catch { return null; } };\n"
        + "\n".join(body)
        + "\nconst back = writeDoc(readDoc(" + json.dumps(json.dumps(doc)) + "));\n"
        "process.stdout.write(back);\n", encoding="utf-8")
    try:
        p = subprocess.run([shutil.which("node"), str(probe)],
                           capture_output=True, text=True, timeout=60)
        assert p.returncode == 0, p.stderr[:500]
        out = json.loads(p.stdout)
    finally:
        probe.unlink(missing_ok=True)

    assert len(out["frames"]) == 2
    a = out["frames"][0]["keypoints"]
    assert a[0]["x"] == pytest.approx(10.5)
    assert a[0]["c"] == pytest.approx(0.5)
    assert out["frames"][1]["keypoints"][0]["c"] == pytest.approx(0.125)


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not on PATH")
def test_the_backend_accepts_what_the_real_editor_emits():
    """The end-to-end contract: JS writes it, Python reads it."""
    from mmx_utils.puppeteer import _keypoints_json_to_array

    src = source()
    body = []
    for fn in ("function emptyDoc", "function readDoc", "function writeDoc"):
        i = src.index(fn)
        j = src.index("\n}\n", i) + 3
        body.append(src[i:j])
    probe = PACK / "tests" / "_kp_probe2.mjs"
    probe.write_text(
        "const parseJsonSafe = (t) => { try { return JSON.parse(t); } catch { return null; } };\n"
        + "\n".join(body)
        + '\nprocess.stdout.write(writeDoc(readDoc(\'{"frames":[{"keypoints":[[5,6,0.4]]}]}\')));\n',
        encoding="utf-8")
    try:
        p = subprocess.run([shutil.which("node"), str(probe)],
                           capture_output=True, text=True, timeout=60)
        assert p.returncode == 0, p.stderr[:500]
        emitted = p.stdout
    finally:
        probe.unlink(missing_ok=True)

    arr = _keypoints_json_to_array(emitted)
    assert arr is not None, f"the backend rejected what the editor emits: {emitted}"
    assert arr[0, 0, 0] == pytest.approx(5)
    assert arr[0, 0, 2] == pytest.approx(0.4)


# -- behaviour the comments promise -----------------------------------------

def test_dragging_marks_a_point_certain():
    """A point placed by hand IS certain. Leaving it at 0.2 tells everything
    downstream to distrust the correction just made."""
    assert re.search(r"k\.c\s*=\s*1\s*;", source()), \
        "dragging no longer sets confidence to 1"


def test_low_confidence_points_are_drawn_hollow():
    """They are the ones worth moving, so the eye should land on them."""
    src = source()
    assert "sure" in src and "ctx.stroke()" in src
    assert "c >= 0.5" in src


def test_drag_listeners_live_on_the_window():
    """A fast drag leaves the canvas; on the canvas the point would stick to
    the cursor after the button came up."""
    src = source()
    assert 'window.addEventListener("pointermove"' in src
    assert 'window.removeEventListener("pointermove"' in src, \
        "the move listener outlives the node"


def test_it_saves_once_per_drag_not_once_per_pixel():
    """Saving on every move makes a drag sixty undo steps."""
    src = source()
    move = src[src.index("const move ="):src.index("const up =")]
    assert "save()" not in move, "the editor saves mid-drag"
    up = src[src.index("const up ="):src.index("// On the window")]
    assert "save()" in up


def test_it_does_not_pretend_to_detect():
    """An editor that could also generate would be a second, worse detector to
    maintain - the one upstream is already better than anything drawn by hand.
    The empty state has to SAY that rather than look broken."""
    src = source()
    assert "does not invent" in src or "does not invent a skeleton" in src
    assert "it does not invent one" in src or "does not invent one" in src


def test_the_editor_uses_the_shared_kit():
    """Every other widget in this pack does; a second set of helpers is a
    second set of theme bugs."""
    src = source()
    assert 'from "./shared.js"' in src
    for fn in ("chainOnRemoved", "disposeState", "rafThrottle", "setupDpiCanvas"):
        assert fn in src, f"{fn} is not used - lifecycle or theming will drift"


def test_the_shared_kit_import_depth_is_right():
    """shared.js reaches ComfyUI itself; this file only reaches shared.js. If
    shared.js ever moves, its own depth has to move with it - one 404 there
    takes out every widget in the pack."""
    assert 'from "../../scripts/app.js"' in SHARED.read_text(encoding="utf-8")
    assert re.search(r'from\s+"\./shared\.js"', source())
