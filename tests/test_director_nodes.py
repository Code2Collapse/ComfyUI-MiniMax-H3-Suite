"""MiniMax H3 Director - the node guards, checked offline.

PORTED FROM: ComfyUI-MiniMaxH3-Director (GPL-3.0). Upstream's own suite, run
under pytest instead of as a script.

Nothing here samples, decodes or talks to a network. It does need ComfyUI
importable, so the whole file skips rather than fails when there is none.

One upstream workaround is deliberately NOT carried over. Its docstring said
PromptServer.instance had to exist before the import, because minimax_media
registered aiohttp routes at import time and died without a server. That is
fixed in the code here - the routes register when they can and defer when
they cannot - so the harness no longer has to fake a server to import a node.
"""
import importlib.util
import inspect
import os
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
PACK = os.path.dirname(HERE)
if PACK not in sys.path:
    sys.path.insert(0, PACK)

import pytest  # noqa: E402

# A plain import, which is the point.
#
# Upstream could not do this. Its loader built a synthetic module under a made-up
# name and stubbed `PromptServer.instance` with a fake routes object FIRST,
# because minimax_media registered aiohttp routes at import time and raised
# AttributeError without a live server. That workaround is gone: the routes now
# register when the server is up and defer when it is not, so the modules
# import like any other Python. If this import ever starts failing again with
# an AttributeError about PromptServer, the guard in minimax_media has been
# reverted.
director = pytest.importorskip(
    "mmx_nodes.director.minimax_director",
    reason="Director nodes need ComfyUI importable")
media = pytest.importorskip("mmx_nodes.director.minimax_media")
package = pytest.importorskip("mmx_nodes.director")

_results = []


def check(name, got, want):
    _results.append((got == want, name, got, want))


def check_raises(name, fn, needle):
    try:
        fn()
    except Exception as e:                                     # noqa: BLE001 - that is the check
        ok = needle in str(e)
        _results.append((ok, name, "raised %r" % str(e)[:90] if not ok else "raised", "raised"))
        return
    _results.append((False, name, "did not raise", "raised"))


# -------------------------------------------------- schema vs execute() signature
# ComfyUI passes every input by keyword, so an input declared in the schema with no
# matching parameter is a TypeError at run time and nowhere earlier — the whole graph
# dies on the node, after the models have loaded. Cheap to catch here instead.
def _schema_inputs(node_cls):
    schema = node_cls.define_schema()
    return [i.id for i in schema.inputs]


for node_cls in (package.MiniMaxH3Director, package.MiniMaxH3EnhancePrompt,
                 package.MiniMaxH3PreviewOverride, package.MiniMaxH3RetakeStitch,
                 package.MiniMaxH3SaveLastFrame):
    params = inspect.signature(node_cls.execute.__func__).parameters
    accepts_kwargs = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())
    missing = [] if accepts_kwargs else [
        name for name in _schema_inputs(node_cls) if name not in params]
    check("%s: every schema input has an execute() parameter" % node_cls.__name__,
          missing, [])

# and the reverse for the two inputs added in 0.2.1, so a rename cannot quietly orphan one
check("width and height are declared on the Director",
      [n for n in ("width", "height") if n in _schema_inputs(package.MiniMaxH3Director)],
      ["width", "height"])
# trap 9: widgets are serialised positionally, so a new one has to be last or every value
# after it lands on the wrong input in workflows already saved
check("api_key_env is the Enhance node's last widget",
      _schema_inputs(package.MiniMaxH3EnhancePrompt)[-1], "api_key_env")


# ------------------------------------------------------------------ resolve_size (#14)
# The settings panel owns custom_width/custom_height and hides them, so these two sockets
# are the only route a resolution node has into the canvas.
rs = director.resolve_size

check("nothing connected leaves the panel's box alone", rs(1344, 768), (1344, 768))
check("an unconnected pair keeps 0 meaning 'derive from the image'", rs(0, 0), (0, 0))
check("a connected width overrides the panel", rs(1344, 768, 1920, None), (1920, 768))
check("a connected height overrides the panel", rs(1344, 768, None, 1088), (1344, 1088))
check("both override", rs(0, 0, 864, 480), (864, 480))
check("floats off a resolution node are taken as pixels", rs(0, 0, 864.0, 480.0), (864, 480))

# A widget carries a minimum; a wire carries none. 0 is what an upstream node hands over
# when its own value was never set, which is the same trap `duration` fell into in #4.
check_raises("a connected width of 0 is refused by name",
             lambda: rs(1344, 768, 0, 768), "the connected 'width' is 0")
check_raises("a connected height of 0 is refused by name",
             lambda: rs(1344, 768, 1344, 0), "the connected 'height' is 0")
check_raises("a negative width is refused too",
             lambda: rs(0, 0, -8, None), "the connected 'width' is -8")
check_raises("the message says how to ask for the automatic canvas",
             lambda: rs(0, 0, 0, None), "Leave the socket unconnected")

# ---------------------------------------------------------------- resolve_window (#4)
# Same automation hazard, older sockets — checked here because the harness now exists.
rw = director.resolve_window
check("no automation returns the panel's window", rw({}, 24.0, 0, 120), (0, 120))
check("a connected duration is converted to frames and replaces the panel's",
      rw({}, 24.0, 0, 120, None, None, 7.0), (0, 168))
check("a connected start is converted too", rw({}, 24.0, 0, 120, 2.0), (48, 120))
check_raises("a connected duration of 0 is refused",
             lambda: rw({}, 24.0, 0, 120, None, None, 0.0), "the connected 'duration' is 0")
check_raises("a negative start is refused",
             lambda: rw({}, 24.0, 0, 120, -1.0), "cannot be negative")

# ------------------------------------------------------------------- API keys (#15)
key = media.resolve_api_key
os.environ.pop("MINIMAX_DIRECTOR_VLM_API_KEY", None)
os.environ.pop("OPENAI_API_KEY", None)
os.environ.pop("MMXD_TEST_KEY", None)

check("no key anywhere is an empty string, not None", key({}), "")
check("no key means no Authorization header at all", media._auth_headers(""), None)
check("None is a missing key too", media._auth_headers(None), None)
check("a key becomes a bearer header", media._auth_headers("sk-abc"),
      {"Authorization": "Bearer sk-abc"})
check("whitespace around a key is not sent", media._auth_headers("  sk-abc  "),
      {"Authorization": "Bearer sk-abc"})

check("an explicit key wins", key({"api_key": "sk-explicit"}), "sk-explicit")
check("a blank explicit key falls through rather than sending an empty bearer",
      key({"api_key": "   "}), "")

os.environ["MMXD_TEST_KEY"] = "sk-from-named-var"
check("a named environment variable is read", key({"api_key_env": "MMXD_TEST_KEY"}),
      "sk-from-named-var")
check("the explicit key still wins over the named variable",
      key({"api_key": "sk-explicit", "api_key_env": "MMXD_TEST_KEY"}), "sk-explicit")
check("a variable that does not exist is not an error",
      key({"api_key_env": "MMXD_NO_SUCH_VAR"}), "")

os.environ["OPENAI_API_KEY"] = "sk-openai"
check("OPENAI_API_KEY is the last resort", key({}), "sk-openai")
os.environ["MINIMAX_DIRECTOR_VLM_API_KEY"] = "sk-pack"
check("the pack's own variable is preferred over OPENAI_API_KEY", key({}), "sk-pack")
check("a named variable still beats both",
      key({"api_key_env": "MMXD_TEST_KEY"}), "sk-from-named-var")
for var in ("MMXD_TEST_KEY", "OPENAI_API_KEY", "MINIMAX_DIRECTOR_VLM_API_KEY"):
    os.environ.pop(var, None)

# The Enhance node's widget names a variable rather than holding a key, because widget
# values are serialised into the workflow. Guard the shape it passes in.
check("an empty widget resolves to no key", key({"api_key_env": ""}), "")

# -------------------------------------------------------- Save Last Frame node
# It sits mid-chain after VAEDecode, so the two things that must hold are that the batch
# comes out untouched and that exactly one file is written — the last frame, whatever the
# length. Saving goes to a temp directory here; the real output folder is left alone.
import shutil
import tempfile

import folder_paths
import torch

last_frame = package.MiniMaxH3SaveLastFrame
_real_output = folder_paths.get_output_directory()
_tmp_output = tempfile.mkdtemp(prefix="mmxd_lastframe_test_")
folder_paths.set_output_directory(_tmp_output)
try:
    def batch(n):
        """[n, 4, 4, 3], each frame a distinct grey so the saved one is identifiable.

        Scaled to stay well inside 0..1: the saver truncates 255*value to uint8, and two
        frames that both clip to 255 would make this test unable to fail.
        """
        frames = [torch.full((1, 4, 4, 3), (i + 1) / 512.0) for i in range(n)]
        return torch.cat(frames, dim=0)

    def written():
        return sorted(f for _r, _d, fs in os.walk(_tmp_output) for f in fs)

    images = batch(124)

    off = last_frame.execute(images, save=False, filename_prefix="off")
    check("save off returns the batch itself, not a copy", off.args[0] is images, True)
    check("save off writes nothing", written(), [])

    on = last_frame.execute(images, save=True, filename_prefix="on")
    check("save on still passes the whole batch through", on.args[0] is images, True)
    check("save on writes exactly one file", len(written()), 1)
    check("the file is a png", written()[0].endswith(".png"), True)
    check("the ui reports the one saved frame", len(on.ui.results), 1)

    # the frame saved has to be the LAST one, not the first — read it back and compare
    from PIL import Image as _PILImage
    saved_path = os.path.join(_tmp_output, on.ui.results[0]["subfolder"],
                              on.ui.results[0]["filename"])
    px = _PILImage.open(saved_path).convert("RGB").getpixel((0, 0))[0]
    expect_last = int(255.0 * float(images[-1, 0, 0, 0].item()))
    expect_first = int(255.0 * float(images[0, 0, 0, 0].item()))
    check("the saved pixel is the last frame's", px, expect_last)
    check("...and the two frames are distinguishable, so that check can fail",
          expect_last != expect_first, True)

    # a one-frame batch has a last frame like any other
    solo = last_frame.execute(batch(1), save=True, filename_prefix="solo")
    check("a single-frame batch saves that frame", len(solo.ui.results), 1)

    # and an empty one must not take a finished render down with it
    empty = torch.zeros((0, 4, 4, 3))
    before = len(written())
    out = last_frame.execute(empty, save=True, filename_prefix="empty")
    check("an empty batch passes through instead of raising", out.args[0] is empty, True)
    check("an empty batch writes nothing", len(written()), before)
    check("an empty batch reports no ui", getattr(out, "ui", None), None)
finally:
    folder_paths.set_output_directory(_real_output)
    shutil.rmtree(_tmp_output, ignore_errors=True)

check("the real output directory is restored", folder_paths.get_output_directory(),
      _real_output)



# ── pytest adapter ──────────────────────────────────────────────────────────
#
# Upstream runs this file as a script: it accumulates (ok, name, got, want)
# into _results and exits non-zero at the end. Under pytest that reports the
# whole file as one pass or one fail, which tells you nothing about WHICH
# check broke - so each accumulated check is re-emitted as its own case,
# carrying upstream's own name for it.

import pytest as _pytest  # noqa: E402


@_pytest.mark.parametrize(
    "ok,name,got,want",
    [_pytest.param(*r, id=r[1][:70]) for r in _results],
)
def test_upstream_check(ok, name, got, want):
    assert ok, f"{name}\n  got:  {got!r}\n  want: {want!r}"


def test_the_suite_actually_ran():
    """A file whose checks all vanished would otherwise pass silently."""
    assert len(_results) >= 20, (
        f"only {{len(_results)}} checks ran; the suite was expected to have at "
        f"least 20. Something stopped the module part-way through.")
