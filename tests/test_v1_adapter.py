"""Adapting a classic-API node into this pack's V3 registration.

The adapter exists because of a loader detail with a very expensive failure
mode. ComfyUI reads:

    if hasattr(module, "NODE_CLASS_MAPPINGS") ...:
        ...
        return True
    elif hasattr(module, "comfy_entrypoint"):

The V1 branch RETURNS. A pack that exports both gets V1 and its
`comfy_entrypoint` is never called - so adding one mapping to register one
classic node would silently unregister every V3 node in the pack, and the only
symptom is a mostly-empty menu with nothing in the log.

So classic nodes are adapted instead, with the schema DERIVED from
`INPUT_TYPES()` rather than transcribed: a hand-copied schema drifts from the
implementation the first time either is touched, and the drift shows up as a
widget that silently stops being read.

CPU-only, no ComfyUI runtime needed beyond comfy_api.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PACK = Path(__file__).resolve().parents[1]
if str(PACK) not in sys.path:
    sys.path.insert(0, str(PACK))

adapter = pytest.importorskip(
    "mmx_nodes.director.v1_adapter", reason="needs comfy_api")
adapt = adapter.adapt
V1AdapterError = adapter.V1AdapterError


class Classic:
    """A classic node exercising every shape the adapter has to handle."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "mode": (["a", "b", "c"], {"default": "b"}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 100}),
                "scale": ("FLOAT", {"default": 1.5, "min": 0.0, "max": 8.0, "step": 0.1}),
                "flag": ("BOOLEAN", {"default": True, "tooltip": "a flag"}),
                "text": ("STRING", {"default": "hi", "multiline": True}),
            },
            "optional": {
                "extra": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("picture", "note")
    FUNCTION = "run"
    CATEGORY = "Old/Place"
    DESCRIPTION = "a classic node"

    def run(self, model, mode, steps, scale, flag, text, extra=None):
        return (f"{model}|{mode}|{steps}|{scale}|{flag}|{text}|{extra}", "ok")


@pytest.fixture
def wrapped():
    return adapt(Classic, node_id="Test_Adapted", display_name="Adapted",
                 category="New/Place")


# ── the schema comes from the implementation ────────────────────────────────

def test_every_declared_input_reaches_the_schema():
    """A dropped input is a widget that silently stops being read."""
    w = adapt(Classic, node_id="x", display_name="x", category="x")
    ids = {i.id for i in w.define_schema().inputs}
    assert ids == {"model", "mode", "steps", "scale", "flag", "text", "extra"}


def test_required_and_optional_are_kept_apart(wrapped):
    by_id = {i.id: i for i in wrapped.define_schema().inputs}
    assert by_id["extra"].optional is True
    assert by_id["model"].optional is False


def test_a_list_in_the_type_slot_becomes_a_combo(wrapped):
    mode = next(i for i in wrapped.define_schema().inputs if i.id == "mode")
    assert list(mode.options) == ["a", "b", "c"]
    assert mode.default == "b"


def test_a_combo_with_no_declared_default_takes_its_first_option():
    class NoDefault(Classic):
        @classmethod
        def INPUT_TYPES(cls):
            return {"required": {"pick": (["x", "y"],)}}

    w = adapt(NoDefault, node_id="n", display_name="n", category="n")
    pick = next(i for i in w.define_schema().inputs if i.id == "pick")
    assert pick.default == "x", "a combo with no default would show empty"


def test_numeric_bounds_survive(wrapped):
    by_id = {i.id: i for i in wrapped.define_schema().inputs}
    assert (by_id["steps"].min, by_id["steps"].max) == (1, 100)
    assert by_id["scale"].max == pytest.approx(8.0)
    assert by_id["scale"].default == pytest.approx(1.5)


def test_tooltips_survive(wrapped):
    flag = next(i for i in wrapped.define_schema().inputs if i.id == "flag")
    assert flag.tooltip == "a flag"


def test_outputs_keep_their_names_and_order(wrapped):
    outs = wrapped.define_schema().outputs
    assert [o.display_name for o in outs] == ["picture", "note"]


def test_the_node_identity_is_what_was_asked_for(wrapped):
    s = wrapped.define_schema()
    assert s.node_id == "Test_Adapted"
    assert s.display_name == "Adapted"
    assert s.category == "New/Place", "the upstream category must not leak through"


# ── options the V3 API cannot take ──────────────────────────────────────────

def test_classic_only_options_are_dropped_rather_than_raising():
    """forceInput, dynamicPrompts and friends have no V3 equivalent. Passing
    them through is a TypeError at schema build, which takes the node out."""
    class Legacy(Classic):
        @classmethod
        def INPUT_TYPES(cls):
            return {"required": {
                "t": ("STRING", {"default": "", "forceInput": True,
                                 "dynamicPrompts": False, "multiline": True}),
                "n": ("INT", {"default": 1, "control_after_generate": True}),
            }}

    w = adapt(Legacy, node_id="l", display_name="l", category="l")
    assert len(w.define_schema().inputs) == 2


def test_multiline_is_not_passed_to_a_non_string_input():
    """It is a String-only option; anywhere else it is a TypeError."""
    class Odd(Classic):
        @classmethod
        def INPUT_TYPES(cls):
            return {"required": {"n": ("INT", {"default": 1, "multiline": True})}}

    w = adapt(Odd, node_id="o", display_name="o", category="o")
    assert len(w.define_schema().inputs) == 1


# ── refusing to guess ───────────────────────────────────────────────────────

def test_an_unknown_input_type_is_named_not_guessed():
    """A generic fallback socket fails at connect time with nothing to read."""
    class Weird(Classic):
        @classmethod
        def INPUT_TYPES(cls):
            return {"required": {"thing": ("SOME_CUSTOM_TYPE",)}}

    with pytest.raises(V1AdapterError, match="SOME_CUSTOM_TYPE"):
        adapt(Weird, node_id="w", display_name="w", category="w")


def test_an_unknown_output_type_is_named():
    class Weird(Classic):
        RETURN_TYPES = ("SOME_CUSTOM_TYPE",)

    with pytest.raises(V1AdapterError, match="unknown"):
        adapt(Weird, node_id="w", display_name="w", category="w")


def test_a_malformed_input_spec_is_named():
    class Broken(Classic):
        @classmethod
        def INPUT_TYPES(cls):
            return {"required": {"bad": "IMAGE"}}   # a string, not a tuple

    with pytest.raises(V1AdapterError, match="bad"):
        adapt(Broken, node_id="b", display_name="b", category="b")


# ── execution ───────────────────────────────────────────────────────────────

def test_execute_delegates_and_returns_a_node_output(wrapped):
    out = wrapped.execute(model="M", mode="b", steps=7, scale=2.0, flag=False,
                          text="hello", extra=None)
    assert out.args[0] == "M|b|7|2.0|False|hello|None"
    assert out.args[1] == "ok"


def test_each_run_gets_a_fresh_instance():
    """The classic API allows per-call state on self; a shared instance would
    leak it from one execution into the next.

    The instances are HELD, not compared by id(): the first one is garbage
    two lines later and CPython hands the same address to the second, so an
    id() comparison reports one instance for code that correctly makes two.
    """
    seen = []

    class Stateful(Classic):
        def run(self, **kw):
            seen.append(self)
            return ("x", "y")

    w = adapt(Stateful, node_id="s", display_name="s", category="s")
    kw = dict(model="M", mode="a", steps=1, scale=1.0, flag=True, text="t")
    w.execute(**kw)
    w.execute(**kw)
    assert len(seen) == 2
    assert seen[0] is not seen[1], "the same instance was reused between runs"


def test_state_written_on_self_does_not_survive_into_the_next_run():
    """The behavioural version of the test above, which is what actually
    matters: a counter on self must start from nothing every execution."""
    counts = []

    class Counter(Classic):
        def run(self, **kw):
            self.n = getattr(self, "n", 0) + 1
            counts.append(self.n)
            return ("x", "y")

    w = adapt(Counter, node_id="c", display_name="c", category="c")
    kw = dict(model="M", mode="a", steps=1, scale=1.0, flag=True, text="t")
    w.execute(**kw)
    w.execute(**kw)
    assert counts == [1, 1], f"state leaked between runs: {counts}"


def test_a_bare_value_is_wrapped_into_a_tuple():
    class Single(Classic):
        RETURN_TYPES = ("STRING",)
        RETURN_NAMES = ("only",)

        def run(self, **kw):
            return "just one"

    w = adapt(Single, node_id="s1", display_name="s1", category="s1")
    assert w.execute(model="M", mode="a", steps=1, scale=1.0, flag=True,
                     text="t").args[0] == "just one"


# ── the node this was built for ─────────────────────────────────────────────

def test_the_all_in_one_director_adapts():
    director = pytest.importorskip(
        "mmx_nodes.director", reason="Director needs ComfyUI importable")
    node = getattr(director, "MiniMaxH3_DirectorAllInOne", None)
    if node is None:
        pytest.skip("all-in-one Director did not import")
    s = node.define_schema()
    assert s.node_id == "MiniMaxH3_DirectorAllInOne"
    ids = {i.id for i in s.inputs}
    assert "prompt" in ids, (
        "the added prompt input is missing - without it the node generates "
        "from an empty prompt, because upstream takes every word from the "
        "timeline JSON that this pack does not yet write")
    assert {"model", "clip", "vae", "audio_vae"} <= ids
    assert [o.display_name for o in s.outputs] == ["images", "audio", "compiled_prompt"]


def test_the_pack_root_must_not_export_node_class_mappings():
    """THE reason the adapter exists. ComfyUI's loader takes the V1 branch and
    RETURNS if a pack exports NODE_CLASS_MAPPINGS, so exporting one would stop
    comfy_entrypoint being called and silently unregister every node here."""
    root = (PACK / "__init__.py").read_text(encoding="utf-8")
    lines = [ln for ln in root.splitlines()
             if ln.lstrip().startswith("NODE_CLASS_MAPPINGS")
             or ln.lstrip().startswith("NODE_DISPLAY_NAME_MAPPINGS")]
    assert not lines, (
        "the pack root exports " + "; ".join(lines) + " - that makes ComfyUI "
        "take the V1 branch and never call comfy_entrypoint, which unregisters "
        "every node in this pack")
