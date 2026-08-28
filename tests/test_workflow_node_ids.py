"""Workflow JSON node types must resolve to registered pack or core ComfyUI ids."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / "workflows"

CORE_NODE_IDS = frozenset(
    {
        "EmptyMiniMaxH3LatentAV",
        "MiniMaxH3ReferenceToVideo",
        "MiniMaxH3SigmaShift",
        "MiniMaxH3ImageToVideo",
        "MiniMaxH3AddGuide",
        "BasicScheduler",
        "BasicGuider",
        "SamplerCustomAdvanced",
        "KSampler",
        "LoadImage",
        "VAEEncode",
        "VAEDecode",
    }
)


def _load_pack():
    module_path = str(ROOT)
    sys_module_name = module_path.replace(".", "_x_")
    spec = importlib.util.spec_from_file_location(sys_module_name, ROOT / "__init__.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[sys_module_name] = module
    spec.loader.exec_module(module)
    return module


def _pack_node_ids():
    pack = _load_pack()

    async def _get():
        ext = await pack.comfy_entrypoint()
        nodes = await ext.get_node_list()
        return {n.define_schema().node_id for n in nodes}

    return asyncio.run(_get())


@pytest.mark.parametrize(
    "workflow_file",
    [
        WORKFLOWS / "h3_masked_face_pipeline.json",
        WORKFLOWS / "h3_full_spine_pipeline.json",
    ],
)
def test_workflow_node_types_resolve(workflow_file):
    data = json.loads(workflow_file.read_text(encoding="utf-8"))
    pack_ids = _pack_node_ids()
    unknown: list[str] = []
    for node in data.get("nodes", []):
        ntype = node.get("type")
        if not ntype:
            continue
        if ntype in pack_ids or ntype in CORE_NODE_IDS:
            continue
        unknown.append(ntype)
    assert not unknown, f"{workflow_file.name}: unknown node types {sorted(set(unknown))}"


def _schema_slots():
    """Map pack node_id -> (input ids, output ids) from the live schemas."""
    pack = _load_pack()

    async def _get():
        ext = await pack.comfy_entrypoint()
        nodes = await ext.get_node_list()
        out = {}
        for n in nodes:
            s = n.define_schema()
            out[s.node_id] = (
                [i.id for i in (s.inputs or [])],
                [o.id for o in (s.outputs or [])],
            )
        return out

    return asyncio.run(_get())


@pytest.mark.parametrize(
    "workflow_file",
    [
        WORKFLOWS / "h3_masked_face_pipeline.json",
        WORKFLOWS / "h3_full_spine_pipeline.json",
    ],
)
def test_workflow_links_are_consistent(workflow_file):
    """Every link must reference real nodes and real slots, both ways.

    A workflow is hand-edited whenever a schema gains a socket, and a wrong slot
    index or a half-updated endpoint still loads far enough to look fine while
    silently dropping an edge. This pins the whole link table instead.
    """
    data = json.loads(workflow_file.read_text(encoding="utf-8"))
    nodes = {n["id"]: n for n in data.get("nodes", [])}
    problems: list[str] = []

    seen_ids = set()
    for link in data.get("links", []):
        lid, src, src_slot, dst, dst_slot, ltype = link
        if lid in seen_ids:
            problems.append(f"link {lid}: duplicate id")
        seen_ids.add(lid)

        if src not in nodes:
            problems.append(f"link {lid}: origin node {src} does not exist")
            continue
        if dst not in nodes:
            problems.append(f"link {lid}: target node {dst} does not exist")
            continue

        souts = nodes[src].get("outputs") or []
        dins = nodes[dst].get("inputs") or []
        if not 0 <= src_slot < len(souts):
            problems.append(f"link {lid}: origin slot {src_slot} out of range on node {src}")
            continue
        if not 0 <= dst_slot < len(dins):
            problems.append(f"link {lid}: target slot {dst_slot} out of range on node {dst}")
            continue

        if souts[src_slot].get("type") != ltype:
            problems.append(
                f"link {lid}: type {ltype} != origin slot type {souts[src_slot].get('type')}"
            )
        if dins[dst_slot].get("type") != ltype:
            problems.append(
                f"link {lid}: type {ltype} != target slot type {dins[dst_slot].get('type')}"
            )
        # both endpoints must actually record the edge
        if dins[dst_slot].get("link") != lid:
            problems.append(
                f"link {lid}: target {dst}.{dins[dst_slot].get('name')} records "
                f"link={dins[dst_slot].get('link')}"
            )
        if lid not in (souts[src_slot].get("links") or []):
            problems.append(
                f"link {lid}: origin {src}.{souts[src_slot].get('name')} does not list it"
            )

    # no input may reference a link that isn't in the table
    for nid, node in nodes.items():
        for inp in node.get("inputs") or []:
            lk = inp.get("link")
            if lk is not None and lk not in seen_ids:
                problems.append(f"node {nid}.{inp.get('name')}: dangling link {lk}")

    assert not problems, f"{workflow_file.name}:\n  " + "\n  ".join(problems)


@pytest.mark.parametrize(
    "workflow_file",
    [
        WORKFLOWS / "h3_masked_face_pipeline.json",
        WORKFLOWS / "h3_full_spine_pipeline.json",
    ],
)
def test_workflow_slots_match_live_schema(workflow_file):
    """Saved graphs must not carry stale sockets after a schema change.

    Outputs are pinned exactly (name AND order): that is what goes stale when a
    node gains an output, which is exactly what Stage 4 did to TrackCrop.
    Inputs are pinned by INDEX. An earlier version checked names as a set, which
    passed while MaskedReplace's saved graph omitted `vae` and delivered a MASK
    to slot 2 — the vae socket. Membership is not enough; position is the thing
    links actually use.
    """
    data = json.loads(workflow_file.read_text(encoding="utf-8"))
    slots = _schema_slots()
    problems: list[str] = []

    for node in data.get("nodes", []):
        ntype = node.get("type")
        if ntype not in slots:
            continue  # core node, covered by test_workflow_node_types_resolve
        want_in, want_out = slots[ntype]
        got_out = [o.get("name") for o in (node.get("outputs") or [])]
        got_in = [i.get("name") for i in (node.get("inputs") or [])]
        if got_out != want_out:
            problems.append(f"{ntype} (id {node.get('id')}) outputs {got_out} != schema {want_out}")

        # INDEX, not just membership. Links address inputs by position, so a name
        # that exists but sits at the wrong index silently delivers the value to a
        # different socket — a MASK arriving on `vae` typechecks as far as the JSON
        # is concerned. A subset check by name cannot see that; this can.
        for k, name in enumerate(got_in):
            if name not in want_in:
                problems.append(f"{ntype} (id {node.get('id')}) input {name!r} is not in schema")
            elif want_in.index(name) != k:
                problems.append(
                    f"{ntype} (id {node.get('id')}) input {name!r} at slot {k}, "
                    f"schema puts it at {want_in.index(name)} "
                    f"(slot {k} is {want_in[k] if k < len(want_in) else 'out of range'!r})"
                )

    assert not problems, f"{workflow_file.name}:\n  " + "\n  ".join(problems)


@pytest.mark.parametrize(
    "workflow_file",
    [
        WORKFLOWS / "h3_masked_face_pipeline.json",
        WORKFLOWS / "h3_full_spine_pipeline.json",
    ],
)
def test_workflow_widget_values_are_legal(workflow_file):
    """Combo widgets in saved graphs must hold a value the schema actually offers.

    A saved workflow stores widget values positionally, so a copy-paste can leave a
    value from a *different* combo in the slot. It still loads, and the node then
    falls through to whatever its default branch is: `canvas_mode="tv_lp"` (a
    planner_mode value) silently became auto-canvas instead of the manual 512x512
    the rest of the graph was built around.
    """
    pack = _load_pack()

    async def _combos():
        ext = await pack.comfy_entrypoint()
        out = {}
        for n in await ext.get_node_list():
            s = n.define_schema()
            # widgets are stored positionally, skipping socket-only inputs
            out[s.node_id] = [
                (i.id, list(getattr(i, "options", []) or []))
                for i in (s.inputs or [])
            ]
        return out

    specs = asyncio.run(_combos())
    data = json.loads(workflow_file.read_text(encoding="utf-8"))
    problems: list[str] = []

    for node in data.get("nodes", []):
        spec = specs.get(node.get("type"))
        if not spec:
            continue
        options_by_name = {name: opts for name, opts in spec if opts}
        if not options_by_name:
            continue
        values = node.get("widgets_values") or []
        # any stored value that looks like a combo choice must belong to SOME combo
        # on this node; a value that belongs to none of them is a positional slip
        legal = {v for opts in options_by_name.values() for v in opts}
        for v in values:
            if isinstance(v, str) and v and not v.startswith("[") and v not in legal:
                problems.append(
                    f"{node['type']} (id {node.get('id')}) widget value {v!r} "
                    f"is not an option of any combo on this node: {sorted(legal)}"
                )

    assert not problems, f"{workflow_file.name}:\n  " + "\n  ".join(problems)
