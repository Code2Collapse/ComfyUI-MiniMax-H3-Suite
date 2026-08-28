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
    Inputs are a subset check, because widget inputs only appear in the workflow
    once they are converted to sockets.
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
        extra_in = [n for n in got_in if n not in want_in]
        if extra_in:
            problems.append(f"{ntype} (id {node.get('id')}) has inputs not in schema: {extra_in}")

    assert not problems, f"{workflow_file.name}:\n  " + "\n  ".join(problems)
