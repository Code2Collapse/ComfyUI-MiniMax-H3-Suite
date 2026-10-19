"""The 14 task workflows (workflows/tasks, A16 s27): saved graphs stay consistent with the live schemas and documented.

They were built in ComfyUI's own front end and validated by the server (evidence: docs/evidence/A16/workflows in the
C2C workspace). These checks keep them honest when a node changes: real node types, intact link tables, pack outputs in
schema order, a guide note naming models / inputs / controls / output / limits, download links on every model loader,
and no node left bypassed or muted unless its title says it is optional.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.test_workflow_node_ids import _load_pack, _pack_node_ids, test_workflow_links_are_consistent

TASKS = sorted((Path(__file__).resolve().parents[1] / "workflows" / "tasks").glob("*.json"))
CORE = frozenset({
    "UNETLoader", "CLIPLoader", "VAELoader", "LoraLoaderModelOnly", "ModelPatchLoader", "LoadImage", "LoadVideo",
    "LoadAudio", "GetVideoComponents", "ImageToMask", "InvertMask", "ImageScale", "GetImageSize",
    "MiniMaxH3ImageToVideo", "MiniMaxH3ReferenceToVideo", "MiniMaxH3AddGuide", "BasicScheduler", "RandomNoise",
    "KSamplerSelect", "BasicGuider", "SamplerCustomAdvanced", "VAEDecode", "VAEDecodeAudio", "CreateVideo", "SaveVideo",
    "SaveImage", "SaveAudio", "ImageCompositeMasked", "PreviewAny", "MarkdownNote",
})
LOADERS = {"UNETLoader", "CLIPLoader", "VAELoader", "LoraLoaderModelOnly", "ModelPatchLoader"}
OPTIONAL_WORDS = ("optional", "enable", "bypass", "stage b", "turbo", "invert", "first-block", "new sound",
                  "keep picture", "load the saved")


def _load(f):
    return json.loads(f.read_text(encoding="utf-8"))


def _output_names():
    """Pack node_id -> output names exactly as core publishes them (display_name, else the type)."""
    import asyncio
    pack = _load_pack()

    async def _get():
        ext = await pack.comfy_entrypoint()
        out = {}
        for n in await ext.get_node_list():
            s = n.define_schema()
            out[s.node_id] = [o.display_name if o.display_name else o.io_type for o in (s.outputs or [])]
        return out

    return asyncio.run(_get())


def test_there_are_fourteen():
    assert len(TASKS) == 14 and [f.name[:2] for f in TASKS] == [f"{i:02d}" for i in range(1, 15)]


@pytest.mark.parametrize("wf", TASKS, ids=lambda f: f.stem)
def test_types_links_and_pack_outputs(wf):
    data = _load(wf)
    pack, outs = _pack_node_ids(), _output_names()
    unknown = sorted({n["type"] for n in data["nodes"] if n["type"] not in pack and n["type"] not in CORE})
    assert not unknown, f"unknown node types {unknown}"
    test_workflow_links_are_consistent(wf)
    for n in data["nodes"]:
        if n["type"] in outs:
            got = [o.get("name") for o in n.get("outputs") or []]
            assert got == outs[n["type"]], f"{n['type']} outputs {got} != schema {outs[n['type']]}"


@pytest.mark.parametrize("wf", TASKS, ids=lambda f: f.stem)
def test_documented_models_and_optional_nodes(wf):
    data = _load(wf)
    notes = [n for n in data["nodes"] if n["type"] == "MarkdownNote"]
    assert len(notes) == 1
    text = notes[0]["widgets_values"][0]
    for part in ("**Models**", "**Inputs**", "**Controls**", "**Output**", "**Limits**", "**Status**"):
        assert part in text, f"the note lacks {part}"
    for n in data["nodes"]:
        if n["type"] in LOADERS:
            models = (n.get("properties") or {}).get("models") or []
            assert models and models[0]["url"].startswith("https://huggingface.co/"), f"{n['type']} has no link"
        if n.get("mode", 0) in (2, 4):
            title = (n.get("title") or "").lower()
            assert any(w in title for w in OPTIONAL_WORDS) or n["type"] in ("GetVideoComponents",), \
                f"{n['type']} '{n.get('title')}' is switched off without saying it is optional"
    groups = [g["title"] for g in data.get("groups", [])]
    assert groups and groups[0] == "Models"
