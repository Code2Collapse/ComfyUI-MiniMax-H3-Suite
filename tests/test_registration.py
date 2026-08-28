"""Registration smoke test.

Loads the pack EXACTLY the way ComfyUI does, because a subtly different loader
gives a subtly wrong answer. `ComfyUI/nodes.py:2243-2263`:

    sys_module_name = module_path.replace(".", "_x_")     # :2250
    module_spec = importlib.util.spec_from_file_location(sys_module_name, .../__init__.py)
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[sys_module_name] = module                 # :2262  <-- BEFORE exec
    module_spec.loader.exec_module(module)                # :2263

The `sys.modules` assignment BEFORE `exec_module` is the part that matters: without
it the package's own submodule imports cannot resolve, every guarded node import
fails, and the pack loads with ZERO nodes while looking perfectly healthy. That is
the real-world "pack vanished from /object_info" failure, so the test must be able
to catch it — which means reproducing the loader faithfully.
"""

import asyncio
import importlib
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

EXPECTED_NODE_COUNT = 25
SPINE_NODES = {
    "MiniMaxH3_TrackCrop",
    "MiniMaxH3_MaskPrep",
    "MiniMaxH3_StitchBack",
    "MiniMaxH3_ControlHints",
    "MiniMaxH3_MaskedReplace",
    "MiniMaxH3_DetailReinject",
    "MiniMaxH3_FrameHandles",
}
SAMPLING_NODES = {
    "MiniMaxH3_ProtectedLayerGuard",
    "MiniMaxH3_AcceleratorConflict",
    "MiniMaxH3_LegalScheduler",
    "MiniMaxH3_SigmaShiftLocked",
    "MiniMaxH3_PerFrameDenoise",
    "MiniMaxH3_BlockCacheT8",
    "MiniMaxH3_SigmaInspector",
}
COLOR_NODES = {
    "MiniMaxH3_OCIOBridge",
    "MiniMaxH3_ColorRoundTripQC",
    "MiniMaxH3_HDRRoundtrip",
}
CONTROL_NODES = {
    "MiniMaxH3_StrongestPose",
    "MiniMaxH3_TemporalDepth",
    "MiniMaxH3_RegionMask",
    "MiniMaxH3_PosePuppeteer",
}
QC_NODES = {
    "MiniMaxH3_DriftQC",
}
AUTHORING_NODES = {
    "MiniMaxH3_FamilyPresets",
    "MiniMaxH3_EditValidator",
}
BRIDGE_NODES = {
    "MiniMaxH3_DCCBridge",
}


def _load_pack_like_comfyui():
    module_path = str(ROOT)
    sys_module_name = module_path.replace(".", "_x_")
    spec = importlib.util.spec_from_file_location(
        sys_module_name, ROOT / "__init__.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[sys_module_name] = module          # BEFORE exec_module — see docstring
    spec.loader.exec_module(module)
    return module


def _node_list(pack):
    async def _get():
        return await (await pack.comfy_entrypoint()).get_node_list()

    return asyncio.run(_get())


def _comfyui_root() -> Path:
    for candidate in (
        ROOT.parent.parent / "ComfyUI_windows_portable" / "ComfyUI",
        ROOT.parent / "third_party" / "ComfyUI",
    ):
        if (candidate / "utils" / "json_util.py").is_file():
            return candidate
    pytest.skip("ComfyUI tree not found for production registration test")


def test_registers_even_when_comfyui_utils_already_imported():
    """ComfyUI imports its own top-level `utils` package at startup, before custom
    nodes load. If our package dirs were named `utils`/`nodes`, sys.modules would
    already hold ComfyUI's and every node import would fail — registering 0 nodes
    while the test suite stayed green. Measured before the rename: 0/19 registered.
    """
    comfy = _comfyui_root()
    comfy_path = str(comfy)
    if comfy_path not in sys.path:
        sys.path.insert(0, comfy_path)

    # Poison sys.modules under BOTH colliding names, the way a running ComfyUI does.
    #
    # ComfyUI's own `nodes.py` cannot be imported on this box — it pulls in
    # comfy.ldm.modules.attention, which hits the comfy_kitchen version skew
    # (AttributeError: no attribute 'int8_attention_is_available'). That is an
    # environment artefact, not a property of our pack, and it must not be allowed to
    # mask the thing under test. What matters is that SOME foreign module already owns
    # the names `utils` and `nodes`; where it came from is irrelevant. So use the real
    # ComfyUI module when it imports, and a stand-in when it does not.
    import types

    for name, real_import in (("utils", "utils.json_util"), ("nodes", "nodes")):
        if name in sys.modules and getattr(sys.modules[name], "__file__", "") and (
            "ComfyUI-MiniMaxSuite" not in str(sys.modules[name].__file__)
        ):
            continue
        try:
            importlib.import_module(real_import)
        except Exception:
            sys.modules[name] = types.ModuleType(name)  # foreign occupant stand-in

    for name in ("utils", "nodes"):
        occupant = sys.modules.get(name)
        assert occupant is not None, f"{name} should be occupied before the test"
        assert "ComfyUI-MiniMaxSuite" not in str(getattr(occupant, "__file__", "") or ""), (
            f"sys.modules[{name!r}] is OUR module — the collision this test guards "
            "against is no longer being simulated"
        )

    pack = _load_pack_like_comfyui()
    assert len(_node_list(pack)) == EXPECTED_NODE_COUNT


def test_all_nodes_register():
    pack = _load_pack_like_comfyui()
    nodes = _node_list(pack)
    ids = {n.define_schema().node_id for n in nodes}

    missing = (
        SPINE_NODES
        | SAMPLING_NODES
        | COLOR_NODES
        | CONTROL_NODES
        | QC_NODES
        | AUTHORING_NODES
        | BRIDGE_NODES
    ) - ids
    assert not missing, f"nodes failed to register: {sorted(missing)}"
    assert len(nodes) == EXPECTED_NODE_COUNT, sorted(ids)


def test_does_not_shadow_core_node_ids():
    """Our nodes must not collide with ComfyUI's own H3 node ids.

    `MiniMaxH3SigmaShift` is a real core node (`comfy_extras/nodes_minimax_h3.py:369`).
    Ours is `MiniMaxH3_SigmaShiftLocked` — the underscore prefix is the whole
    convention that keeps us out of core's namespace.
    """
    pack = _load_pack_like_comfyui()
    ids = {n.define_schema().node_id for n in _node_list(pack)}

    core_ids = {
        "EmptyMiniMaxH3LatentAV",
        "MiniMaxH3ImageToVideo",
        "MiniMaxH3AddGuide",
        "MiniMaxH3ReferenceToVideo",
        "MiniMaxH3SigmaShift",
    }
    assert not (ids & core_ids), f"shadows core node ids: {sorted(ids & core_ids)}"
    assert all(i.startswith("MiniMaxH3_") for i in ids), sorted(ids)


def test_every_node_has_a_category_and_outputs():
    pack = _load_pack_like_comfyui()
    for node in _node_list(pack):
        s = node.define_schema()
        assert s.category, f"{s.node_id} has no category"
        assert s.category.startswith("MiniMax H3/"), f"{s.node_id}: {s.category}"
        assert s.outputs, f"{s.node_id} declares no outputs"


def test_web_directory_configured():
    pack = _load_pack_like_comfyui()
    assert getattr(pack, "WEB_DIRECTORY", None) == "web"
    web_dir = ROOT / "web"
    assert web_dir.is_dir(), "WEB_DIRECTORY must point at an existing folder"
    assert (web_dir / "shared.js").is_file()
    for name in (
        "w1_box_trajectory.js",
        "w2_token_grid.js",
        "w3_sigma_plot.js",
        "w4_drift_curve.js",
    ):
        assert (web_dir / name).is_file(), f"missing web/{name}"
