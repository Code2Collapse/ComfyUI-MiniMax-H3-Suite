"""Schema additions for Stage 4 frontend widgets — regression on output order."""

from __future__ import annotations

import asyncio
import builtins
import importlib.util
import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
WEB_DIR = (ROOT / "web").resolve()


def _load_pack():
    module_path = str(ROOT)
    sys_module_name = module_path.replace(".", "_x_")
    spec = importlib.util.spec_from_file_location(sys_module_name, ROOT / "__init__.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[sys_module_name] = module
    spec.loader.exec_module(module)
    return module


def _nodes():
    pack = _load_pack()

    async def _get():
        ext = await pack.comfy_entrypoint()
        return await ext.get_node_list()

    return asyncio.run(_get())


def _schema(node_id: str):
    for n in _nodes():
        s = n.define_schema()
        if s.node_id == node_id:
            return s
    pytest.fail(f"node {node_id!r} not registered")


def _path_under_web(p: Path) -> bool:
    try:
        p.resolve().relative_to(WEB_DIR)
        return True
    except ValueError:
        return False


def test_track_crop_boxes_json_output_appended():
    s = _schema("MiniMaxH3_TrackCrop")
    names = [o.id for o in s.outputs]
    assert names == ["crops", "transform", "preview", "report", "boxes_json"]
    assert names[-1] == "boxes_json"


def test_sigma_inspector_sigma_json_output_appended():
    s = _schema("MiniMaxH3_SigmaInspector")
    names = [o.id for o in s.outputs]
    assert names == ["report", "sigma_json"]
    assert names[-1] == "sigma_json"


def test_headless_schemas_build_without_web_reads():
    """F-INV1 — schema construction must not read web/ (JS is browser-only)."""
    read_paths: list[str] = []
    orig_open = builtins.open
    orig_read_text = Path.read_text
    orig_read_bytes = Path.read_bytes
    orig_path_open = Path.open

    def record(path: Path) -> None:
        if _path_under_web(path):
            read_paths.append(str(path.resolve()))

    def tracking_open(file, *args, **kwargs):
        if not isinstance(file, int):
            record(Path(file))
        return orig_open(file, *args, **kwargs)

    def tracking_read_text(self, *args, **kwargs):
        record(self)
        return orig_read_text(self, *args, **kwargs)

    def tracking_read_bytes(self, *args, **kwargs):
        record(self)
        return orig_read_bytes(self, *args, **kwargs)

    def tracking_path_open(self, *args, **kwargs):
        record(self)
        return orig_path_open(self, *args, **kwargs)

    with patch("builtins.open", tracking_open), patch.object(
        Path, "read_text", tracking_read_text
    ), patch.object(Path, "read_bytes", tracking_read_bytes), patch.object(
        Path, "open", tracking_path_open
    ):
        for node in _nodes():
            schema = node.define_schema()
            assert schema.node_id
            assert schema.outputs

    assert read_paths == [], f"schema build read web/: {read_paths}"


def test_sigma_inspector_emits_ui_payload():
    from mmx_nodes.sigma_inspector import MiniMaxH3_SigmaInspector

    out = MiniMaxH3_SigmaInspector.execute(torch.tensor([1.0, 0.5, 0.0]))
    assert out.ui is not None
    assert "mmx_sigma" in out.ui
    assert out.ui["mmx_sigma"][0]


def test_all_four_widget_nodes_emit_ui_payloads():
    """Bug #14 regression, for all four widget nodes rather than one.

    Only NodeOutput.ui reaches the browser (execution.py:377-381, :563). If any of
    these stops emitting its key, its widget goes back to sitting on the loading
    spinner forever — and nothing else in this suite would notice, because the
    socket outputs would still be perfectly correct.
    """
    from mmx_nodes.drift_qc import MiniMaxH3_DriftQC
    from mmx_nodes.mask_prep import MiniMaxH3_MaskPrep
    from mmx_nodes.sigma_inspector import MiniMaxH3_SigmaInspector
    from mmx_nodes.track_crop import MiniMaxH3_TrackCrop

    torch.manual_seed(0)

    images = torch.rand(22, 128, 128, 3)
    boxes = json.dumps([[40.0, 40.0, 48.0, 48.0]] * 22)
    tc = MiniMaxH3_TrackCrop.execute(
        images, bboxes_json=boxes, canvas_width=512, canvas_height=512
    )
    assert tc.ui is not None and "mmx_boxes" in tc.ui, "W1 lost its trajectory payload"
    assert "images" in tc.ui, "W1 lost its preview backdrop"
    assert json.loads(tc.ui["mmx_boxes"][0])["boxes"], "mmx_boxes carries no boxes"

    mask = torch.zeros(22, 128, 128)
    mask[:, 40:88, 40:88] = 1.0
    mp = MiniMaxH3_MaskPrep.execute(mask, width=512, height=512, frame_count=22)
    assert mp.ui is not None
    for key in ("mmx_pixel_mask", "mmx_token_preview"):
        assert key in mp.ui, f"W2 lost {key}"

    si = MiniMaxH3_SigmaInspector.execute(torch.tensor([1.0, 0.5, 0.0]))
    assert si.ui is not None and si.ui["mmx_sigma"][0], "W3 lost its sigma payload"

    original = torch.rand(2, 192, 192, 3)
    dq = MiniMaxH3_DriftQC.execute(original, original.clone(), torch.zeros(2, 192, 192), 2.0)
    assert dq.ui is not None and "mmx_drift" in dq.ui, "W4 lost its drift payload"
    payload = json.loads(dq.ui["mmx_drift"][0])
    assert "drift_px" in payload and "passed" in payload
