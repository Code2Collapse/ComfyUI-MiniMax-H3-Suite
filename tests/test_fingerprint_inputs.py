"""Every node's fingerprint_inputs must work the way ComfyUI calls it (A16 audit).

ComfyUI calls IS_CHANGED / fingerprint_inputs with WIDGET VALUES ONLY - linked inputs (latents, images, models) are
never passed (execution.py IsChangedCache: "We only want constants in IS_CHANGED"). A fingerprint that names a linked
input raises TypeError there, ComfyUI logs a warning and stores NaN, and the node plus everything after it re-runs on
every queue (the sampler included). Seven nodes did that (continuation.py x5, differential_denoise, tone_compensate),
with id() of the tensor as the key. This test calls each fingerprint exactly like ComfyUI does.
"""

from __future__ import annotations

import asyncio
import inspect

import pytest

pytest.importorskip("comfy_api.latest", reason="needs ComfyUI")

WIDGET_TYPES = {"INT", "FLOAT", "STRING", "BOOLEAN", "COMBO"}


def _nodes():
    import importlib.util
    import sys
    from pathlib import Path
    pkg = Path(__file__).resolve().parents[1]
    name = str(pkg).replace(".", "_x_") + "_fingerprint_probe"
    spec = importlib.util.spec_from_file_location(name, pkg / "__init__.py", submodule_search_locations=[str(pkg)])
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    ext = asyncio.run(mod.comfy_entrypoint())
    return asyncio.run(ext.get_node_list())


def _widget_kwargs(schema):
    """What ComfyUI hands a V3 fingerprint: widget values, and None for every linked input (measured live on core
    0.39: the log reads "WARNING: 'NoneType' object has no attribute 'cpu'" for a fingerprint that hashes one)."""
    kw = {}
    for inp in schema.inputs or []:
        io_type = str(getattr(inp, "io_type", "")).upper()
        if io_type not in WIDGET_TYPES or getattr(inp, "force_input", False):
            kw[inp.id] = None
            continue
        default = getattr(inp, "default", None)
        if default is None and io_type == "COMBO":
            opts = getattr(inp, "options", None) or []
            default = opts[0] if opts else None
        if default is None:
            default = {"INT": 0, "FLOAT": 0.0, "STRING": "", "BOOLEAN": False}.get(io_type)
        kw[inp.id] = default
    return kw


def test_fingerprints_take_widget_values_only():
    bad = []
    for cls in _nodes():
        fp = getattr(cls, "fingerprint_inputs", None)
        if fp is None or getattr(fp, "__func__", None) is getattr(getattr(cls.__mro__[-2], "fingerprint_inputs", None),
                                                                   "__func__", object()):
            continue
        try:
            schema = cls.define_schema()
        except Exception:  # noqa: BLE001 - schema errors are test_registration's job
            continue
        own = "fingerprint_inputs" in cls.__dict__
        if not own:
            continue
        kw = _widget_kwargs(schema)
        try:
            res = fp(**kw)
            if inspect.isawaitable(res):
                asyncio.run(res)
        except Exception as exc:  # noqa: BLE001 - ANY exception here is ComfyUI's NaN: the node never caches
            bad.append(f"{schema.node_id}: {type(exc).__name__}: {exc}")
    assert not bad, "fingerprints that ComfyUI cannot call (node re-runs every queue):\n" + "\n".join(bad)
