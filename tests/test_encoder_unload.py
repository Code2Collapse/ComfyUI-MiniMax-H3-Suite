"""Freeing one named text encoder.

Every failure here is a node that reports success and frees nothing, which on
a tight card looks like the node not working rather than like a bug:

  * --gpu-only: load and offload device are the same accelerator, so the
    weights move from the GPU to the GPU
  * clones: a LoRA-patched encoder is a CLONE, and unloading only the original
    leaves the clone holding the weights
  * caching: the unload is a side effect, so a cache hit skips it entirely

CPU-only, no models, no ComfyUI runtime beyond comfy_api.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

PACK = Path(__file__).resolve().parents[1]
if str(PACK) not in sys.path:
    sys.path.insert(0, str(PACK))

from mmx_utils.encoder_unload import (  # noqa: E402
    EncoderUnloadError,
    check_can_leave_accelerator,
    describe,
    unload_clip,
)
import mmx_utils.encoder_unload as eu  # noqa: E402


class Patcher:
    def __init__(self, load="cuda:0", offload="cpu"):
        self.load_device = load
        self.offload_device = offload


class Clip:
    def __init__(self, patcher=None):
        self.patcher = patcher


@pytest.fixture
def mm(monkeypatch):
    """A stand-in for comfy.model_management that records what it was asked."""
    calls = {"unload": [], "empty": 0}

    def unload_model_and_clones(patcher, all_devices=False):
        calls["unload"].append((patcher, all_devices))

    def soft_empty_cache(force=False):
        calls["empty"] += 1

    monkeypatch.setattr(eu, "model_management", types.SimpleNamespace(
        unload_model_and_clones=unload_model_and_clones,
        soft_empty_cache=soft_empty_cache))
    return calls


# ── the --gpu-only case ─────────────────────────────────────────────────────

def test_an_encoder_with_nowhere_to_go_is_refused():
    """With --gpu-only, load and offload are the same accelerator: an unload
    moves the weights from the GPU to the GPU and frees nothing. Reporting
    success there is the worst outcome - the next step still runs out of
    memory and this node looks innocent."""
    with pytest.raises(EncoderUnloadError, match="gpu-only"):
        check_can_leave_accelerator(Clip(Patcher(load="cuda:0", offload="cuda:0")))


def test_a_normal_gpu_to_cpu_offload_is_allowed():
    check_can_leave_accelerator(Clip(Patcher(load="cuda:0", offload="cpu")))


def test_a_cpu_only_setup_is_not_refused():
    """Both devices are the CPU, which is not the --gpu-only trap - there is
    simply no accelerator memory to free, and that is fine."""
    check_can_leave_accelerator(Clip(Patcher(load="cpu", offload="cpu")))


def test_a_meta_device_is_not_refused():
    check_can_leave_accelerator(Clip(Patcher(load="meta", offload="meta")))


def test_a_clip_with_no_patcher_defers_to_the_real_check():
    """One message, in one place - unload_clip() gives it."""
    check_can_leave_accelerator(Clip(None))


# ── the unload itself ───────────────────────────────────────────────────────

def test_clones_are_unloaded_too(mm):
    """THE quiet failure. A LoRA-patched encoder is a clone; unloading only
    the original leaves the clone holding the weights, so nothing is freed
    and the node still reports success."""
    patcher = Patcher()
    unload_clip(Clip(patcher))
    assert mm["unload"] == [(patcher, True)], "all_devices was not requested"


def test_the_allocator_cache_is_returned(mm):
    unload_clip(Clip(Patcher()))
    assert mm["empty"] == 1


def test_the_cache_is_returned_even_when_the_unload_raises(monkeypatch):
    """Otherwise a part-way failure leaves freed blocks held by the allocator,
    making the failure worse than it needed to be."""
    calls = {"empty": 0}

    def boom(patcher, all_devices=False):
        raise RuntimeError("something went wrong")

    def soft_empty_cache(force=False):
        calls["empty"] += 1

    monkeypatch.setattr(eu, "model_management", types.SimpleNamespace(
        unload_model_and_clones=boom, soft_empty_cache=soft_empty_cache))
    with pytest.raises(RuntimeError, match="something went wrong"):
        unload_clip(Clip(Patcher()))
    assert calls["empty"] == 1


def test_an_older_signature_without_all_devices_still_works(monkeypatch):
    """Not every ComfyUI build takes all_devices. Passing it blindly is a
    TypeError; refusing to run at all would be worse than a narrower unload."""
    seen = []

    def unload_model_and_clones(patcher):          # no all_devices
        seen.append(patcher)

    monkeypatch.setattr(eu, "model_management", types.SimpleNamespace(
        unload_model_and_clones=unload_model_and_clones,
        soft_empty_cache=lambda force=False: None))
    patcher = Patcher()
    unload_clip(Clip(patcher))
    assert seen == [patcher]


def test_a_kwargs_signature_gets_all_devices(monkeypatch):
    seen = []

    def unload_model_and_clones(patcher, **kw):
        seen.append(kw)

    monkeypatch.setattr(eu, "model_management", types.SimpleNamespace(
        unload_model_and_clones=unload_model_and_clones,
        soft_empty_cache=lambda force=False: None))
    unload_clip(Clip(Patcher()))
    assert seen == [{"all_devices": True}]


# ── refusing rather than pretending ─────────────────────────────────────────

def test_a_clip_with_no_patcher_is_named(mm):
    with pytest.raises(EncoderUnloadError, match="no patcher"):
        unload_clip(Clip(None))


def test_missing_model_management_is_named(monkeypatch):
    monkeypatch.setattr(eu, "model_management", None)
    with pytest.raises(EncoderUnloadError, match="model management"):
        unload_clip(Clip(Patcher()))


def test_an_old_comfyui_says_what_version_is_needed(monkeypatch):
    """Older builds can only free everything at once, which would take the
    diffusion model with it - so this refuses rather than doing that."""
    monkeypatch.setattr(eu, "model_management", types.SimpleNamespace())
    with pytest.raises(EncoderUnloadError, match="0.23.0"):
        unload_clip(Clip(Patcher()))


# ── the report ──────────────────────────────────────────────────────────────

def test_the_report_names_the_device_it_freed_from():
    assert "cuda:0" in describe(Clip(Patcher(load="cuda:0")))


def test_the_report_says_what_was_not_touched():
    """A user watching nvidia-smi needs to know this is not a general free."""
    text = describe(Clip(Patcher()))
    assert "diffusion model" in text
    assert "clones" in text
    assert "zero VRAM" in text


def test_the_report_survives_a_clip_with_no_patcher():
    assert "released" in describe(Clip(None))


# ── the node ────────────────────────────────────────────────────────────────

def test_the_node_never_caches():
    """The unload is a side effect, not a value. A cache hit would skip it, so
    a second run - or a run after another branch reloaded the encoder - would
    free nothing while reporting success. NaN is correct here and is the one
    place in this pack where it is."""
    node = pytest.importorskip("mmx_nodes.encoder_unload",
                               reason="needs comfy_api")
    import math
    cls = node.MiniMaxH3_TextEncoderUnload
    assert math.isnan(cls.IS_CHANGED())
    assert math.isnan(cls.fingerprint_inputs())


def test_the_node_passes_conditioning_through_unchanged(mm):
    node = pytest.importorskip("mmx_nodes.encoder_unload",
                               reason="needs comfy_api")
    pos, neg = ["positive"], ["negative"]
    out = node.MiniMaxH3_TextEncoderUnload.execute(
        positive=pos, clip=Clip(Patcher()), negative=neg)
    assert out.args[0] is pos, "the conditioning was not passed through"
    assert out.args[1] is neg
    assert "released" in out.args[2]


def test_an_unconnected_negative_blocks_that_branch_rather_than_faking_one(mm):
    """Returning None or an empty conditioning would be accepted by a sampler
    and silently generate against nothing."""
    node = pytest.importorskip("mmx_nodes.encoder_unload",
                               reason="needs comfy_api")
    out = node.MiniMaxH3_TextEncoderUnload.execute(
        positive=["p"], clip=Clip(Patcher()), negative=None)
    blocked = out.args[1]
    assert blocked is not None
    assert type(blocked).__name__ == "ExecutionBlocker"


def test_the_node_is_in_the_sampling_category():
    node = pytest.importorskip("mmx_nodes.encoder_unload",
                               reason="needs comfy_api")
    s = node.MiniMaxH3_TextEncoderUnload.define_schema()
    assert s.node_id == "MiniMaxH3_TextEncoderUnload"
    assert s.category == "MiniMax H3/Sampling"
    ids = {i.id for i in s.inputs}
    assert ids == {"positive", "negative", "clip"}
