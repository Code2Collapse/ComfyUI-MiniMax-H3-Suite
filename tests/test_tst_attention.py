"""H3 TST Attention (ported from supElement/ComfyUI_MinimaxH3_AutoContext, Apache-2.0; arXiv 2609.08505): the tension
at its two extremes, the query scaling of the video rows only, the pass-through cases, chaining, the node."""
from __future__ import annotations

import math

import pytest
import torch

from mmx_utils import tst_attention as TA

pytest.importorskip("comfy.patcher_extension")


def frame_qk(frames, rows, heads=2, dim=16, kind="identity", scale=40.0):
    """q, k [1, heads, frames*rows, dim] whose frame means give near-identity or uniform frame attention."""
    q = torch.zeros(1, heads, frames * rows, dim)
    k = torch.zeros_like(q)
    if kind == "identity":
        for f in range(frames):
            q[0, :, f * rows:(f + 1) * rows, f] = scale
            k[0, :, f * rows:(f + 1) * rows, f] = scale
    return q, k


def test_tension_extremes():
    q, k = frame_qk(6, 4, kind="identity")
    t = TA.spectral_tension(q, k, 0, 6, 4)
    assert torch.allclose(t, torch.full_like(t, -1.0), atol=1e-3)          # fragmented: each frame only itself
    q, k = frame_qk(6, 4, kind="uniform")
    t = TA.spectral_tension(q, k, 0, 6, 4)
    assert torch.allclose(t, torch.full_like(t, 1.0), atol=1e-3)           # over-mixed: all frames alike


def fake_func(seen):
    def func(q, k, v, heads, mask=None, **kw):
        seen.append(q.clone())
        return q
    return func


def video_call(state, text_rows=5, frames=6, rows=4, kind="uniform"):
    q, k = frame_qk(frames, rows, kind=kind)
    q = torch.cat([torch.ones(1, 2, text_rows, 16), q], dim=2)
    k = torch.cat([torch.ones(1, 2, text_rows, 16), k], dim=2)
    state["frames"], state["rows_per_frame"] = frames, rows
    return q, k


def test_only_video_queries_are_scaled():
    st = TA.new_state(0.3)
    st["layers"], st["total_steps"], st["step"] = 1, 1, 0
    seen = []
    ov = TA.attention_override(st)
    q, k = video_call(st)
    q[0, :, 5:] = 1.0                                     # uniform frame attention, tension +1
    orig = q.clone()
    ov(fake_func(seen), q, k, q, 2)
    out = seen[-1]
    assert torch.equal(out[0, :, :5], orig[0, :, :5])     # text / reference rows untouched
    assert torch.allclose(out[0, :, 5:], orig[0, :, 5:] * math.exp(0.3), atol=1e-4)
    assert st["stats"]["calls"] == 1 and st["stats"]["active"] == 1
    assert "1 video attention calls corrected" in TA.summary(st)


def test_pass_through_cases(monkeypatch):
    st = TA.new_state(0.3)
    seen = []
    ov = TA.attention_override(st)
    short = torch.ones(1, 2, 10, 16)                      # token refiner: shorter than the video rows
    st["frames"], st["rows_per_frame"] = 6, 4
    ov(fake_func(seen), short.clone(), short, short, 2)
    assert torch.equal(seen[-1], short) and st["calls_this_forward"] == 0
    q, k = video_call(st)
    monkeypatch.setenv(TA.KILL_SWITCH, "off")
    ov(fake_func(seen), q.clone(), k, q, 2)
    assert torch.equal(seen[-1], q) and st["stats"]["calls"] == 0
    monkeypatch.delenv(TA.KILL_SWITCH)
    st0 = TA.new_state(0.0)
    ov0 = TA.attention_override(st0)
    q, k = video_call(st0)
    ov0(fake_func(seen), q.clone(), k, q, 2)
    assert torch.equal(seen[-1], q)                       # tau 0: measured, never changed


def test_layer_and_step_weights():
    st = TA.new_state(0.5)
    st["layers"], st["total_steps"] = 3, 5
    ov = TA.attention_override(st)
    gains = []
    for step in (0, 4):
        st["step"], st["calls_this_forward"] = step, 0
        for _layer in range(3):
            seen = []
            q, k = video_call(st)
            q[0, :, 5:] = 1.0
            ov(fake_func(seen), q, k, q, 2)
            gains.append(float(seen[-1][0, 0, 5, 0]))
    first_step, last_step = gains[:3], gains[3:]
    assert first_step[0] == pytest.approx(1.0) and first_step[2] == pytest.approx(math.exp(0.5), rel=1e-4)
    assert all(g == pytest.approx(1.0) for g in last_step)   # step weight is 0 at the last step


def test_forward_wrapper_reads_shape_and_step():
    st = TA.new_state(0.2)
    st["calls_this_forward"] = 7
    x = [torch.zeros(1, 24, 3, 9, 12), torch.zeros(1, 32, 2, 6)]
    seen = {}

    def executor(x, t, c, to, *a, **kw):
        seen["ok"] = True
        return x
    to = {"sample_sigmas": torch.tensor([1.0, 0.6, 0.3, 0.0])}
    TA.forward_wrapper(st, executor, x, torch.tensor([600.0]), None, to)
    assert seen["ok"] and st["layers"] == 7 and st["frames"] == 3 and st["rows_per_frame"] == 5 * 6
    assert st["total_steps"] == 3 and st["step"] == 1


class FakePatcher:
    def __init__(self, options=None):
        self.model_options = {k: (dict(v) if isinstance(v, dict) else v) for k, v in (options or {}).items()}
        self.wrappers = []

    def clone(self):
        c = FakePatcher(self.model_options)
        c.wrappers = list(self.wrappers)
        return c

    def add_wrapper_with_key(self, kind, key, fn):
        self.wrappers.append((kind, key, fn))


def test_patch_chains_an_earlier_override_and_logs_a_summary():
    calls = []

    def earlier(func, q, k, v, heads, mask=None, **kw):
        calls.append("earlier")
        return func(q, k, v, heads, mask=mask, **kw)
    base = FakePatcher({"transformer_options": {"optimized_attention_override": earlier}})
    logs = []
    patched, st = TA.patch_model(base, 0.2, log=logs.append)
    assert base.model_options["transformer_options"]["optimized_attention_override"] is earlier
    ov = patched.model_options["transformer_options"]["optimized_attention_override"]
    q, k = video_call(st)
    ov(fake_func([]), q, k, q, 2)
    assert calls == ["earlier"]
    kinds = {kind for kind, _key, _fn in patched.wrappers}
    assert kinds == {"diffusion_model", "outer_sample"}
    outer = next(fn for kind, _key, fn in patched.wrappers if kind == "outer_sample")
    outer(lambda: "sampled")
    assert logs and logs[-1].startswith("TST")


def test_node_tau_zero_passes_through():
    from mmx_nodes.tst_attention import MiniMaxH3_TSTAttention
    base = FakePatcher()
    out = MiniMaxH3_TSTAttention.execute(base, 0.0)
    res = out.result if hasattr(out, "result") else out.args
    assert res[0] is base and "unchanged" in res[1]
    res = MiniMaxH3_TSTAttention.execute(base, 0.25)
    res = res.result if hasattr(res, "result") else res.args
    assert res[0] is not base and "tau 0.25" in res[1]


def test_against_core_h3_attention_module():
    """Core's own H3 Attention (tiny, random weights, no RoPE path): the override receives q as [1, heads, S, dim]
    with the video rows last, and only the output changes, not the shapes."""
    mm = pytest.importorskip("comfy.ldm.minimax.model")
    import comfy.ops
    torch.manual_seed(0)
    att = mm.Attention(32, 2, 16, 1e-6, operations=comfy.ops.disable_weight_init)
    for p in att.parameters():
        torch.nn.init.normal_(p, std=0.2)
    att.q_norm.weight.data.fill_(1.0)
    att.k_norm.weight.data.fill_(1.0)
    frames, rows, text = 4, 6, 7
    x = torch.randn(text + frames * rows, 32)
    shapes = []

    from comfy.ldm.modules.attention import attention_pytorch

    def spy(func, q, k, v, heads, mask=None, **kw):     # core's torch SDPA: xformers has no CPU kernel here
        shapes.append(tuple(q.shape))
        return attention_pytorch(q, k, v, heads, mask=mask, **kw)

    def run(tau):
        st = TA.new_state(tau)
        st["frames"], st["rows_per_frame"], st["layers"] = frames, rows, 1
        to = {"optimized_attention_override": TA.attention_override(st, spy)}
        with torch.no_grad():
            return att(x, rope_freqs=None, transformer_options=to), st
    base, st0 = run(0.0)
    moved, st1 = run(0.5)
    assert shapes[0] == (1, 2, text + frames * rows, 16)
    assert base.shape == moved.shape == x.shape
    assert st1["stats"]["calls"] == 1 and not torch.allclose(base, moved)
    assert torch.allclose(base[:0], moved[:0])
