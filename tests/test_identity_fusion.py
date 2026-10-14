"""H3 Identity Fusion Controller (C2C L8.26, owner A12 s1.2): mmx_utils/identity_fusion.py + the node's attention
override. CPU-only; the reference for every bias is a plain SDPA with a dense additive mask."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

PACK = Path(__file__).resolve().parents[1]
if str(PACK) not in sys.path:
    sys.path.insert(0, str(PACK))

from mmx_utils.identity_fusion import (  # noqa: E402
    LOG_ZERO,
    biased_attention,
    column_flags,
    ref_groups,
    row_logweights,
    token_mask,
)


def _qkv(S=40, H=2, d=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(1, H, S, d, generator=g) for _ in range(3))


def _dense(q, k, v, bias):
    return F.scaled_dot_product_attention(q, k, v, attn_mask=bias)


def test_padded_bias_equals_a_dense_mask():
    # INVARIANT: the padded-channel bias IS the additive mask log w_i on identity keys, log p_i on performance keys,
    # -30 on target keys for reference rows - exact up to float error.
    q, k, v = _qkv()
    S = q.shape[2]
    w = torch.rand(S) * 2
    w[:5] = 0.0
    logw = torch.where(w > 0, w.clamp_min(1e-13).log(), torch.tensor(LOG_ZERO))
    logp = torch.rand(S) - 0.5
    idc, pfc = column_flags(S, [(4, 10)]), column_flags(S, [(10, 16)])
    asym, tgt = column_flags(S, [(4, 16)]), column_flags(S, [(20, 40)])
    dense = logw[:, None] * idc[None, :] + logp[:, None] * pfc[None, :] + (asym[:, None] * LOG_ZERO) * tgt[None, :]
    got = biased_attention(q, k, v, logw=logw, id_cols=idc, logp=logp, perf_cols=pfc, asym_rows=asym, tgt_cols=tgt)
    assert (got - _dense(q, k, v, dense)).abs().max().item() < 1e-5


def test_weight_one_everywhere_is_plain_attention():
    # INVARIANT: all-ones weights change nothing.
    q, k, v = _qkv()
    S = q.shape[2]
    got = biased_attention(q, k, v, logw=torch.zeros(S), id_cols=column_flags(S, [(4, 10)]))
    assert (got - F.scaled_dot_product_attention(q, k, v)).abs().max().item() < 1e-5


def test_token_rows_follow_packed_layout_order():
    # INVARIANT: latent pixel (t=1, y=5, x=2) on a 3x8x6 latent grid -> token row 1*(4*3) + 2*3 + 1 = 19.
    m = torch.zeros(3, 8, 6)
    m[1, 5, 2] = 1.0
    rows = token_mask(m, 3, 8, 6)
    assert rows.shape == (3 * 4 * 3,) and rows.nonzero().flatten().tolist() == [19]


def test_token_mask_resizes_frames_and_grid():
    # INVARIANT: a mask on another grid / frame count is resized (nearest), never dropped.
    m = torch.ones(5, 16, 12)
    assert token_mask(m, 2, 8, 6).all() and token_mask(m, 2, 8, 6).numel() == 2 * 4 * 3


def test_reference_groups_follow_core_order():
    # INVARIANT: images first then videos, mapped onto the layout's ref_img segments in order; a mismatch is named.
    seg = [(0, 10, "text"), (10, 14, "ref_img"), (14, 18, "ref_img"), (18, 30, "ref_img"), (30, 34, "audio"),
           (34, 70, "video")]
    g = ref_groups(["image", "image", "video"], seg)
    assert g.identity == [(10, 14), (14, 18)] and g.performance == [(18, 30)] and g.target == (34, 70)
    assert not g.problem
    assert "visual reference" in ref_groups(["image"], seg).problem
    assert "no identity picture" in ref_groups(["video", "video", "video"], seg).problem


def test_row_weights_region_expression_occluder():
    # INVARIANT: outside R -> w_out; inside R -> w_in; E inside R -> w_expr (and p_expr on the plate); O wins.
    S, tgt = 20, (10, 20)
    R = torch.tensor([1, 1, 1, 1, 1, 0, 0, 0, 0, 0], dtype=torch.bool)
    E = torch.tensor([0, 1, 0, 0, 0, 0, 1, 0, 0, 0], dtype=torch.bool)
    O = torch.tensor([0, 0, 1, 0, 0, 0, 0, 0, 0, 0], dtype=torch.bool)
    lw, lp = row_logweights(S, tgt, R, E, O, w_inside=2.0, w_outside=0.0, w_expression=0.5, p_expression=1.5,
                            w_occluder=0.0)
    assert lw[:10].abs().sum() == 0 and lp[:10].abs().sum() == 0             # text / reference rows untouched
    assert lw[10] == pytest.approx(math.log(2.0)) and lw[11] == pytest.approx(math.log(0.5))
    assert lw[12] == LOG_ZERO and lw[15] == LOG_ZERO and lw[16] == LOG_ZERO  # occluder; outside (E outside R ignored)
    assert lp[11] == pytest.approx(math.log(1.5)) and lp[16] == 0.0


# ── the override ─────────────────────────────────────────────────────────

class _Layout:
    def __init__(self):
        # text 4 | picture 4 | plate video 4 | audio 2 | target video: 2 frames x (4/2 x 4/2) = 8 rows
        self.segments = [(0, 4, "text"), (4, 8, "ref_img"), (8, 12, "ref_img"), (12, 14, "audio"), (14, 22, "video")]
        self.seq_len = 22
        self.signature = (4, 2, 4, 4, 1)


def _override(region=None, **cfg_over):
    from mmx_nodes.identity_fusion import _FusionAttention, _Plan

    cfg = {"w_inside": 1.0, "w_outside": 0.0, "w_expression": 0.6, "p_expression": 1.4, "w_occluder": 0.0,
           "block_start": 0, "block_end": -1, "start_percent": 0.0, "end_percent": 1.0, "asymmetric": False,
           "performance_index": 0}
    cfg.update(cfg_over)
    if region is None:
        region = torch.zeros(2, 4, 4)
        region[:, :2, :2] = 1.0                                # token (0, 0) of each frame
    return _FusionAttention(_Plan(["image", "video"], region, None, None, cfg, (100.0, 0.0, 50.0)))


def _plain(q, k, v, heads, skip_reshape=True, skip_output_reshape=False, **kw):
    out = F.scaled_dot_product_attention(q, k, v)
    if skip_output_reshape:
        return out
    b, h, s, d = out.shape
    return out.transpose(1, 2).reshape(b, s, h * d)


def test_gates_leave_attention_bit_identical():
    # INVARIANT: not H3 / the token refiner / the CFG negative / outside sigmas / outside blocks -> plain result.
    ov = _override(block_start=3, block_end=5)
    q, k, v = _qkv(S=22)
    ref = _plain(q, k, v, 2)
    lay = _Layout()
    for opts in ({}, {"minimax_h3_layout": lay, "cond_or_uncond": [1]},
                 {"minimax_h3_layout": lay, "sigmas": torch.tensor([500.0])},
                 {"minimax_h3_layout": lay, "block_index": 1}):
        assert torch.equal(ov(_plain, q, k, v, 2, skip_reshape=True, transformer_options=opts), ref)
    q2, k2, v2 = _qkv(S=10)
    assert torch.equal(ov(_plain, q2, k2, v2, 2, skip_reshape=True, transformer_options={"minimax_h3_layout": lay}),
                       _plain(q2, k2, v2, 2))
    assert ov.stats["biased"] == 0


def test_active_override_matches_the_dense_reference():
    # INVARIANT: inside the window the result equals SDPA with the dense bias: identity only on region rows
    # (rows 14 and 18 = token (0,0) of each target frame), zero identity attention on every other target row.
    ov = _override()
    q, k, v = _qkv(S=22)
    got = ov(_plain, q, k, v, 2, skip_reshape=True,
             transformer_options={"minimax_h3_layout": _Layout(), "block_index": 2, "sigmas": torch.tensor([10.0]),
                                  "cond_or_uncond": [0]})
    bias = torch.zeros(22, 22)
    for row in range(14, 22):
        if row not in (14, 18):
            bias[row, 4:8] = LOG_ZERO
    want = _dense(q, k, v, bias)
    b, h, s, d = want.shape
    assert ov.stats["biased"] == 1
    assert (got - want.transpose(1, 2).reshape(b, s, h * d)).abs().max().item() < 1e-5


def test_an_inner_override_still_runs_inside():
    # INVARIANT: chained like NegPiP - the inner override receives the padded tensors and its edit (here: flip V on
    # the text rows) is in the result.
    ov = _override()
    seen = {}

    def inner(func, q, k, v, heads, **kw):
        seen["shape"] = tuple(v.shape)
        v = v.clone()
        v[:, :, :4, :] *= -1
        return func(q, k, v, heads, **kw)

    ov.inner = inner
    q, k, v = _qkv(S=22)
    got = ov(_plain, q, k, v, 2, skip_reshape=True, skip_output_reshape=True,
             transformer_options={"minimax_h3_layout": _Layout()})
    assert seen["shape"][-1] == 16 + 8
    vf = v.clone()
    vf[:, :, :4, :] *= -1
    bias = torch.zeros(22, 22)
    for row in range(14, 22):
        if row not in (14, 18):
            bias[row, 4:8] = LOG_ZERO
    assert (got - _dense(q, k, vf, bias)).abs().max().item() < 1e-5


class _FakeSampling:
    def percent_to_sigma(self, p):
        return 100.0 * (1.0 - p)


class _FakeModel:
    def __init__(self, opts=None):
        self.model_options = {"transformer_options": dict(opts or {})}

    def clone(self):
        return _FakeModel(self.model_options["transformer_options"])

    def get_model_object(self, name):
        return _FakeSampling()


def _run_node(positive, model=None):
    from mmx_nodes.identity_fusion import MiniMaxH3_IdentityFusion

    out = MiniMaxH3_IdentityFusion.execute(model or _FakeModel(), positive, torch.ones(2, 4, 4))
    return out.args if hasattr(out, "args") else out


def test_node_needs_an_identity_picture_and_refuses_a_second_controller():
    # INVARIANT: no <Picture N> in the positive -> a plain error; a second controller on the same model is refused;
    # the patched model is a clone carrying the override.
    with pytest.raises(ValueError, match="reference image"):
        _run_node([[torch.zeros(1), {"minimax_refs": [{"kind": "video"}]}]])
    pos = [[torch.zeros(1), {"minimax_refs": [{"kind": "image"}, {"kind": "video"}]}]]
    m, report = _run_node(pos)
    assert "optimized_attention_override" in m.model_options["transformer_options"] and "experimental" in report
    with pytest.raises(ValueError, match="already has"):
        _run_node(pos, model=m)


def test_real_h3_attention_module_runs_through_the_override():
    # INVARIANT: through core's own minimax Attention (containers, wrapper kwargs, output reshape, out_proj) the
    # active override returns out_proj(dense-biased attention) - the hook works on the real call path, not a mock.
    import comfy.ops
    from comfy.ldm.minimax.model import Attention

    torch.manual_seed(0)
    att = Attention(hidden=64, heads=2, head_dim=32, eps=1e-6, operations=comfy.ops.disable_weight_init)
    with torch.no_grad():
        for prm in att.parameters():
            prm.copy_(torch.randn_like(prm) * 0.2)
    x = torch.randn(22, 64)
    ov = _override()
    opts = {"minimax_h3_layout": _Layout(), "optimized_attention_override": ov}
    got = att(x, rope_freqs=None, transformer_options=opts)
    assert ov.stats["biased"] == 1

    with torch.no_grad():
        q, k, v = att.qkv_proj(x).split(64, dim=-1)
        q = att.q_norm(q.view(22, 2, 32)).transpose(0, 1)[None]
        k = att.k_norm(k.view(22, 2, 32)).transpose(0, 1)[None]
        v = v.view(22, 2, 32).transpose(0, 1)[None]
        bias = torch.zeros(22, 22)
        for row in range(14, 22):
            if row not in (14, 18):
                bias[row, 4:8] = LOG_ZERO
        ref = _dense(q, k, v, bias).transpose(1, 2).reshape(22, 64)
        want = att.out_proj(ref)
    assert (got - want).abs().max().item() < 1e-4
