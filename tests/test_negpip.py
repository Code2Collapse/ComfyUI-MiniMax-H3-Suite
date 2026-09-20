"""NegPiP / Value Sign Flip for H3.

The method is eq. 6 of arXiv:2508.10931:

    Z = softmax(Q (K+ (+) K-)^T / sqrt(d)) (V+ (+) -a V-)

so the tests that matter are (a) the arithmetic is that and not something
adjacent, and (b) the flip happens in the DiT and NOWHERE else. Gate 2 in
particular protects H3's token refiner, which is a text-only self-attention
stack that BUILDS the prompt representation - flipping there would corrupt the
prompt before the DiT ever applies it, and the symptom would be "the prompt
stopped working", not "negative guidance is too strong".

CPU-only, weight-free.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

PACK = Path(__file__).resolve().parents[1]
if str(PACK) not in sys.path:
    sys.path.insert(0, str(PACK))

from mmx_utils.negpip import (  # noqa: E402
    MAX_WEIGHT,
    NegPiPPlan,
    Span,
    Term,
    apply_value_signs,
    describe_plan,
    extend_token_tags,
    in_sigma_window,
    parse_terms,
    sign_vector,
    spans_from_lengths,
)


# ── reading the term box ────────────────────────────────────────────────────

def test_a_bare_line_is_a_term_at_weight_one():
    assert parse_terms("blurry") == [Term("blurry", 1.0)]


def test_a_trailing_colon_number_is_the_weight():
    assert parse_terms("blurry : 1.5") == [Term("blurry", 1.5)]
    assert parse_terms("blurry:2") == [Term("blurry", 2.0)]


def test_a_colon_inside_the_phrase_is_not_a_weight():
    # "close-up: hands" is a phrase, not a term with a broken weight.
    assert parse_terms("close-up: hands") == [Term("close-up: hands", 1.0)]


def test_blanks_and_comments_are_skipped():
    assert parse_terms("\n# a note\n\nblurry\n  \n") == [Term("blurry", 1.0)]


def test_a_negative_weight_is_refused_not_silently_flipped():
    # Two minus signs cancelling would turn a negative term into positive
    # guidance, which reads as "NegPiP made it worse".
    with pytest.raises(ValueError, match="push away|must be 0 or more"):
        parse_terms("blurry : -1.0")


def test_an_absurd_weight_is_refused_with_the_reason():
    with pytest.raises(ValueError, match="ceiling"):
        parse_terms(f"blurry : {MAX_WEIGHT + 1}")


def test_too_many_terms_is_refused_because_each_one_costs_sequence():
    with pytest.raises(ValueError, match="lengthens the packed sequence"):
        parse_terms("\n".join(f"term{i}" for i in range(40)))


# ── the sign vector ─────────────────────────────────────────────────────────

def test_rows_outside_a_span_are_left_at_exactly_one():
    # The positive prompt must not be re-weighted just because a negative
    # term exists - that is the whole point of concatenating rather than
    # blending two conditionings.
    signs = sign_vector(10, [Span(6, 8, 1.0, "blurry")], 1.0)
    assert signs[:6].unique().tolist() == [1.0]
    assert signs[8:].unique().tolist() == [1.0]


def test_the_negative_rows_are_negative_alpha_times_weight():
    signs = sign_vector(10, [Span(6, 8, 1.5, "blurry")], 2.0)
    assert signs[6:8].unique().tolist() == [-3.0]


def test_strength_zero_neutralises_the_terms_without_inverting_them():
    signs = sign_vector(10, [Span(6, 8, 1.0, "x")], 0.0)
    assert signs[6:8].unique().tolist() == [0.0]


def test_a_span_past_the_text_length_names_the_cause():
    with pytest.raises(ValueError, match="re-run the NegPiP node"):
        sign_vector(6, [Span(6, 8, 1.0, "blurry")], 1.0)


def test_spans_are_laid_out_in_order_after_the_prompt():
    spans = spans_from_lengths(100, [Term("a", 1.0), Term("b", 2.0)], [3, 5])
    assert (spans[0].start, spans[0].stop) == (100, 103)
    assert (spans[1].start, spans[1].stop) == (103, 108)
    assert spans[1].weight == 2.0


def test_a_term_that_encoded_to_nothing_is_named():
    with pytest.raises(ValueError, match="at least one"):
        spans_from_lengths(100, [Term("", 1.0)], [0])


# ── applying it to V ────────────────────────────────────────────────────────

def test_only_the_named_rows_change():
    v = torch.randn(1, 4, 20, 8)
    signs = sign_vector(10, [Span(6, 8, 1.0, "x")], 1.0)
    out = apply_value_signs(v, signs, text_start=0)
    assert torch.equal(out[:, :, :6], v[:, :, :6])
    assert torch.equal(out[:, :, 6:8], -v[:, :, 6:8])
    assert torch.equal(out[:, :, 8:], v[:, :, 8:])


def test_the_input_is_not_mutated():
    # Sampling runs inside torch.inference_mode(); an in-place write here
    # poisons the buffer for every later consumer and the next node dies with
    # "Inplace update to inference tensor outside InferenceMode".
    v = torch.randn(1, 2, 12, 4)
    before = v.clone()
    apply_value_signs(v, sign_vector(6, [Span(4, 6, 1.0, "x")], 1.0))
    assert torch.equal(v, before)


def test_an_all_positive_vector_is_a_no_op_returned_cheaply():
    v = torch.randn(1, 2, 12, 4)
    assert apply_value_signs(v, torch.ones(6)) is v


def test_a_text_span_that_does_not_fit_is_named():
    with pytest.raises(ValueError, match="layout changed"):
        apply_value_signs(torch.randn(1, 2, 5, 4), torch.ones(6))


def test_a_wrong_rank_is_refused():
    with pytest.raises(ValueError, match=r"B, heads, S, head_dim"):
        apply_value_signs(torch.randn(2, 5, 4), torch.ones(5))


# ── the method actually does what the paper says ────────────────────────────

def _attend(q, k, v):
    return torch.softmax(q @ k.transpose(-1, -2) / (q.shape[-1] ** 0.5), dim=-1) @ v


def test_flipping_V_moves_the_output_away_from_the_negative_concept():
    """The end-to-end claim, on a real softmax.

    A query aimed at the negative token gets that token's own direction
    subtracted, so its projection onto that direction must DROP - and drop
    further as alpha rises. This is the property that makes VSF adaptive:
    nothing is subtracted from a query that never attended to the term.
    """
    torch.manual_seed(0)
    d = 16
    concept = torch.nn.functional.normalize(torch.randn(d), dim=0)

    k = torch.randn(1, 1, 8, d)
    v = torch.randn(1, 1, 8, d)
    k[0, 0, 7] = concept * 4.0            # row 7 is the negative term
    v[0, 0, 7] = concept * 4.0
    q = (concept * 4.0).view(1, 1, 1, d)  # a query that matches it

    plain = _attend(q, k, v)[0, 0, 0] @ concept
    projections = [plain.item()]
    for alpha in (0.5, 1.0, 2.0):
        signs = sign_vector(8, [Span(7, 8, 1.0, "concept")], alpha)
        flipped = _attend(q, k, apply_value_signs(v, signs))[0, 0, 0] @ concept
        projections.append(flipped.item())

    assert projections == sorted(projections, reverse=True), (
        f"raising alpha must keep pushing away from the concept: {projections}"
    )
    assert projections[2] < 0 < projections[0], (
        "at alpha=1 a query aimed straight at the negative term should end up "
        f"on the other side of it: {projections}"
    )


def test_a_query_that_ignores_the_term_is_barely_touched():
    torch.manual_seed(1)
    d = 16
    concept = torch.nn.functional.normalize(torch.randn(d), dim=0)
    other = torch.nn.functional.normalize(torch.randn(d), dim=0)

    k = torch.randn(1, 1, 8, d) * 0.1
    v = torch.randn(1, 1, 8, d) * 0.1
    k[0, 0, 7] = concept * 6.0
    v[0, 0, 7] = concept * 6.0
    k[0, 0, 3] = other * 6.0
    q = (other * 6.0).view(1, 1, 1, d)     # aimed at row 3, not row 7

    plain = _attend(q, k, v)[0, 0, 0]
    signs = sign_vector(8, [Span(7, 8, 1.0, "concept")], 1.0)
    flipped = _attend(q, k, apply_value_signs(v, signs))[0, 0, 0]
    assert (plain - flipped).norm().item() < 0.05 * plain.norm().item()


def test_the_keys_are_never_touched():
    # VSF is explicit that K- keeps its meaning; flipping it too would make the
    # term repel attention instead of receiving it, and the subtraction would
    # never happen.
    v = torch.randn(1, 2, 8, 4)
    k = torch.randn(1, 2, 8, 4)
    before = k.clone()
    apply_value_signs(v, sign_vector(8, [Span(6, 8, 1.0, "x")], 1.0))
    assert torch.equal(k, before)


# ── H3's token tags ─────────────────────────────────────────────────────────

def test_appended_rows_get_the_text_tag():
    # comfy/text_encoders/minimax.py:78 - text is 1, vision pads are 0.
    tags = torch.tensor([1, 0, 0, 1])
    assert extend_token_tags(tags, 3).tolist() == [1, 0, 0, 1, 1, 1, 1]


def test_extending_by_nothing_returns_the_same_tags():
    tags = torch.tensor([1, 1])
    assert extend_token_tags(tags, 0) is tags


def test_short_tags_would_have_crashed_the_first_dit_block():
    """Mirrors comfy/ldm/minimax/model.py:680-687 exactly.

    That loop does `tags = text_token_tags.view(-1).tolist()` and then walks
    `range(1, b - a + 1)` indexing `tags[i]`, so a tags vector shorter than the
    text span is an IndexError inside block 0 - not a subtle quality problem.
    """
    def run(tags, span_len):
        t = tags.view(-1).tolist()
        runs, run_start = [], 0
        for i in range(1, span_len + 1):
            if i == span_len or t[i] != t[run_start]:
                runs.append((run_start, i, int(t[run_start])))
                run_start = i
        return runs

    original = torch.ones(10, dtype=torch.long)
    with pytest.raises(IndexError):
        run(original, 14)                       # the bug this guards
    assert run(extend_token_tags(original, 4), 14) == [(0, 14, 1)]


# ── the sigma window ────────────────────────────────────────────────────────

def test_the_window_is_inclusive_and_order_agnostic():
    assert in_sigma_window(0.5, 1.0, 0.0)
    assert in_sigma_window(1.0, 1.0, 0.0)
    assert in_sigma_window(0.0, 1.0, 0.0)
    assert not in_sigma_window(1.5, 1.0, 0.0)
    assert in_sigma_window(0.5, 0.0, 1.0)       # swapped bounds still work


# ── the report ──────────────────────────────────────────────────────────────

def test_the_report_states_the_rows_and_the_multiplier():
    plan = NegPiPPlan(positive_len=100, spans=[Span(100, 103, 1.5, "blurry")],
                      strength=2.0)
    text = describe_plan(plan)
    assert "100-102" in text
    assert "-3.00" in text
    assert plan.text_len == 103


def test_an_empty_plan_says_nothing_happens():
    assert "Nothing is flipped" in describe_plan(NegPiPPlan(positive_len=10))


# ── the gates ───────────────────────────────────────────────────────────────

class _Layout:
    def __init__(self, text_len, seq_len):
        self.seq_len = seq_len
        self.segments = [(0, text_len, "text"),
                         (text_len, seq_len, "video")]


def _attn(plan=None):
    from mmx_nodes.negpip import _NegPiPAttention
    plan = plan or NegPiPPlan(positive_len=10,
                              spans=[Span(10, 12, 1.0, "blurry")])
    return _NegPiPAttention(plan)


def test_gate1_nothing_happens_outside_the_h3_forward():
    a = _attn()
    assert a._should_flip(torch.randn(1, 2, 60, 4), {}) is None


def test_gate2_the_token_refiner_is_left_alone():
    """The refiner's sequence is the text only, so it is shorter than the
    packed sequence. If this gate ever fails open, the prompt itself is
    corrupted before the DiT applies it."""
    a = _attn()
    opts = {"minimax_h3_layout": _Layout(12, 60)}
    refiner_v = torch.randn(1, 2, 12, 4)       # text-only sequence
    assert a._should_flip(refiner_v, opts) is None
    dit_v = torch.randn(1, 2, 60, 4)           # full packed sequence
    assert a._should_flip(dit_v, opts) == 0


def test_gate3_another_conditioning_is_left_alone():
    a = _attn()
    # CFG's negative prompt encodes to a different length -> different seq_len
    opts = {"minimax_h3_layout": _Layout(9, 57)}
    assert a._should_flip(torch.randn(1, 2, 57, 4), opts) is None


def test_the_cfg_negative_pass_is_skipped_even_at_the_same_length():
    a = _attn()
    opts = {"minimax_h3_layout": _Layout(12, 60), "cond_or_uncond": [1]}
    assert a._should_flip(torch.randn(1, 2, 60, 4), opts) is None
    opts["cond_or_uncond"] = [0]
    assert a._should_flip(torch.randn(1, 2, 60, 4), opts) == 0


def test_the_sigma_window_gates_the_flip():
    plan = NegPiPPlan(positive_len=10, spans=[Span(10, 12, 1.0, "x")],
                      sigma_start=1.0, sigma_end=0.5)
    a = _attn(plan)
    opts = {"minimax_h3_layout": _Layout(12, 60)}
    v = torch.randn(1, 2, 60, 4)
    opts["sigmas"] = torch.tensor([0.75])
    assert a._should_flip(v, opts) == 0
    opts["sigmas"] = torch.tensor([0.25])
    assert a._should_flip(v, opts) is None


def test_the_text_start_offset_is_honoured_when_text_is_not_first():
    a = _attn()
    layout = _Layout(12, 60)
    layout.segments = [(0, 5, "ref_img"), (5, 17, "text"), (17, 60, "video")]
    assert a._should_flip(torch.randn(1, 2, 60, 4), {"minimax_h3_layout": layout}) == 5


def test_the_override_passes_through_untouched_when_no_gate_opens():
    a = _attn()
    seen = {}

    def backend(q, k, v, heads, **kw):
        seen["v"] = v
        return v

    v = torch.randn(1, 2, 60, 4)
    a(backend, torch.randn(1, 2, 60, 4), torch.randn(1, 2, 60, 4), v, 2,
      transformer_options={})
    assert seen["v"] is v


def test_the_override_chains_an_existing_one():
    from mmx_nodes.negpip import _NegPiPAttention

    calls = []

    def inner(func, q, k, v, heads, **kw):
        calls.append(v)
        return func(q, k, v, heads, **kw)

    plan = NegPiPPlan(positive_len=10, spans=[Span(10, 12, 1.0, "x")])
    a = _NegPiPAttention(plan, inner)
    v = torch.randn(1, 2, 60, 4)
    a(lambda q, k, v, h, **kw: v, torch.randn(1, 2, 60, 4),
      torch.randn(1, 2, 60, 4), v, 2,
      transformer_options={"minimax_h3_layout": _Layout(12, 60)})
    assert len(calls) == 1
    assert torch.equal(calls[0][:, :, 10:12], -v[:, :, 10:12])
    assert torch.equal(calls[0][:, :, :10], v[:, :, :10])


# ── the node ────────────────────────────────────────────────────────────────

def test_the_node_returns_model_conditioning_and_report():
    from mmx_nodes.negpip import MiniMaxH3_NegPiP

    schema = MiniMaxH3_NegPiP.define_schema()
    assert [o.display_name for o in schema.outputs] == ["model", "conditioning", "report"]


def test_an_empty_term_box_is_a_pass_through_not_an_error():
    from mmx_nodes.negpip import MiniMaxH3_NegPiP

    out = MiniMaxH3_NegPiP.execute(model="M", clip=None, conditioning=[["c", {}]],
                                   terms="\n# nothing\n", strength=1.0)
    assert out[0] == "M"
    assert out[1] == [["c", {}]]
    assert "Nothing is flipped" in out[2]


def test_prompt_prefix_mode_without_the_prompt_is_refused_by_name():
    from mmx_nodes.negpip import MiniMaxH3_NegPiP

    with pytest.raises(ValueError, match="prompt_prefix is empty"):
        MiniMaxH3_NegPiP.execute(
            model=None, clip=None, conditioning=[[torch.zeros(1, 4, 8), {}]],
            terms="blurry", strength=1.0, context_mode="prompt_prefix")


def test_mixed_prompt_lengths_are_refused_because_the_gate_needs_one():
    from mmx_nodes.negpip import MiniMaxH3_NegPiP

    conditioning = [[torch.zeros(1, 4, 8), {}], [torch.zeros(1, 6, 8), {}]]
    with pytest.raises(ValueError, match="different prompt"):
        MiniMaxH3_NegPiP.execute(model=None, clip=None, conditioning=conditioning,
                                 terms="blurry", strength=1.0)
