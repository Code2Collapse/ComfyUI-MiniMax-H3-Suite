"""NegPiP / Value-Sign-Flip negative guidance for MiniMax H3.

The method
----------
NegPiP (hako-mikan, sd-webui-negpip) and VSF (Guo et al., arXiv:2508.10931) are
the same trick found twice. VSF states it as

    Z = softmax( Q (K+ (+) K-)^T / sqrt(d) ) (V+ (+) -a V-)          (eq. 6)

where (+) is concatenation along the sequence, and `a` is a strength factor.
The negative prompt's KEYS are left alone - they keep their meaning, so the
attention score still answers "how much does this query look like `blurry`?" -
and only its VALUES are negated. A patch that matches the negative concept
therefore receives that concept's own direction, subtracted. Softmax is not
touched, so the weighting between positive and negative is decided by the
attention magnitudes and varies per layer, per step and per token.

Why H3 is an unusually good fit
-------------------------------
In a cross-attention UNet the concatenation in eq. 6 has to be built by hand.
H3 does not have cross-attention at all: `comfy/ldm/minimax/model.py:158`
`Attention` is joint self-attention over ONE packed sequence, with a fused
`qkv_proj` and no `context` argument. Text, audio and video rows are laid out
contiguously by `PackedLayout` (`model.py:350`), text first. So the (+) of
eq. 6 is already there: appending the negative phrase's embeddings to the text
conditioning puts K- and V- in the sequence, and all that remains is to flip
the sign of V on those rows.

What this module does and does not do
-------------------------------------
This module is the arithmetic only - no torch.nn, no ComfyUI imports - so it
can be tested on CPU with no weights. It knows:

  * how to read the term list a user types,
  * how to turn that into a per-row sign/scale vector over the text span,
  * how to apply that vector to a [B, heads, S, head_dim] value tensor,
  * how to say, in English, what it is about to do.

The wiring (where the flip is allowed to happen, and where it must not) lives
in mmx_nodes/negpip.py.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import torch

# transformer_options key. Read by the attention override and the block gate.
NEGPIP_KEY = "minimax_h3_negpip"
# transformer_options flag, set only while a patched DiT block is running.
NEGPIP_ACTIVE = "minimax_h3_negpip_active"

# "phrase", "phrase : 1.4", "phrase:1.4". A trailing ":number" is the weight;
# anything else - including colons inside the phrase - is part of the phrase.
_WEIGHT_TAIL = re.compile(r"^(?P<phrase>.*?)\s*:\s*(?P<weight>-?\d+(?:\.\d+)?)\s*$")

MAX_TERMS = 16
MAX_WEIGHT = 10.0


@dataclass(frozen=True)
class Term:
    """One negative phrase and how hard to push against it."""

    phrase: str
    weight: float


@dataclass(frozen=True)
class Span:
    """Where a term's tokens landed in the concatenated text sequence."""

    start: int
    stop: int
    weight: float
    phrase: str = ""

    def __post_init__(self) -> None:
        if self.stop <= self.start:
            raise ValueError(
                f"negative term {self.phrase!r} encoded to {self.stop - self.start} "
                "tokens; it needs at least one"
            )


@dataclass
class NegPiPPlan:
    """Everything the attention override needs, fixed once at patch time.

    `positive_len` is the guard. H3 runs the positive and the CFG-negative
    conditionings through separate forward passes with different text lengths,
    so a length that does not match this one is a different conditioning and
    must be left alone.
    """

    positive_len: int
    spans: list[Span] = field(default_factory=list)
    strength: float = 1.0
    sigma_start: float = 1.0
    sigma_end: float = 0.0
    positive_only: bool = True

    @property
    def text_len(self) -> int:
        return self.positive_len + sum(s.stop - s.start for s in self.spans)


def parse_terms(text: str, *, max_terms: int = MAX_TERMS) -> list[Term]:
    """Read the multiline term box.

    One phrase per line. `blurry : 1.5` pushes harder than `blurry`. Blank
    lines and `#` comments are skipped. A negative weight is refused by name
    rather than silently turned into positive guidance - two minus signs
    cancelling is exactly the kind of thing that costs an hour to notice.
    """
    terms: list[Term] = []
    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        m = _WEIGHT_TAIL.match(line)
        if m and m.group("phrase").strip():
            phrase = m.group("phrase").strip()
            weight = float(m.group("weight"))
        else:
            phrase, weight = line, 1.0
        if weight < 0.0:
            raise ValueError(
                f"line {lineno}: {phrase!r} has weight {weight}. These terms are "
                "already negative - the weight says how strongly to push away "
                "from the phrase, so it must be 0 or more. A negative weight "
                "here would push you towards it."
            )
        if weight > MAX_WEIGHT:
            raise ValueError(
                f"line {lineno}: weight {weight} on {phrase!r} is past the "
                f"{MAX_WEIGHT} ceiling. Past roughly 3 the flipped values "
                "dominate the attention output and the frame falls apart."
            )
        terms.append(Term(phrase, weight))
        if len(terms) > max_terms:
            raise ValueError(
                f"more than {max_terms} negative terms. Each one lengthens the "
                "packed sequence for every DiT block, so this is a real cost, "
                "not a formality - merge them into fewer phrases."
            )
    return terms


def spans_from_lengths(positive_len: int, terms: list[Term],
                       lengths: list[int]) -> list[Span]:
    """Lay the encoded terms out after the positive prompt, in order."""
    if len(terms) != len(lengths):
        raise ValueError(
            f"{len(terms)} terms but {len(lengths)} encoded lengths - the "
            "encoder dropped or added a phrase"
        )
    spans: list[Span] = []
    cursor = positive_len
    for term, n in zip(terms, lengths):
        spans.append(Span(cursor, cursor + n, term.weight, term.phrase))
        cursor += n
    return spans


def sign_vector(text_len: int, spans: list[Span], strength: float,
                *, device=None, dtype=torch.float32) -> torch.Tensor:
    """Per-text-row multiplier for V: +1 normally, -a*w on the negative rows.

    This is the whole of eq. 6's right-hand factor. Rows outside any span keep
    their values untouched, which is what makes the concatenation work: the
    positive prompt is not re-weighted just because a negative term exists.
    """
    if text_len <= 0:
        raise ValueError("text_len must be positive")
    signs = torch.ones(text_len, device=device, dtype=dtype)
    for s in spans:
        if s.stop > text_len:
            raise ValueError(
                f"term {s.phrase!r} occupies rows {s.start}:{s.stop} but the "
                f"text span is only {text_len} rows. The conditioning reaching "
                "the model is not the one this patch was built from - re-run "
                "the NegPiP node after changing the prompt."
            )
        signs[s.start:s.stop] = -abs(strength) * s.weight
    return signs


def apply_value_signs(v: torch.Tensor, signs: torch.Tensor,
                      text_start: int = 0) -> torch.Tensor:
    """Scale V rows [text_start, text_start+len(signs)) by `signs`.

    `v` is H3's post-split value tensor, [B, heads, S, head_dim] - see
    `comfy/ldm/minimax/model.py:196`, which transposes to heads-major and adds
    the batch axis before handing it to `optimized_attention`.

    Returned as a new tensor. The caller is inside `torch.inference_mode()`
    during sampling, so writing into `v` in place would poison it for every
    later consumer.
    """
    if v.ndim != 4:
        raise ValueError(f"expected [B, heads, S, head_dim], got {tuple(v.shape)}")
    n = signs.shape[0]
    stop = text_start + n
    if stop > v.shape[2]:
        raise ValueError(
            f"text rows {text_start}:{stop} do not fit a sequence of "
            f"{v.shape[2]}. The layout changed under the patch."
        )
    if bool(torch.all(signs == 1.0)):
        return v
    out = v.clone()
    scale = signs.to(device=v.device, dtype=v.dtype).view(1, 1, n, 1)
    out[:, :, text_start:stop, :] = out[:, :, text_start:stop, :] * scale
    return out


def in_sigma_window(sigma: float, sigma_start: float, sigma_end: float) -> bool:
    """Inclusive window on the raw sigma, matching the other H3 patch nodes.

    Sigma runs high to low, so `sigma_start` is the larger number.
    """
    lo, hi = min(sigma_start, sigma_end), max(sigma_start, sigma_end)
    return lo - 1e-6 <= sigma <= hi + 1e-6


def extend_token_tags(tags: torch.Tensor, extra_rows: int) -> torch.Tensor:
    """Grow H3's per-token adaLN tag vector to cover the appended rows.

    Not optional. `comfy/ldm/minimax/model.py:684` does
    `tags = text_token_tags.view(-1).tolist()` and then indexes it up to the
    text segment's length, so a tags vector shorter than the text span raises
    IndexError inside the first DiT block. Appended rows are ordinary text, so
    they take tag 1 (see `comfy/text_encoders/minimax.py:78`, where text is 1
    and vision pads are 0).
    """
    if extra_rows < 0:
        raise ValueError("extra_rows must not be negative")
    if extra_rows == 0:
        return tags
    flat = tags.view(-1)
    pad = torch.ones(extra_rows, dtype=flat.dtype, device=flat.device)
    return torch.cat([flat, pad])


def describe_plan(plan: NegPiPPlan) -> str:
    """Plain English, for the node's report output and its on-node panel."""
    if not plan.spans:
        return (
            "No negative terms. Nothing is flipped and the model runs exactly "
            "as it would without this node."
        )
    rows = sum(s.stop - s.start for s in plan.spans)
    lines = [
        f"{len(plan.spans)} negative term(s), {rows} token row(s) appended to a "
        f"{plan.positive_len}-row prompt (text span is now {plan.text_len}).",
        f"Their attention VALUES are negated and scaled by {plan.strength:.2f}; "
        "their keys are untouched, so they still mean what they say.",
    ]
    for s in plan.spans:
        lines.append(
            f"  - {s.phrase!r}: rows {s.start}-{s.stop - 1}, "
            f"V x {-abs(plan.strength) * s.weight:+.2f}"
        )
    if plan.sigma_start < 1.0 or plan.sigma_end > 0.0:
        lines.append(
            f"Active only for sigma {plan.sigma_end:.3f}-{plan.sigma_start:.3f}."
        )
    lines.append(
        "This costs no extra sampling pass: unlike CFG, the negative terms ride "
        "in the same forward as the prompt."
    )
    return "\n".join(lines)
