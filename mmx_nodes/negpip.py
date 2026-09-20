"""MiniMax H3 NegPiP — negative prompting that costs no second forward pass.

See mmx_utils/negpip.py for the method and the citations. This file is the
wiring: where the value flip is allowed to happen, and - more importantly -
where it must not.

Three gates decide that, and all three are free:

1. `transformer_options["minimax_h3_layout"]` only exists inside
   `MiniMaxH3Model._forward` (set at `comfy/ldm/minimax/model.py:623`, whose
   comment says "segment spans for attention patches"). Its absence means we
   are somewhere else entirely.
2. The value tensor's sequence length must equal `layout.seq_len`. H3's token
   refiner (`model.py:263`) runs the same `optimized_attention` over the TEXT
   ONLY, so it has a shorter sequence. Flipping there would corrupt the text
   representation itself before the DiT ever sees it - the refiner is how the
   prompt becomes usable, not how it is applied.
3. The text segment's length must equal the length this patch was built from.
   H3 runs the positive and the CFG-negative conditionings as separate forward
   passes, and a length that does not match ours belongs to another one.

Because none of that needs `patches_replace`, this node does not reserve a
single DiT block, so it composes with Block Cache and the Spectrum sampler
instead of fighting them for the same slot.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import torch
from comfy_api.latest import io

_PKG = Path(__file__).resolve().parents[1]
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from mmx_utils.negpip import (  # noqa: E402
    NEGPIP_KEY,
    NegPiPPlan,
    describe_plan,
    extend_token_tags,
    in_sigma_window,
    parse_terms,
    sign_vector,
    spans_from_lengths,
)

_LOG = logging.getLogger(__name__)

# H3's own text tag. `comfy/text_encoders/minimax.py:78` fills the tag vector
# with ones and only writes 0 over vision-pad spans, so appended prompt text
# is 1.
TAGS_KEY = "minimax_token_tags"


# ── the attention override ──────────────────────────────────────────────────

class _NegPiPAttention:
    """Stands in for `optimized_attention`, flipping V on the negative rows.

    ComfyUI hands an override `(func, q, k, v, heads, **kwargs)` where `func`
    is the undecorated backend (`comfy/ldm/modules/attention.py:193`). We are
    a plain object with no `container_function`, so the q/k/v containers are
    unwrapped for us and `v` arrives as a real tensor shaped
    [B, heads, S, head_dim].
    """

    def __init__(self, plan: NegPiPPlan, inner=None):
        self.plan = plan
        self.inner = inner            # whatever override was already installed
        self._signs: torch.Tensor | None = None
        self._warned = False

    def _cached_signs(self, text_len: int, device, dtype) -> torch.Tensor:
        s = self._signs
        if (s is None or s.shape[0] != text_len
                or s.device != device or s.dtype != dtype):
            s = sign_vector(text_len, self.plan.spans, self.plan.strength,
                            device=device, dtype=dtype)
            self._signs = s
        return s

    def _should_flip(self, v: torch.Tensor, transformer_options: dict) -> int | None:
        """Return the text segment's start row, or None to leave V alone."""
        layout = transformer_options.get("minimax_h3_layout")
        if layout is None:
            return None                                    # gate 1
        if v.ndim != 4 or v.shape[2] != getattr(layout, "seq_len", -1):
            return None                                    # gate 2 - the refiner
        text = next((s for s in layout.segments if s[2] == "text"), None)
        if text is None:
            return None
        start, stop, _ = text
        if stop - start != self.plan.text_len:
            return None                                    # gate 3 - other cond
        if self.plan.positive_only:
            cou = transformer_options.get("cond_or_uncond")
            # CFG's negative pass must keep its own meaning; flipping there
            # would subtract the terms from the thing already being subtracted.
            if cou is not None and 0 not in cou:
                return None
        sigmas = transformer_options.get("sigmas")
        if sigmas is not None:
            try:
                sigma = float(torch.as_tensor(sigmas).flatten()[0].item())
            except Exception:  # noqa: BLE001 - a weird sigma must not kill sampling
                sigma = None
            if sigma is not None and not in_sigma_window(
                    sigma, self.plan.sigma_start, self.plan.sigma_end):
                return None
        return start

    def __call__(self, func, *args, **kwargs):
        transformer_options = kwargs.get("transformer_options") or {}
        if len(args) >= 3 and isinstance(args[2], torch.Tensor):
            try:
                start = self._should_flip(args[2], transformer_options)
            except Exception as e:  # noqa: BLE001
                if not self._warned:
                    self._warned = True
                    _LOG.warning("MiniMax H3 NegPiP disabled itself: %s", e)
                start = None
            if start is not None:
                v = args[2]
                signs = self._cached_signs(self.plan.text_len, v.device, v.dtype)
                # Not in place: sampling runs under torch.inference_mode(), and
                # writing into `v` would poison the buffer for every later read.
                flipped = v.clone()
                n = signs.shape[0]
                flipped[:, :, start:start + n, :] *= signs.view(1, 1, n, 1)
                args = (args[0], args[1], flipped) + args[3:]
        if self.inner is not None:
            return self.inner(func, *args, **kwargs)
        return func(*args, **kwargs)


# ── the node ────────────────────────────────────────────────────────────────

class MiniMaxH3_NegPiP(io.ComfyNode):

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3_NegPiP",
            display_name="MiniMax H3 NegPiP (negative prompt in prompt)",
            category="MiniMax H3/Conditioning",
            description=(
                "Negative prompting for H3 that rides in the SAME forward pass "
                "as the prompt, so it costs no extra step time the way CFG "
                "does. The terms you list are encoded and appended to the text "
                "conditioning; inside the DiT their attention VALUES are "
                "negated while their keys are left alone (Value Sign Flip, "
                "arXiv:2508.10931, the same trick as hako-mikan's NegPiP). A "
                "patch that matches 'blurry' therefore gets the direction of "
                "'blurry' subtracted from it, weighted by how strongly it "
                "matched - so the push is adaptive per layer, per step and per "
                "token instead of a single global CFG scale. H3 suits this "
                "unusually well: it has no cross-attention, just one joint "
                "self-attention over a packed sequence, so the concatenation "
                "the method needs is already there."
            ),
            inputs=[
                io.Model.Input(
                    "model",
                    tooltip="A MiniMax H3 model. The patch is a pure attention "
                            "override - it reserves no DiT block, so it stacks "
                            "with Block Cache and the Spectrum sampler."),
                io.Clip.Input(
                    "clip",
                    tooltip="The same H3 text encoder that built the "
                            "conditioning. Each term is encoded through it "
                            "once, here, not per sampling step."),
                io.Conditioning.Input(
                    "conditioning",
                    tooltip="Your POSITIVE conditioning, from any H3 encode "
                            "node - the terms are appended to it. Feed the "
                            "result to the sampler in its place."),
                io.String.Input(
                    "terms", multiline=True, default="",
                    tooltip="One phrase per line, each a thing to push AWAY "
                            "from. Add a weight with a trailing colon - "
                            "'blurry : 1.5'. Blank lines and # comments are "
                            "ignored. These are already negative, so the "
                            "weight is how hard to push, never a sign: a "
                            "negative number here is refused rather than "
                            "quietly turned back into positive guidance."),
                io.Float.Input(
                    "strength", default=1.0, min=0.0, max=3.0, step=0.05,
                    tooltip="The alpha of eq. 6 - a global multiplier on every "
                            "term. 1.0 is the paper's setting. Past about 2 "
                            "the flipped values start to dominate the "
                            "attention output and the frame degrades, which "
                            "looks like burn-in rather than like a bad prompt."),
                io.Combo.Input(
                    "context_mode", options=["standalone", "prompt_prefix"],
                    default="standalone",
                    tooltip="standalone: encode each term on its own. Cheap, "
                            "and what the original method does. prompt_prefix: "
                            "encode it after your prompt and keep only the "
                            "term's rows, so the text encoder (a causal LLM) "
                            "sees the term in context and produces states on "
                            "the same scale as the prompt's. Better aimed, but "
                            "one extra encode per term, and it needs the "
                            "prompt text below."),
                io.String.Input(
                    "prompt_prefix", multiline=True, default="", optional=True,
                    tooltip="Only for context_mode=prompt_prefix: the exact "
                            "prompt text used to build the conditioning above. "
                            "The node verifies that the prefix rows came back "
                            "unchanged and falls back to standalone, saying so "
                            "in the report, if they did not."),
                io.Float.Input(
                    "start_percent", default=0.0, min=0.0, max=1.0, step=0.01,
                    optional=True,
                    tooltip="Begin flipping at this point through the "
                            "schedule. Leaving it at 0 applies the terms "
                            "throughout, which is the usual choice - the early "
                            "steps are where layout is decided, so that is "
                            "where a 'no extra limbs' term earns its keep."),
                io.Float.Input(
                    "end_percent", default=1.0, min=0.0, max=1.0, step=0.01,
                    optional=True,
                    tooltip="Stop flipping here. Ending around 0.8 lets the "
                            "last steps refine texture without the negative "
                            "terms still pulling at it."),
            ],
            outputs=[
                io.Model.Output(display_name="model"),
                io.Conditioning.Output(display_name="conditioning"),
                io.String.Output(display_name="report"),
            ],
        )

    # ── helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def _encode(clip, text: str) -> torch.Tensor:
        out = clip.encode_from_tokens(clip.tokenize(text), return_dict=True)
        cond = out["cond"]
        if cond.ndim != 3:
            raise ValueError(
                f"the text encoder returned a {cond.ndim}-D conditioning; "
                "NegPiP needs [batch, tokens, features]"
            )
        return cond

    @classmethod
    def _encode_terms(cls, clip, terms, context_mode, prompt_prefix):
        """Return (list of [1, n, D] row blocks, list of notes)."""
        notes: list[str] = []
        if context_mode == "prompt_prefix":
            if not prompt_prefix.strip():
                raise ValueError(
                    "context_mode is prompt_prefix but prompt_prefix is empty. "
                    "Paste the same prompt text you used to build the "
                    "conditioning, or switch context_mode back to standalone."
                )
            prefix = cls._encode(clip, prompt_prefix)
            lp = prefix.shape[1]
            blocks = []
            for term in terms:
                joined = cls._encode(clip, f"{prompt_prefix.rstrip()} {term.phrase}")
                if joined.shape[1] <= lp:
                    notes.append(
                        f"{term.phrase!r} added no tokens after the prompt; "
                        "encoded it standalone instead."
                    )
                    blocks.append(cls._encode(clip, term.phrase))
                    continue
                head = joined[:, :lp]
                # The encoder is causal, so the prefix rows must be bit-stable.
                # If they are not, the tokeniser merged across the boundary and
                # the tail is not cleanly the term's.
                drift = (head.float() - prefix.float()).abs().max().item()
                if drift > 1e-3:
                    notes.append(
                        f"{term.phrase!r} changed the prompt's own token states "
                        f"by {drift:.4f} (the tokeniser merged at the join), so "
                        "it was encoded standalone instead."
                    )
                    blocks.append(cls._encode(clip, term.phrase))
                else:
                    blocks.append(joined[:, lp:])
            return blocks, notes
        return [cls._encode(clip, t.phrase) for t in terms], notes

    # ── execute ────────────────────────────────────────────────────────────

    @classmethod
    def execute(cls, model, clip, conditioning, terms, strength,
                context_mode="standalone", prompt_prefix="",
                start_percent=0.0, end_percent=1.0):
        parsed = parse_terms(terms)
        if not parsed:
            return io.NodeOutput(
                model, conditioning,
                "No negative terms. Nothing is flipped and the model runs "
                "exactly as it would without this node.",
            )
        if not conditioning:
            raise ValueError("conditioning is empty - connect an H3 text encode node")

        lengths = {c[0].shape[1] for c in conditioning}
        if len(lengths) != 1:
            raise ValueError(
                f"this conditioning holds {len(lengths)} different prompt "
                f"lengths ({sorted(lengths)}). NegPiP identifies its own "
                "conditioning at sample time by that length, so combine the "
                "prompts before this node, not after."
            )
        positive_len = lengths.pop()

        blocks, notes = cls._encode_terms(clip, parsed, context_mode, prompt_prefix)
        spans = spans_from_lengths(positive_len, parsed, [b.shape[1] for b in blocks])

        out_conditioning = []
        for cond, extra in conditioning:
            extra = extra.copy()
            rows = [b.to(device=cond.device, dtype=cond.dtype) for b in blocks]
            if any(r.shape[-1] != cond.shape[-1] for r in rows):
                raise ValueError(
                    "the terms encoded to a different feature width than the "
                    "conditioning - the clip input is not the encoder that "
                    "built it"
                )
            new_cond = torch.cat([cond] + rows, dim=1)
            tags = extra.get(TAGS_KEY)
            if tags is not None:
                # Mandatory. model.py:684 indexes this vector across the whole
                # text span, so a short one is an IndexError in block 0.
                extra[TAGS_KEY] = extend_token_tags(tags, new_cond.shape[1] - cond.shape[1])
            out_conditioning.append([new_cond, extra])

        model_sampling = model.get_model_object("model_sampling")
        plan = NegPiPPlan(
            positive_len=positive_len,
            spans=spans,
            strength=strength,
            sigma_start=float(model_sampling.percent_to_sigma(start_percent)),
            sigma_end=float(model_sampling.percent_to_sigma(end_percent)),
        )

        model = model.clone()
        transformer_options = model.model_options["transformer_options"].copy()
        existing = transformer_options.get("optimized_attention_override")
        if isinstance(existing, _NegPiPAttention):
            # A second NegPiP node would otherwise stack two flips and the
            # inner one would silently win on the same rows.
            raise ValueError(
                "this model already has a NegPiP patch. Put all your terms in "
                "one node - chaining two of them flips the same rows twice."
            )
        transformer_options[NEGPIP_KEY] = plan
        transformer_options["optimized_attention_override"] = _NegPiPAttention(plan, existing)
        model.model_options["transformer_options"] = transformer_options

        report = [describe_plan(plan)]
        if context_mode == "prompt_prefix":
            report.append(
                "Terms encoded with your prompt as a causal prefix, so their "
                "hidden states sit on the same scale as the prompt's."
            )
        report.extend(notes)
        if existing is not None:
            report.append(
                "Another attention override was already installed; it still "
                "runs, with the flipped values passed through to it."
            )
        report.append(
            "Not flipped: the token refiner (it builds the prompt, it does not "
            "apply it) and the CFG negative pass."
        )
        return io.NodeOutput(model, out_conditioning, "\n".join(report))
