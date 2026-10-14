"""H3 Identity Fusion Controller (experimental) - C2C L8.26, owner A12 s1.2.

Controls WHERE, WHEN and HOW the character's identity enters H3 instead of HOW MUCH reference there is (RefMod's job,
C2C L8.29): per target-video token row, a multiplier on the attention that goes to the identity pictures
(<Picture N>), and optionally to the performance reference (the cropped plate, <Video 1>), in chosen blocks and a
sigma window. Inside the replacement region identity flows; outside it does not (no identity on the background or a
second person); on mouth / eye rows identity is eased and the plate's performance strengthened; on an occluder (a hand
over the face) identity is off.

Wiring follows NegPiP: an `optimized_attention_override` keyed off `transformer_options["minimax_h3_layout"]`, so no
DiT block is reserved and Block Cache / Spectrum keep working. Active blocks run ComfyUI's attention_pytorch with the
bias on padded head channels (exact; no S x S mask). Design: C2C docs/research/l826_identity_fusion_controller.md.
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

from mmx_utils.identity_fusion import (  # noqa: E402
    biased_attention,
    column_flags,
    describe,
    ref_groups,
    row_logweights,
    token_mask,
)
from mmx_utils.negpip import in_sigma_window  # noqa: E402

_LOG = logging.getLogger(__name__)
PLAN_KEY = "mmx_identity_fusion"


class _Plan:
    def __init__(self, ref_kinds, region, expression, occluder, cfg, sigmas):
        self.ref_kinds = list(ref_kinds)
        self.region, self.expression, self.occluder = region, expression, occluder
        self.cfg = cfg
        self.sigma_start, self.sigma_end, self.sigma_expr = sigmas


class _FusionAttention:
    """Stands in for `optimized_attention`. ComfyUI hands an override `(func, q, k, v, heads, **kwargs)` with the q/k/v
    containers already unwrapped (we have no `container_function`), `v` shaped [B, heads, S, d]."""

    def __init__(self, plan: _Plan, inner=None):
        self.plan = plan
        self.inner = inner
        self._cache: dict = {}
        self._warned = False
        self.stats = {"biased": 0, "passed": 0}

    def _bias(self, layout, device, dtype, with_expression: bool):
        key = (getattr(layout, "signature", None), layout.seq_len, str(device), dtype, with_expression)
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        p = self.plan
        groups = ref_groups(p.ref_kinds, layout.segments, int(p.cfg.get("performance_index", 0)))
        if groups.problem or groups.target is None:
            self._cache[key] = False
            return False
        _text_len, lt, lh, lw, _audio = layout.signature
        S = layout.seq_len
        region = token_mask(p.region, lt, lh, lw)
        expr = token_mask(p.expression, lt, lh, lw) if with_expression else None
        occ = token_mask(p.occluder, lt, lh, lw)
        c = p.cfg
        logw, logp = row_logweights(S, groups.target, region, expr, occ, w_inside=c["w_inside"],
                                    w_outside=c["w_outside"], w_expression=c["w_expression"],
                                    p_expression=c["p_expression"], w_occluder=c["w_occluder"])
        bias = {
            "logw": logw.to(device, dtype), "id_cols": column_flags(S, groups.identity).to(device, dtype),
            "logp": logp.to(device, dtype) if groups.performance else None,
            "perf_cols": column_flags(S, groups.performance).to(device, dtype) if groups.performance else None,
            "asym_rows": column_flags(S, groups.refs).to(device, dtype) if c.get("asymmetric") else None,
            "tgt_cols": column_flags(S, [groups.target]).to(device, dtype) if c.get("asymmetric") else None,
        }
        self._cache[key] = bias
        return bias

    def _active(self, v: torch.Tensor, opts: dict):
        layout = opts.get("minimax_h3_layout")
        if layout is None or v.ndim != 4 or v.shape[2] != getattr(layout, "seq_len", -1):
            return None                                        # not the H3 DiT (the token refiner is shorter)
        cou = opts.get("cond_or_uncond")
        if cou is not None and (0 not in cou or 1 in cou):
            return None                                        # positive pass only (H3 normally runs without CFG)
        c = self.plan.cfg
        bi = opts.get("block_index")
        if bi is not None and not (c["block_start"] <= int(bi) <= (c["block_end"] if c["block_end"] >= 0 else 10 ** 6)):
            return None
        sigma = None
        sig = opts.get("sigmas")
        if sig is not None:
            try:
                sigma = float(torch.as_tensor(sig).flatten()[0].item())
            except Exception:  # noqa: BLE001 - a weird sigma must not kill sampling
                sigma = None
        if sigma is not None and not in_sigma_window(sigma, self.plan.sigma_start, self.plan.sigma_end):
            return None
        with_expr = self.plan.expression is not None and (sigma is None or sigma <= self.plan.sigma_expr + 1e-6)
        bias = self._bias(layout, v.device, v.dtype, with_expr)
        return bias or None

    def __call__(self, func, *args, **kwargs):
        opts = kwargs.get("transformer_options") or {}
        bias = None
        if len(args) >= 4 and isinstance(args[2], torch.Tensor):
            try:
                bias = self._active(args[2], opts)
            except Exception as e:  # noqa: BLE001
                if not self._warned:
                    self._warned = True
                    _LOG.warning("H3 Identity Fusion disabled itself: %s", e)
                bias = None
        if bias is None:
            self.stats["passed"] += 1
            if self.inner is not None:
                return self.inner(func, *args, **kwargs)
            return func(*args, **kwargs)

        from comfy.ldm.modules.attention import attention_pytorch

        q, k, v, heads = args[0], args[1], args[2], args[3]
        want_heads_layout = bool(kwargs.get("skip_output_reshape"))
        kw = {key: val for key, val in kwargs.items() if key not in ("scale", "skip_reshape", "skip_output_reshape", "mask")}
        kw["_inside_attn_wrapper"] = True                      # never re-enter the override chain

        def attention(q2, k2, v2, scale, **_):
            call_kw = dict(kw, scale=scale, skip_reshape=True, skip_output_reshape=True)
            if self.inner is not None:                         # NegPiP & co. still see their rows (padded v)
                return self.inner(attention_pytorch, q2, k2, v2, heads, **call_kw)
            return attention_pytorch(q2, k2, v2, heads, **call_kw)

        out = biased_attention(q, k, v, logw=bias["logw"], id_cols=bias["id_cols"], logp=bias["logp"],
                               perf_cols=bias["perf_cols"], asym_rows=bias["asym_rows"], tgt_cols=bias["tgt_cols"],
                               attention=attention)
        self.stats["biased"] += 1
        if want_heads_layout:
            return out
        b, h, s, d = out.shape
        return out.transpose(1, 2).reshape(b, s, h * d)


def _visual_ref_kinds(positive) -> list[str]:
    for item in positive or []:
        extra = item[1] if isinstance(item, (list, tuple)) and len(item) > 1 else {}
        refs = (extra or {}).get("minimax_refs")
        if refs:
            return [str(r.get("kind", "")) for r in refs]
    return []


class MiniMaxH3_IdentityFusion(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        adv = {"optional": True, "advanced": True}
        return io.Schema(
            node_id="MiniMaxH3_IdentityFusion",
            display_name="H3 Identity Fusion Controller (experimental)",
            category="MiniMax H3/Conditioning",
            description=(
                "Where, when and how the character's identity enters H3: identity attention only inside the "
                "replacement region, eased on mouth / eye rows (the plate's performance strengthened there), off on "
                "occluders, in chosen blocks and steps. Use with RefMod or plain references - it weights attention, "
                "not the reference latents. Experimental: not A/B-tested on H3 weights."
            ),
            inputs=[
                io.Model.Input("model", tooltip="The H3 model, after Mask-Aware ControlNet if you use one."),
                io.Conditioning.Input("positive", tooltip="The positive from Reference to Video: it tells which "
                                      "references are the identity pictures and which is the plate video."),
                io.Mask.Input("replacement_mask", tooltip="Latent-space mask of the region to replace - H3 Mask To "
                              "Latent Space of head_room_masks, the same one Masked Replace uses."),
                io.Mask.Input("expression_mask", optional=True,
                              tooltip="Latent-space mouth + eyes mask of the PLATE (e.g. H3 Swap Control render -> "
                                      "mask -> Mask To Latent Space). There identity is eased so the performance "
                                      "comes through."),
                io.Mask.Input("occluder_mask", optional=True,
                              tooltip="Latent-space mask of things in front of the face (hands, hair, props): no "
                                      "identity there."),
                io.Boolean.Input("enabled", default=True),
                io.Float.Input("identity_inside", default=1.0, min=0.0, max=4.0, step=0.05, **adv,
                               tooltip="Multiplier on the attention to the identity pictures inside the region."),
                io.Float.Input("identity_outside", default=0.0, min=0.0, max=4.0, step=0.05, **adv,
                               tooltip="Outside the region. 0 keeps identity off the background and other people."),
                io.Float.Input("identity_on_expression", default=0.6, min=0.0, max=4.0, step=0.05, **adv),
                io.Float.Input("plate_on_expression", default=1.4, min=0.0, max=4.0, step=0.05, **adv,
                               tooltip="Multiplier on the attention to the plate video (<Video 1>) on mouth / eye "
                                       "rows."),
                io.Float.Input("identity_on_occluder", default=0.0, min=0.0, max=4.0, step=0.05, **adv),
                io.Float.Input("expression_from", default=0.4, min=0.0, max=1.0, step=0.05, **adv,
                               tooltip="Expression easing starts at this fraction of the steps (fine detail is late)."),
                io.Float.Input("start_percent", default=0.0, min=0.0, max=1.0, step=0.05, **adv),
                io.Float.Input("end_percent", default=1.0, min=0.0, max=1.0, step=0.05, **adv),
                io.Int.Input("block_start", default=0, min=0, max=200, **adv),
                io.Int.Input("block_end", default=-1, min=-1, max=200, **adv, tooltip="-1 = the last block."),
                io.Boolean.Input("asymmetric_flow", default=False, **adv,
                                 tooltip="References do not read the noisy target video (ContextAnyone's idea; "
                                         "untested on H3)."),
            ],
            outputs=[io.Model.Output(display_name="model"), io.String.Output(display_name="report")],
        )

    @classmethod
    def execute(cls, model, positive, replacement_mask, expression_mask=None, occluder_mask=None, enabled=True,
                identity_inside=1.0, identity_outside=0.0, identity_on_expression=0.6, plate_on_expression=1.4,
                identity_on_occluder=0.0, expression_from=0.4, start_percent=0.0, end_percent=1.0, block_start=0,
                block_end=-1, asymmetric_flow=False) -> io.NodeOutput:
        if not enabled:
            return io.NodeOutput(model, "H3 Identity Fusion: disabled - the model passes through unchanged.")
        kinds = _visual_ref_kinds(positive)
        if not any(k == "image" for k in kinds):
            raise ValueError("H3 Identity Fusion needs the positive from Reference to Video with at least one "
                             "reference image (<Picture 1>, the character).")
        ms = model.get_model_object("model_sampling")
        sigmas = (float(ms.percent_to_sigma(start_percent)), float(ms.percent_to_sigma(end_percent)),
                  float(ms.percent_to_sigma(expression_from)))
        cfg = {"w_inside": identity_inside, "w_outside": identity_outside, "w_expression": identity_on_expression,
               "p_expression": plate_on_expression, "w_occluder": identity_on_occluder, "block_start": block_start,
               "block_end": block_end, "start_percent": start_percent, "end_percent": end_percent,
               "asymmetric": bool(asymmetric_flow), "performance_index": 0}
        plan = _Plan(kinds, replacement_mask, expression_mask, occluder_mask, cfg, sigmas)

        model = model.clone()
        opts = model.model_options["transformer_options"].copy()
        existing = opts.get("optimized_attention_override")
        if isinstance(existing, _FusionAttention):
            raise ValueError("this model already has an Identity Fusion Controller; use one per model.")
        opts[PLAN_KEY] = plan
        opts["optimized_attention_override"] = _FusionAttention(plan, existing)
        model.model_options["transformer_options"] = opts

        def rows(m):
            return int((m.detach().float() > 0.5).sum().item()) if m is not None else 0

        from mmx_utils.identity_fusion import RefGroups
        g = RefGroups(identity=[(0, 1)] * kinds.count("image"),
                      performance=[(0, 1)] if any(k != "image" for k in kinds) else [])
        report = describe(g, rows(replacement_mask), rows(expression_mask), rows(occluder_mask), cfg)
        if existing is not None:
            report += "\nAnother attention override (e.g. NegPiP) was installed first; it still runs inside this one."
        report += "\n(row counts are latent pixels of the masks; token rows are counted at run time)"
        return io.NodeOutput(model, report)
