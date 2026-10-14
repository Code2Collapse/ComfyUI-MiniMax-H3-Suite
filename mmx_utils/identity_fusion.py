"""H3 Identity Fusion Controller - the pure core (C2C L8.26, owner A12 s1.2).

WHERE, WHEN and HOW identity enters H3, at the attention level: per target-video query row, a multiplier on the
attention mass that goes to the identity-reference keys (and optionally to the performance reference, the cropped
plate). RefMod already scales reference LATENTS ("not an attention weight", C2C L8.29); this is the part it does not do.

Mechanism (design: C2C docs/research/l826_identity_fusion_controller.md s3). The bias a row needs is rank-1 per key
group, so it rides on extra head channels instead of a dense S x S mask (GBs at H3 sequence lengths, and refused by
flash / sage):

    q' = [q, log w_i, log p_i, a_i, 0...]   k' = [k, s*[j in ID], s*[j in PERF], s*[j in TARGET], 0...]   v' = [v, 0...]
    softmax(q'.k'^T / sqrt(d)) with s = sqrt(d)  ==  softmax(q.k^T / sqrt(d) + bias)        (exact)

Pure torch; no ComfyUI import, so it is tested on CPU without a model.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence

import torch
import torch.nn.functional as F

LOG_ZERO = -30.0          # e^-30 ~ 1e-13: "no attention" without an infinity in the channel
MAX_WEIGHT = 4.0
PAD = 8                   # extra head channels, a multiple of 8 for the attention kernels

VISUAL_KINDS = ("image", "video", "video_audio")


@dataclass
class RefGroups:
    """Key-column spans, in sequence rows."""
    identity: list[tuple[int, int]] = field(default_factory=list)       # <Picture N> image references
    performance: list[tuple[int, int]] = field(default_factory=list)    # <Video N> video references
    target: tuple[int, int] | None = None                               # the target video rows
    refs: list[tuple[int, int]] = field(default_factory=list)          # every visual reference span (for asymmetry)
    problem: str = ""


def ref_groups(ref_kinds: Sequence[str], segments: Sequence[tuple[int, int, str]],
               performance_index: int = 0) -> RefGroups:
    """Map the conditioning's visual reference blocks (images first, then videos, as core 0.39's Reference to Video
    emits them) onto the layout's `ref_img` segments, which appear in the same order."""
    g = RefGroups()
    spans = [(a, b) for a, b, kind in segments if kind == "ref_img"]
    visual = [k for k in ref_kinds if k in VISUAL_KINDS]
    vid = [(a, b, kind) for a, b, kind in segments if kind == "video"]
    if vid:
        g.target = (vid[-1][0], vid[-1][1])
    if len(spans) != len(visual):
        g.problem = (f"the layout has {len(spans)} visual reference segment(s) but the conditioning lists "
                     f"{len(visual)} visual reference(s)")
        return g
    g.refs = spans
    videos = [s for s, k in zip(spans, visual) if k != "image"]
    g.identity = [s for s, k in zip(spans, visual) if k == "image"]
    if videos and 0 <= performance_index < len(videos):
        g.performance = [videos[performance_index]]
    if not g.identity:
        g.problem = "no identity picture (<Picture N>) in the conditioning"
    return g


def token_mask(mask: torch.Tensor | None, latent_t: int, latent_h: int, latent_w: int) -> torch.Tensor | None:
    """A latent-space mask [T,H,W] (or [T,1,H,W] / [H,W]) -> one bool per target-video token row
    (frame-major, then the 2x2 patch grid row by row - core's PackedLayout order). Nearest resize to the layout's
    latent grid, then 2x2 max: a token touched by the mask counts."""
    if mask is None:
        return None
    m = mask.detach().float()
    if m.ndim == 2:
        m = m[None]
    if m.ndim == 4:
        m = m[:, 0] if m.shape[1] == 1 else m[..., 0]
    t = m.shape[0]
    idx = torch.linspace(0, t - 1, latent_t).round().long() if t != latent_t else torch.arange(t)
    m = m[idx]
    m = F.interpolate(m[:, None], size=(latent_h, latent_w), mode="nearest")[:, 0]
    m = F.max_pool2d(m[:, None], kernel_size=2, stride=2)[:, 0]          # [T, H/2, W/2]
    return (m > 0.5).reshape(-1)


def _log(w: float) -> float:
    w = max(0.0, min(MAX_WEIGHT, float(w)))
    return LOG_ZERO if w <= 0.0 else math.log(w)


def row_logweights(seq_len: int, target: tuple[int, int], region: torch.Tensor | None,
                   expression: torch.Tensor | None, occluder: torch.Tensor | None, *,
                   w_inside: float = 1.0, w_outside: float = 0.0, w_expression: float = 0.6,
                   p_expression: float = 1.4, w_occluder: float = 0.0) -> tuple[torch.Tensor, torch.Tensor]:
    """log-weights per sequence row for (identity keys, performance keys). Only target-video rows change;
    w = O ? w_occ : (R ? (E ? w_expr : w_in) : w_out); p = E ? p_expr : 1."""
    lw = torch.zeros(seq_len)
    lp = torch.zeros(seq_len)
    a, b = target
    n = b - a

    def fit(x):
        if x is None:
            return None
        x = x.bool()
        if x.numel() >= n:
            return x[:n]
        return torch.cat([x, torch.zeros(n - x.numel(), dtype=torch.bool)])

    R, E, O = fit(region), fit(expression), fit(occluder)
    w = torch.full((n,), _log(w_inside))
    if R is not None:
        w = torch.where(R, w, torch.tensor(_log(w_outside)))
    if E is not None:
        inside_e = E if R is None else (E & R)
        w = torch.where(inside_e, torch.tensor(_log(w_expression)), w)
        lp[a:b] = torch.where(inside_e, torch.tensor(_log(p_expression)), torch.tensor(0.0))
    if O is not None:
        w = torch.where(O, torch.tensor(_log(w_occluder)), w)
    lw[a:b] = w
    return lw, lp


def column_flags(seq_len: int, spans: Sequence[tuple[int, int]]) -> torch.Tensor:
    f = torch.zeros(seq_len)
    for a, b in spans:
        f[a:b] = 1.0
    return f


def biased_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *, logw: torch.Tensor, id_cols: torch.Tensor,
                     logp: torch.Tensor | None = None, perf_cols: torch.Tensor | None = None,
                     asym_rows: torch.Tensor | None = None, tgt_cols: torch.Tensor | None = None,
                     attention=None, **attn_kwargs) -> torch.Tensor:
    """q,k,v [B, heads, S, d] -> [B, heads, S, d], with the per-row bias carried on padded channels.
    `attention(q, k, v, scale=..., **attn_kwargs)` must return the HEAD layout [B, heads, S, d+PAD] (ComfyUI's
    attention_pytorch with skip_reshape=True, skip_output_reshape=True); default torch SDPA. The padded output
    channels are cut off here, so the caller reshapes a plain d-wide result."""
    B, H, S, d = q.shape
    s = math.sqrt(d)
    pq = torch.zeros(B, H, S, PAD, dtype=q.dtype, device=q.device)
    pk = torch.zeros(B, H, S, PAD, dtype=k.dtype, device=k.device)
    pq[..., 0] = logw.to(q.device, q.dtype)
    pk[..., 0] = (id_cols * s).to(k.device, k.dtype)
    if logp is not None and perf_cols is not None:
        pq[..., 1] = logp.to(q.device, q.dtype)
        pk[..., 1] = (perf_cols * s).to(k.device, k.dtype)
    if asym_rows is not None and tgt_cols is not None:
        pq[..., 2] = (asym_rows * LOG_ZERO).to(q.device, q.dtype)
        pk[..., 2] = (tgt_cols * s).to(k.device, k.dtype)
    q2 = torch.cat([q, pq], dim=-1)
    k2 = torch.cat([k, pk], dim=-1)
    v2 = torch.cat([v, torch.zeros(B, H, S, PAD, dtype=v.dtype, device=v.device)], dim=-1)
    if attention is None:
        out = F.scaled_dot_product_attention(q2, k2, v2, scale=1.0 / s)
    else:
        out = attention(q2, k2, v2, scale=1.0 / s, **attn_kwargs)
    return out[..., :d]


def describe(groups: RefGroups, region_rows: int, expr_rows: int, occ_rows: int, cfg: dict[str, Any]) -> str:
    if groups.problem:
        return f"H3 Identity Fusion: inactive - {groups.problem}."
    lines = [
        f"H3 Identity Fusion (experimental): identity = {len(groups.identity)} picture(s), "
        f"performance reference = {'<Video 1>' if groups.performance else 'none'}.",
        f"replacement region {region_rows} token rows: identity x{cfg['w_inside']:.2f} inside, "
        f"x{cfg['w_outside']:.2f} outside.",
    ]
    if expr_rows:
        lines.append(f"expression rows {expr_rows}: identity x{cfg['w_expression']:.2f}, "
                     f"performance x{cfg['p_expression']:.2f}.")
    if occ_rows:
        lines.append(f"occluder rows {occ_rows}: identity x{cfg['w_occluder']:.2f}.")
    if cfg.get("asymmetric"):
        lines.append("asymmetric flow: reference rows do not read the target video.")
    lines.append(f"blocks {cfg['block_start']}-{cfg['block_end']}, steps {cfg['start_percent']:.0%}-"
                 f"{cfg['end_percent']:.0%}. Not A/B-tested on H3 weights.")
    return "\n".join(lines)
