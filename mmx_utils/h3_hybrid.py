"""H3 hybrid HDR — FL projection, LoRA factorisation, key helpers (pure math, no I/O)."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Callable

import torch

_BLOCK_RE = re.compile(r"^diffusion_model\.blocks\.(\d+)\.")
_WEIGHT_SUFFIX = ".weight"


def normalize_diffusion_key(raw: str) -> str:
    """Normalise checkpoint keys to diffusion_model.<rest>."""
    key = raw.strip()
    if key.startswith("model."):
        key = key[len("model.") :]
    if key.startswith("diffusion_model."):
        return key
    if key.startswith("blocks."):
        return f"diffusion_model.{key}"
    return key


def parse_block_index(key: str) -> int | None:
    norm = normalize_diffusion_key(key)
    m = _BLOCK_RE.match(norm)
    return int(m.group(1)) if m else None


def parse_block_range(spec: str) -> tuple[int, int]:
    text = spec.strip()
    if "-" not in text:
        n = int(text)
        return n, n
    lo_s, hi_s = text.split("-", 1)
    lo, hi = int(lo_s.strip()), int(hi_s.strip())
    if lo > hi:
        raise ValueError(f"Invalid block range {spec!r}: start must be <= end")
    return lo, hi


def key_in_block_range(key: str, start: int, end: int) -> bool:
    idx = parse_block_index(key)
    if idx is None:
        return True
    return start <= idx <= end


def is_diffusion_weight_key(key: str) -> bool:
    norm = normalize_diffusion_key(key)
    return norm.endswith(_WEIGHT_SUFFIX)


def weight_base_key(norm_weight_key: str) -> str:
    if not norm_weight_key.endswith(_WEIGHT_SUFFIX):
        raise ValueError(f"Expected a .weight key, got {norm_weight_key!r}")
    return norm_weight_key[: -len(_WEIGHT_SUFFIX)]


def fl_projection_coeff(
    delta: torch.Tensor,
    fl_dir: torch.Tensor,
    *,
    eps: float = 1e-12,
) -> float:
    """a = clamp(<delta, fl_dir> / <fl_dir, fl_dir>, 0, 1); 0 when ||D||^2 <= eps."""
    d32 = fl_dir.reshape(-1).float()
    num = (delta.reshape(-1).float() * d32).sum()
    den = (d32 * d32).sum()
    if float(den) <= eps:
        return 0.0
    a = float(num / den)
    return max(0.0, min(1.0, a))


def residual_after_projection(
    delta: torch.Tensor,
    fl_dir: torch.Tensor,
    a: float,
) -> torch.Tensor:
    return delta - float(a) * fl_dir


def tensor_energy_norm(t: torch.Tensor) -> float:
    return float(t.reshape(-1).float().pow(2).sum().sqrt().item())


def captured_energy_fraction(H: torch.Tensor, H_approx: torch.Tensor) -> float:
    h_norm = (H.reshape(-1).float().pow(2).sum()).item()
    if h_norm <= 0.0:
        return 1.0
    approx_norm = (H_approx.reshape(-1).float().pow(2).sum()).item()
    return float(min(1.0, max(0.0, approx_norm / h_norm)))


def combined_h_noise_floor(
    noise_r: float | None,
    noise_f: float | None,
    noise_s: float | None,
    a: float,
    *,
    no_fl_projection: bool = False,
) -> float | None:
    """Expected ||quant noise in H||^2 for H = S-R-a(F-R), weighted (1,(1-a)^2,a^2)."""
    if no_fl_projection:
        if noise_r is None or noise_s is None:
            return None
        return float(noise_s + noise_r)
    if noise_r is None or noise_f is None or noise_s is None:
        return None
    aa = float(a)
    return float(noise_s + (1.0 - aa) ** 2 * noise_r + aa**2 * noise_f)


def classify_parent(
    rel_to_ref: float,
    rel_to_fl: float,
    r_norm: float,
    f_norm: float,
    noise_r: float | None,
    noise_f: float | None,
    noise_s: float | None,
) -> str:
    if noise_r is not None and noise_f is not None and noise_s is not None and r_norm > 0 and f_norm > 0:
        thresh_ref = 3.0 * math.sqrt(max(noise_s + noise_r, 0.0)) / r_norm
        thresh_fl = 3.0 * math.sqrt(max(noise_s + noise_f, 0.0)) / f_norm
        if rel_to_ref <= thresh_ref:
            return "Ref2VA"
        if rel_to_fl <= thresh_fl:
            return "FL2VA"
        return "tuned"
    if rel_to_ref <= 0.005:
        return "Ref2VA"
    if rel_to_fl <= 0.005:
        return "FL2VA"
    return "tuned"


def is_trunk_weight_key(key: str) -> bool:
    return "adaln_proj" not in key


def is_adaln_weight_key(key: str) -> bool:
    return "adaln_proj.linear" in key


def noise_edge(floor_energy: float, m: int, n: int, margin: float) -> float:
    """Marchenko-Pastur noise edge for singular values of quantised residual H."""
    if floor_energy <= 0.0 or m <= 0 or n <= 0:
        return 0.0
    return float(
        margin * math.sqrt(floor_energy / (m * n)) * (math.sqrt(m) + math.sqrt(n))
    )


@dataclass(frozen=True)
class LoraFactors:
    up: torch.Tensor
    down: torch.Tensor
    alpha: float
    captured: float
    k: int
    n_above_edge: int
    rank_capped: bool
    energy_above_edge: float


def svd_lora_from_residual(
    H: torch.Tensor,
    rank: int,
    *,
    edge: float | None = None,
    niter: int = 4,
) -> LoraFactors | None:
    """Randomised SVD LoRA; optional MP edge keeps only s_i > edge."""
    if H.ndim != 2:
        raise ValueError(f"LoRA residual must be 2-D, got shape {tuple(H.shape)}")
    q = min(int(rank), min(H.shape))
    h32 = H.float()
    h_norm = float((h32 * h32).sum().item())
    if h32.numel() == 0 or h_norm <= 0.0:
        return None if edge is not None else LoraFactors(
            up=torch.zeros(H.shape, dtype=torch.bfloat16),
            down=torch.zeros(H.shape, dtype=torch.bfloat16),
            alpha=0.0,
            captured=1.0,
            k=0,
            n_above_edge=0,
            rank_capped=False,
            energy_above_edge=0.0,
        )

    u, s, v = torch.svd_lowrank(h32, q=q, niter=niter)

    if edge is not None:
        keep = s > edge
        k = int(keep.sum().item())
        if k == 0:
            return None
        idx = keep.nonzero(as_tuple=False).squeeze(-1)
        u_k = u[:, idx]
        s_k = s[idx]
        v_k = v[:, idx]
        n_above_edge = k
        rank_capped = bool(s[q - 1].item() > edge)
        energy_above_edge = float((s_k * s_k).sum().item() / h_norm)
    else:
        u_k = u[:, :q]
        s_k = s[:q]
        v_k = v[:, :q]
        k = q
        n_above_edge = q
        rank_capped = False
        energy_above_edge = float((s_k * s_k).sum().item() / h_norm)

    s_sqrt = s_k.sqrt()
    up32 = u_k * s_sqrt.unsqueeze(0)
    down32 = s_sqrt.unsqueeze(1) * v_k.T
    approx_norm = float((up32 @ down32).pow(2).sum().item())
    captured = approx_norm / h_norm if h_norm > 0.0 else 1.0
    up = up32.to(torch.bfloat16).contiguous()
    down = down32.to(torch.bfloat16).contiguous()
    return LoraFactors(
        up=up,
        down=down,
        alpha=float(k),
        captured=captured,
        k=k,
        n_above_edge=n_above_edge,
        rank_capped=rank_capped,
        energy_above_edge=energy_above_edge,
    )


def lora_state_dict_entries(
    norm_weight_key: str,
    factors: LoraFactors,
) -> dict[str, torch.Tensor]:
    base = weight_base_key(norm_weight_key)
    return {
        f"{base}.lora_up.weight": factors.up,
        f"{base}.lora_down.weight": factors.down,
        f"{base}.alpha": torch.tensor(factors.alpha, dtype=torch.float32),
    }


def diff_state_dict_entry(norm_weight_key: str, diff: torch.Tensor) -> dict[str, torch.Tensor]:
    base = weight_base_key(norm_weight_key)
    return {f"{base}.diff": diff.to(torch.bfloat16)}


def pair_tensor_keys(
    keys_ref: set[str],
    keys_fl: set[str],
    keys_sing: set[str],
    *,
    shapes_ref: Callable[[str], tuple[int, ...]],
    shapes_fl: Callable[[str], tuple[int, ...]],
    shapes_sing: Callable[[str], tuple[int, ...]],
) -> list[str]:
    common = sorted(keys_ref & keys_fl & keys_sing)
    out: list[str] = []
    for key in common:
        s0, s1, s2 = shapes_ref(key), shapes_fl(key), shapes_sing(key)
        if s0 != s1 or s0 != s2:
            raise ValueError(
                f"Shape mismatch on {key}: ref {list(s0)} fl {list(s1)} sing {list(s2)} — "
                "variants differ (pruned vs full); use matching builds."
            )
        out.append(key)
    return out


def aggregate_block_report(rows: list[dict]) -> dict[int, dict[str, float | str | int | None]]:
    """Aggregate per-tensor rows into per-block means."""
    buckets: dict[int, list[dict]] = {}
    for row in rows:
        blk = row.get("block")
        if blk is None:
            continue
        buckets.setdefault(int(blk), []).append(row)

    agg: dict[int, dict[str, float | str | int | None]] = {}
    for blk, items in sorted(buckets.items()):
        if not items:
            continue
        adaln = [x for x in items if is_adaln_weight_key(x.get("key", ""))]
        trunk = [x for x in items if is_trunk_weight_key(x.get("key", ""))]
        trunk_a = [x for x in trunk if math.isfinite(x.get("a", float("nan")))]
        trunk_h = [x for x in trunk if math.isfinite(x.get("h_over_r", float("nan")))]
        emitted_rows = [x for x in items if x.get("emitted")]
        trunk_emitted = [x for x in trunk if x.get("emitted")]
        ks = [int(x["k"]) for x in trunk_emitted if x.get("k") is not None]
        capped = [x for x in emitted_rows if x.get("rank_capped")]
        energies = [
            x["energy_above_edge"]
            for x in trunk_emitted
            if x.get("energy_above_edge") is not None
        ]
        emitted = len(emitted_rows)
        skipped = sum(1 for x in items if x.get("noise_only"))
        agg[blk] = {
            "adaln_parent": adaln[0]["parent"] if adaln else "?",
            "mean_a": float(sum(x["a"] for x in trunk_a) / len(trunk_a)) if trunk_a else float("nan"),
            "mean_h_over_r": float(sum(x["h_over_r"] for x in trunk_h) / len(trunk_h)) if trunk_h else float("nan"),
            "mean_k": float(sum(ks) / len(ks)) if ks else None,
            "rank_capped_pct": float(100.0 * len(capped) / len(emitted_rows)) if emitted_rows else 0.0,
            "mean_energy_above_edge": float(sum(energies) / len(energies)) if energies else None,
            "emitted": emitted,
            "skipped": skipped,
            "tensor_count": len(items),
        }
    return agg


def format_rank_recommendation(rows: list[dict], rank_cap: int) -> str:
    emitted = [r for r in rows if r.get("emitted")]
    if not emitted:
        return "Recommended rank: N/A (no tensors emitted)"
    capped = [r for r in emitted if r.get("rank_capped")]
    if capped:
        pct = 100.0 * len(capped) / len(emitted)
        return f"raise --rank: {pct:.0f}% of tensors hit the cap ({rank_cap})"
    ks = sorted(int(r["k"]) for r in emitted if r.get("k") is not None)
    if not ks:
        return "Recommended rank: N/A"
    idx = max(0, min(len(ks) - 1, int(math.ceil(0.9 * len(ks)) - 1)))
    return f"Recommended rank: {ks[idx]}"


def format_decomposition_header(block_agg: dict[int, dict]) -> str:
    if not block_agg:
        return "No block statistics available."
    adaln_by_block = {b: row.get("adaln_parent", "?") for b, row in block_agg.items()}
    lines = ["## Decomposition summary", ""]
    lo = min(adaln_by_block)
    hi = max(adaln_by_block)
    lo_parent = adaln_by_block.get(lo, "?")
    hi_parent = adaln_by_block.get(hi, "?")
    if lo <= 24 and 25 <= hi:
        p024 = adaln_by_block.get(0, lo_parent)
        p2549 = adaln_by_block.get(25, hi_parent)
        lines.append(
            f"Blocks 0-24 route like **{p024}** (adaln_proj); blocks 25-49 route like **{p2549}**. "
            "Singularity trunk weights moved largely along the FL2VA direction (a near 1.0 on attn/MLP); "
            "HDR fine-tune signal is selected by Marchenko-Pastur edge (singular values above quant noise)."
        )
    else:
        lines.append(
            "Per-block adaln_proj parent and trunk statistics are in the table below. "
            "Components above the MP noise edge are kept; spread-out quant noise stays out of the LoRA."
        )
    return "\n".join(lines)


def format_block_markdown(agg: dict[int, dict]) -> str:
    lines = [
        "| block | adaln parent | mean a (trunk) | mean ||H||/||R|| | mean k | capped % | mean energy above edge | emitted / skipped |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for blk in sorted(agg):
        row = agg[blk]
        mk = f"{row['mean_k']:.1f}" if row.get("mean_k") is not None else "n/a"
        eae = (
            f"{100.0 * row['mean_energy_above_edge']:.1f}%"
            if row.get("mean_energy_above_edge") is not None
            else "n/a"
        )
        lines.append(
            f"| {blk} | {row['adaln_parent']} | {row['mean_a']:.4f} | "
            f"{row['mean_h_over_r']:.4f} | {mk} | {row['rank_capped_pct']:.0f}% | {eae} | "
            f"{row['emitted']} / {row['skipped']} |"
        )
    return "\n".join(lines) + "\n"
