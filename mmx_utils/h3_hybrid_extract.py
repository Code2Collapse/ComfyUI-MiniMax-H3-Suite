"""H3 hybrid LoRA extraction — stream three matched checkpoints, FL-project, SVD compress."""

from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Callable

import torch
from safetensors.torch import save_file

from mmx_utils.h3_checkpoint_io import open_checkpoint
from mmx_utils.h3_hybrid import (
    aggregate_block_report,
    classify_parent,
    combined_h_noise_floor,
    diff_state_dict_entry,
    fl_projection_coeff,
    format_block_markdown,
    format_decomposition_header,
    format_rank_recommendation,
    is_diffusion_weight_key,
    key_in_block_range,
    lora_state_dict_entries,
    normalize_diffusion_key,
    noise_edge,
    pair_tensor_keys,
    parse_block_index,
    residual_after_projection,
    svd_lora_from_residual,
    tensor_energy_norm,
)


def extract_hybrid_lora(
    ref_path: str,
    fl_path: str,
    sing_path: str,
    out_path: str,
    *,
    rank: int = 64,
    no_fl_projection: bool = False,
    block_start: int | None = None,
    block_end: int | None = None,
    edge_margin: float = 1.1,
    device: str = "cpu",
    report_stem: str | None = None,
    progress_cb: Callable[[], None] | None = None,
    progress_init: Callable[[int], None] | None = None,
    interrupt_check: Callable[[], None] | None = None,
) -> dict:
    """
    Extract rank-r LoRA of HDR residual H from Ref2VA/FL2VA/Singularity triple.

    Streams one 2-D weight at a time. Returns summary dict with markdown report text.
    """
    ref_r = open_checkpoint(ref_path, device=device)
    fl_r = open_checkpoint(fl_path, device=device)
    sing_r = open_checkpoint(sing_path, device=device)

    try:
        weight_keys = {
            normalize_diffusion_key(k)
            for k in ref_r.tensor_keys()
            if is_diffusion_weight_key(k)
        }
        weight_keys &= {normalize_diffusion_key(k) for k in fl_r.tensor_keys() if is_diffusion_weight_key(k)}
        weight_keys &= {normalize_diffusion_key(k) for k in sing_r.tensor_keys() if is_diffusion_weight_key(k)}

        paired = pair_tensor_keys(
            weight_keys,
            weight_keys,
            weight_keys,
            shapes_ref=ref_r.shape,
            shapes_fl=fl_r.shape,
            shapes_sing=sing_r.shape,
        )

        if block_start is not None and block_end is not None:
            paired = [
                k
                for k in paired
                if key_in_block_range(k, block_start, block_end)
            ]

        if progress_init is not None:
            progress_init(len(paired))

        lora_out: dict[str, torch.Tensor] = {}
        rows: list[dict] = []
        skipped_1d: list[str] = []
        noise_only: list[str] = []
        emitted_count = 0

        for key in paired:
            if interrupt_check is not None:
                interrupt_check()

            shape = ref_r.shape(key)
            if len(shape) != 2:
                if len(shape) == 1:
                    delta = sing_r.read_tensor(key) - ref_r.read_tensor(key)
                    skipped_1d.append(key)
                    try:
                        lora_out.update(diff_state_dict_entry(key, delta.float()))
                    except Exception:
                        pass
                if progress_cb is not None:
                    progress_cb()
                continue

            R = ref_r.read_tensor(key)
            F = fl_r.read_tensor(key)
            S = sing_r.read_tensor(key)
            D = F - R
            delta = S - R
            if no_fl_projection:
                a = 0.0
                H = delta
            else:
                a = fl_projection_coeff(delta, D)
                H = residual_after_projection(delta, D, a)

            r_norm = tensor_energy_norm(R)
            f_norm = tensor_energy_norm(F)
            rel_to_ref = tensor_energy_norm(S - R) / r_norm if r_norm > 0 else float("nan")
            rel_to_fl = tensor_energy_norm(S - F) / f_norm if f_norm > 0 else float("nan")

            noise_r = ref_r.noise_energy(key)
            noise_f = fl_r.noise_energy(key)
            noise_s = sing_r.noise_energy(key)
            floor = combined_h_noise_floor(
                noise_r, noise_f, noise_s, a, no_fl_projection=no_fl_projection
            )
            h_norm = tensor_energy_norm(H)
            h_sq = h_norm * h_norm
            snr = h_sq / floor if floor is not None and floor > 0.0 else None
            parent = classify_parent(rel_to_ref, rel_to_fl, r_norm, f_norm, noise_r, noise_f, noise_s)

            edge_val: float | None = None
            k: int | None = None
            rank_capped = False
            energy_above_edge: float | None = None
            captured_pct: float | None = None
            emit = False

            if floor is not None and floor > 0.0:
                edge_val = noise_edge(floor, H.shape[0], H.shape[1], float(edge_margin))
                factors = svd_lora_from_residual(H, rank, edge=edge_val)
                if factors is not None:
                    emit = True
                    k = factors.k
                    rank_capped = factors.rank_capped
                    energy_above_edge = factors.energy_above_edge
                    captured_pct = factors.captured * 100.0
                    lora_out.update(lora_state_dict_entries(key, factors))
                    emitted_count += 1
                else:
                    k = 0
                    noise_only.append(key)
            else:
                factors = svd_lora_from_residual(H, rank, edge=None)
                if factors is not None:
                    emit = True
                    k = factors.k
                    energy_above_edge = factors.energy_above_edge
                    captured_pct = factors.captured * 100.0
                    lora_out.update(lora_state_dict_entries(key, factors))
                    emitted_count += 1

            noise_only_flag = k == 0
            # The parent test tolerates quantisation noise, so a tensor can
            # match its parent within noise and still carry a structured
            # fine-tune above the edge (real Singularity trunk layers do).
            if k and parent in ("FL2VA", "Ref2VA"):
                parent = f"{parent} + fine-tune"

            blk = parse_block_index(key)
            rows.append(
                {
                    "key": key,
                    "block": blk,
                    "parent": parent,
                    "rel_to_ref": rel_to_ref,
                    "rel_to_fl": rel_to_fl,
                    "a": a,
                    "delta_norm": tensor_energy_norm(delta),
                    "h_norm": h_norm,
                    "h_over_r": h_norm / r_norm if r_norm > 0 else float("nan"),
                    "noise_floor": floor,
                    "noise_edge": edge_val,
                    "snr": snr,
                    "k": k,
                    "rank_capped": rank_capped,
                    "energy_above_edge": energy_above_edge,
                    "captured_pct": captured_pct,
                    "emitted": emit,
                    "noise_only": noise_only_flag,
                }
            )

            del R, F, S, D, delta, H
            gc.collect()

            if progress_cb is not None:
                progress_cb()

        out_path_p = Path(out_path)
        out_path_p.parent.mkdir(parents=True, exist_ok=True)
        save_file(lora_out, str(out_path_p))

        block_agg = aggregate_block_report(rows)
        md_table = format_block_markdown(block_agg)
        md_header = format_decomposition_header(block_agg)
        summary = {
            "ref": str(ref_path),
            "fl": str(fl_path),
            "sing": str(sing_path),
            "out": str(out_path_p),
            "rank": int(rank),
            "edge_margin": float(edge_margin),
            "fl_projection": not no_fl_projection,
            "block_start": block_start,
            "block_end": block_end,
            "tensor_count": emitted_count,
            "noise_only_count": len(noise_only),
            "skipped_1d": skipped_1d,
            "noise_only": noise_only,
            "per_tensor": rows,
            "per_block": {str(k): v for k, v in block_agg.items()},
        }

        md_lines = [
            "# H3 Hybrid HDR extraction report",
            "",
            md_header,
            "",
            f"- ref: `{ref_path}`",
            f"- fl: `{fl_path}`",
            f"- sing: `{sing_path}`",
            f"- rank: {rank}",
            f"- edge_margin: {edge_margin}",
            f"- FL projection: {'on' if not no_fl_projection else 'off'}",
            f"- blocks: {block_start}-{block_end}" if block_start is not None else "- blocks: all",
            f"- tensors emitted: {emitted_count}",
            f"- noise-only skipped: {len(noise_only)}",
            f"- {format_rank_recommendation(rows, int(rank))}",
            "",
            md_table,
        ]
        if skipped_1d:
            md_lines.extend(["", "## Skipped / diff-only 1-D tensors", ""])
            md_lines.extend(f"- `{k}`" for k in skipped_1d)
        if noise_only:
            md_lines.extend(["", "## Noise-only (no singular values above MP edge)", ""])
            md_lines.extend(f"- `{k}`" for k in noise_only)

        summary["markdown"] = "\n".join(md_lines)

        if report_stem:
            stem = Path(report_stem)
            stem.parent.mkdir(parents=True, exist_ok=True)
            json_path = stem.with_suffix(".json") if stem.suffix != ".json" else stem
            md_path = stem.with_suffix(".md") if stem.suffix != ".md" else Path(str(stem) + ".md")
            json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
            md_path.write_text(summary["markdown"], encoding="utf-8")

        return summary
    finally:
        ref_r.close()
        fl_r.close()
        sing_r.close()
