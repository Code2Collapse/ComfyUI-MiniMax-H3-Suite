"""H3 hybrid LoRA extraction — stream three matched checkpoints, FL-project, SVD compress.

A CHECKPOINT-DELTA EXPERIMENT (A26 3.1 goal C / 3.6): with D = FL2VA - Ref2VA and Delta = Singularity - Ref2VA,
a = clamp(<Delta, D> / <D, D>, 0, 1) and H = Delta - a D. H has not been shown to carry only HDR information; it can
hold any other difference between the checkpoints (training, pruning). It is never evidence that H3 learned HDR.

Memory (measured, reported): source weights are STREAMED one tensor at a time, but every emitted LoRA factor is kept
in RAM until the file is written, so peak RAM grows with rank x layers. The report records the retained factor bytes
and the process peak RSS.

Safeguards (A26 3.6): the three checkpoints must be the same H3 variant (block count, hidden size, pruned / full,
compressed attention); keys outside every DiT block ("global parameters") are an explicit choice; every factor is
checked finite before the file is published; the file is written to a temporary name and renamed (atomic); a sidecar
report records source identities, coverage, rank, block range, per-layer SVD metrics and the extractor version.
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
import platform
from pathlib import Path
from typing import Callable

EXTRACTOR_VERSION = 2
EXPERIMENT_LABEL = ("checkpoint-delta experiment: H = (Singularity - Ref2VA) - a (FL2VA - Ref2VA); not proven to be "
                    "HDR-specific (may contain other training / pruning differences)")

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


class VariantMismatch(ValueError):
    """The three checkpoints (or a LoRA and a model) are different H3 variants."""


def file_identity(path: str, full: bool = False) -> dict:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        if full:
            for chunk in iter(lambda: f.read(1 << 24), b""):
                h.update(chunk)
        else:
            h.update(f.read(1 << 20))
    return {"file": os.path.basename(str(path)), "bytes": os.path.getsize(path),
            ("sha256" if full else "sha256_first_mib"): h.hexdigest()}


def variant_fingerprint(reader) -> dict:
    """The structure that must match for a delta (and its LoRA) to mean anything: same markers core 0.39's
    model_detection uses (blocks counted from keys, hidden size from video_patch_proj, adaln_t_table = pruned)."""
    keys = reader.tensor_keys()
    blocks = {parse_block_index(k) for k in keys}
    blocks.discard(None)
    hidden = None
    for cand in ("diffusion_model.video_patch_proj.weight", "diffusion_model.blocks.0.attn.out_proj.weight"):
        if cand in keys:
            hidden = int(reader.shape(cand)[0])
            break
    return {"n_blocks": (max(blocks) + 1) if blocks else 0, "hidden_size": hidden,
            "pruned_adaln_curves": "diffusion_model.adaln_t_table" in keys,
            "gate_compress": any(k.endswith(".attn.to_gate_compress.weight") for k in keys),
            "diffusion_weight_keys": sum(1 for k in keys if is_diffusion_weight_key(k))}


_STRUCTURE = ("n_blocks", "hidden_size", "pruned_adaln_curves", "gate_compress")


def check_same_variant(fps: dict[str, dict]) -> None:
    names = list(fps)
    for k in _STRUCTURE:
        vals = {n: fps[n][k] for n in names}
        if len(set(vals.values())) > 1:
            raise VariantMismatch(f"The checkpoints are different H3 variants ({k}: " +
                                  ", ".join(f"{n}={v}" for n, v in vals.items()) +
                                  "). Use Ref2VA, FL2VA and Singularity of the SAME variant (pruned with pruned, "
                                  "full with full).")


def _rss() -> int | None:
    try:
        import psutil
        return int(psutil.Process().memory_info().rss)
    except Exception:  # noqa: BLE001 - psutil is optional; the retained-bytes figure is still exact
        return None


def _publish(lora_out: dict, out_path: Path, metadata: dict) -> None:
    """Validate every tensor (present, finite) and write atomically: temp file + os.replace."""
    import torch
    from safetensors.torch import save_file as _save
    if not lora_out:
        raise ValueError("Nothing to write: no layer had a residual above the noise edge in this block range.")
    bad = [k for k, t in lora_out.items() if not bool(torch.isfinite(t.float()).all())]
    if bad:
        raise ValueError(f"{len(bad)} extracted tensors contain NaN / inf (first: {bad[0]}); nothing was written.")
    tmp = out_path.with_name(out_path.name + f".tmp-{os.getpid()}")
    try:
        _save(lora_out, str(tmp), metadata={"c2c_hybrid": json.dumps(metadata, sort_keys=True)})
        os.replace(tmp, out_path)
    finally:
        if tmp.exists():
            tmp.unlink()


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
    include_global_parameters: bool = True,
    full_hash: bool = False,
) -> dict:
    """
    Extract rank-r LoRA of HDR residual H from Ref2VA/FL2VA/Singularity triple.

    Streams one 2-D weight at a time. Returns summary dict with markdown report text.
    """
    ref_r = open_checkpoint(ref_path, device=device)
    fl_r = open_checkpoint(fl_path, device=device)
    sing_r = open_checkpoint(sing_path, device=device)

    try:
        fps = {"ref": variant_fingerprint(ref_r), "fl": variant_fingerprint(fl_r), "sing": variant_fingerprint(sing_r)}
        check_same_variant(fps)
        rss0 = _rss()
        peak_rss = rss0
        retained_bytes = 0
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

        n_paired = len(paired)
        n_global = sum(1 for k in paired if parse_block_index(k) is None)
        if block_start is not None and block_end is not None:
            in_range = [k for k in paired if parse_block_index(k) is not None
                        and key_in_block_range(k, block_start, block_end)]
            glob = [k for k in paired if parse_block_index(k) is None]
            paired = in_range + (glob if include_global_parameters else [])
        else:
            in_range = [k for k in paired if parse_block_index(k) is not None]
            paired = in_range + ([k for k in paired if parse_block_index(k) is None] if include_global_parameters
                                 else [])
        coverage = {"common_weight_keys": n_paired, "matched_block_parameters": len(in_range),
                    "global_parameters": n_global, "global_included": bool(include_global_parameters),
                    "skipped_out_of_range": n_paired - n_global - len(in_range),
                    "key_overlap": {n: (n_paired / fps[n]["diffusion_weight_keys"])
                                    if fps[n]["diffusion_weight_keys"] else 0.0 for n in fps}}

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
            retained_bytes = sum(t.numel() * t.element_size() for t in lora_out.values())
            r = _rss()
            if r is not None and (peak_rss is None or r > peak_rss):
                peak_rss = r

            if progress_cb is not None:
                progress_cb()

        out_path_p = Path(out_path)
        out_path_p.parent.mkdir(parents=True, exist_ok=True)
        import torch as _torch
        sources = {"ref": file_identity(ref_path, full_hash), "fl": file_identity(fl_path, full_hash),
                   "sing": file_identity(sing_path, full_hash)}
        meta = {"extractor_version": EXTRACTOR_VERSION, "experiment": EXPERIMENT_LABEL, "variant": fps["ref"],
                "sources": sources, "rank": int(rank), "block_start": block_start, "block_end": block_end,
                "include_global_parameters": bool(include_global_parameters), "fl_projection": not no_fl_projection}
        if not lora_out:                                   # nothing to publish: keep the explanation, not an empty file
            sidecar = out_path_p.with_name(out_path_p.stem + ".report.json")
            sidecar.write_text(json.dumps({"extractor_version": EXTRACTOR_VERSION, "experiment": EXPERIMENT_LABEL,
                                           "sources": sources, "variant": fps, "coverage": coverage,
                                           "noise_only": noise_only, "per_tensor": rows, "written": False},
                                          indent=2, default=str), encoding="utf-8")
            raise ValueError("Nothing to write: no layer had a residual above the noise edge in this block range "
                             f"({len(noise_only)} noise-only). Report: {sidecar}")
        _publish(lora_out, out_path_p, meta)

        block_agg = aggregate_block_report(rows)
        md_table = format_block_markdown(block_agg)
        md_header = format_decomposition_header(block_agg)
        summary = {
            "extractor_version": EXTRACTOR_VERSION,
            "experiment": EXPERIMENT_LABEL,
            "software": {"torch": _torch.__version__, "python": platform.python_version()},
            "sources": sources,
            "variant": fps,
            "coverage": coverage,
            "memory": {"retained_factor_bytes": retained_bytes, "peak_rss_bytes": peak_rss,
                       "start_rss_bytes": rss0,
                       "note": "sources are streamed one tensor at a time; emitted factors stay in RAM until the "
                               "file is written (not memory-constant)"},
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
            f"- {EXPERIMENT_LABEL}",
            f"- variant: {fps['ref']['n_blocks']} blocks, hidden {fps['ref']['hidden_size']}, "
            f"{'pruned' if fps['ref']['pruned_adaln_curves'] else 'full'}",
            f"- coverage: {coverage['matched_block_parameters']} block weights, {coverage['global_parameters']} global "
            f"({'included' if include_global_parameters else 'excluded'}), {coverage['skipped_out_of_range']} skipped",
            f"- memory: factors retained {retained_bytes / 2**20:.1f} MiB, peak RSS "
            + (f"{peak_rss / 2**20:.0f} MiB" if peak_rss else "not measured (psutil missing)"),
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

        sidecar = out_path_p.with_name(out_path_p.stem + ".report.json")
        sidecar.write_text(json.dumps({k: v for k, v in summary.items() if k != "markdown"}, indent=2, default=str),
                           encoding="utf-8")
        summary["sidecar"] = str(sidecar)
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
