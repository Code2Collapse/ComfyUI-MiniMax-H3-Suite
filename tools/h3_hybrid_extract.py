#!/usr/bin/env python3
"""CLI for H3 hybrid HDR LoRA extraction."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mmx_utils.h3_hybrid import parse_block_range  # noqa: E402
from mmx_utils.h3_hybrid_extract import extract_hybrid_lora  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Extract H3 Hybrid HDR LoRA from Ref2VA/FL2VA/Singularity")
    p.add_argument("--ref", required=True, help="Ref2VA checkpoint (.safetensors or .gguf)")
    p.add_argument("--fl", required=True, help="FL2VA checkpoint")
    p.add_argument("--sing", required=True, help="Singularity checkpoint")
    p.add_argument("--out", required=True, help="Output LoRA .safetensors path")
    p.add_argument("--rank", type=int, default=64)
    p.add_argument("--no-fl-projection", action="store_true")
    p.add_argument("--blocks", default=None, help="Block range e.g. 25-49")
    p.add_argument("--edge-margin", type=float, default=1.1, help="MP noise edge multiplier when floor is known")
    p.add_argument("--device", default="cpu")
    p.add_argument("--report", default=None, help="Report stem (.json + .md written)")
    args = p.parse_args(argv)

    block_start = block_end = None
    if args.blocks:
        block_start, block_end = parse_block_range(args.blocks)

    summary = extract_hybrid_lora(
        args.ref,
        args.fl,
        args.sing,
        args.out,
        rank=args.rank,
        no_fl_projection=args.no_fl_projection,
        block_start=block_start,
        block_end=block_end,
        edge_margin=args.edge_margin,
        device=args.device,
        report_stem=args.report,
    )
    print(summary["markdown"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
