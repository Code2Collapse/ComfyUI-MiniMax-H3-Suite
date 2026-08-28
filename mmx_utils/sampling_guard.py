"""Graph-time sampling safety checks — P10 protected layers, P11 accelerator conflicts."""

from __future__ import annotations

from typing import Any

CACHE_KEY = "minimax_h3_block_cache_t8"
SPECTRUM_BINDING_KEY = "spectrum_h3_binding"
WRAPPER_BLOCK_CACHE = "minimax_h3_block_cache_t8"

PROTECTED_UNION_BLOCK_INDICES = frozenset({0, 10, 20, 30, 40})
FINAL_LAYER_PATH_FRAGMENTS = (
    "final_layer",
    "norm_out",
    "proj_out",
    "audio_proj_out",
)


def _dit_replacements(transformer_options: dict) -> dict:
    pr = transformer_options.get("patches_replace") or {}
    if not isinstance(pr, dict):
        return {}
    dit = pr.get("dit") or {}
    return dit if isinstance(dit, dict) else {}


def check_accelerator_conflicts(
    model_options: dict,
    *,
    total_blocks: int | None = None,
) -> None:
    """P11 — fail loud on incompatible accelerator combinations."""
    to = model_options.get("transformer_options") or {}
    if not isinstance(to, dict):
        to = {}

    if model_options.get(SPECTRUM_BINDING_KEY) is not None and CACHE_KEY in to:
        raise ValueError(
            "MiniMax H3: Spectrum binding and Block Cache T8 cannot run together — "
            "use only one acceleration path."
        )
    if "easycache" in to and CACHE_KEY in to:
        raise ValueError(
            "MiniMax H3: EasyCache/LazyCache cannot combine with Block Cache T8 — "
            "the audio stream needs dual-metric skip decisions."
        )
    if "easycache" in to and "lazycache" in to:
        raise ValueError("MiniMax H3: EasyCache and LazyCache cannot both be active.")

    dit = _dit_replacements(to)
    if not dit:
        return

    # Block cache legitimately occupies block 0 and last; anything else is a conflict.
    allowed_cache = set()
    if CACHE_KEY in to and total_blocks is not None and total_blocks >= 2:
        allowed_cache = {("double_block", 0), ("double_block", total_blocks - 1)}

    occupied: set[tuple] = set()
    for key in dit:
        if key in allowed_cache:
            continue
        if key in occupied:
            raise ValueError(
                f"MiniMax H3: duplicate patches_replace['dit'] key {key!r} — "
                "two nodes wrote the same block index."
            )
        occupied.add(key)

    if total_blocks is not None:
        for idx in range(total_blocks):
            key = ("double_block", idx)
            if key in dit and key not in allowed_cache:
                # Installing block cache is pre-checked in block_cache node; other replaces warn here
                pass


def check_dit_replace_conflicts_before_install(
    existing_dit: dict,
    requested_keys: list[tuple],
) -> None:
    """Refuse to overwrite an existing dit patch (P9 install guard)."""
    for key in requested_keys:
        if key in existing_dit:
            raise ValueError(
                f"MiniMax H3: cannot patch {key!r} — already occupied by another dit replace. "
                "Only one block-replace accelerator per branch."
            )


def check_protected_layers(
    model_options: dict,
    *,
    total_blocks: int,
) -> None:
    """P10 — ensure protected DiT blocks and FinalLayer paths are not hijacked."""
    if total_blocks < 1:
        raise ValueError("MiniMax H3 protected-layer check: total_blocks must be >= 1")

    to = model_options.get("transformer_options") or {}
    dit = _dit_replacements(to if isinstance(to, dict) else {})

    protected_indices = {0, total_blocks - 1} | {
        i for i in PROTECTED_UNION_BLOCK_INDICES if i < total_blocks
    }

    for key in dit:
        if not isinstance(key, tuple) or len(key) != 2:
            continue
        kind, idx = key
        if kind != "double_block":
            label = f"{kind}:{idx}"
            for frag in FINAL_LAYER_PATH_FRAGMENTS:
                if frag in str(label).lower():
                    raise ValueError(
                        f"MiniMax H3: protected path {label!r} must stay native — "
                        "remove the patch on FinalLayer / proj_out / audio_proj_out."
                    )
            continue
        if idx not in protected_indices:
            continue
        # Block cache is allowed on 0 and last when CACHE_KEY present
        if CACHE_KEY in (to if isinstance(to, dict) else {}):
            if idx in (0, total_blocks - 1):
                continue
        raise ValueError(
            f"MiniMax H3: protected DiT block index {idx} has a patches_replace entry {key!r}. "
            "Block 0, the last block, and Union injection blocks must not be skipped or replaced "
            "except by the official Block Cache T8 first/last hooks."
        )

    object_patches = model_options.get("object_patches") or model_options.get("model_patches") or {}
    if isinstance(object_patches, dict):
        for path in object_patches:
            low = str(path).lower()
            for frag in FINAL_LAYER_PATH_FRAGMENTS:
                if frag in low:
                    raise ValueError(
                        f"MiniMax H3: object_patch on protected path '{path}' — "
                        "FinalLayer / norm_out / proj_out / audio_proj_out must stay native."
                    )


def infer_total_blocks(model: Any) -> int:
    """Best-effort block count from a MODEL or diffusion_model."""
    dm = getattr(getattr(model, "model", None), "diffusion_model", None)
    if dm is None:
        blocks = getattr(model, "blocks", None)
        if blocks is not None:
            return len(blocks)
        raise ValueError("Cannot infer DiT block count from MODEL")
    blocks = getattr(dm, "blocks", None)
    if blocks is None:
        raise ValueError("MODEL diffusion_model has no blocks attribute")
    return len(blocks)
