# PORTED FROM: https://github.com/bbaudio-2025/Comfyui-MMH3-UltimateUpscale :: nodes/nodes.py _chunk_fun_control /
#   _tile_fun_control / _build_fun_net (MIT) - the idea (each tile and chunk gets only its own window of the control
#   video). Rewritten here: upstream drives a comfy.controlnet.MiniMaxH3ControlNet that no ComfyUI release has (checked
#   core 0.39 and v0.45.0, 2026-11-11); core applies the H3 Fun ControlNet as a MODEL PATCH
#   (comfy_extras/nodes_minimax_h3.py MiniMaxH3FunControlPatch + MiniMaxH3FunControlBlockPatch), so the patch is
#   rebuilt per tile on a model clone instead.
"""Per-tile Fun ControlNet for H3 Split Upscale (L9.197).

Core's Apply MiniMax H3 Fun ControlNet stores the WHOLE control video on a patch and, at sampling time, fits it to the
size of whatever latent is being sampled (center crop to that aspect, bilinear resize). Sampled tile by tile, every
tile would get the whole frame squeezed into it. Here each tile gets a clone of the model whose Fun control patches
carry only the pixels core's whole-frame fit would have put under that tile, and only that chunk's frames.

A one-tile, one-chunk run gives exactly the control core would have built (tested against core's own _fit_frames).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

_CP_ATTRS = ("model_patch", "vae", "control_video", "mask", "source_video", "strength", "sigma_start", "sigma_end",
             "diffusion_model_wrapper", "register")


def _center_crop_box(src_h: int, src_w: int, dst_h: int, dst_w: int) -> tuple[int, int, int, int]:
    """(x, y, width, height) of the source region core's common_upscale(..., crop="center") keeps."""
    old_aspect, new_aspect = src_w / src_h, dst_w / dst_h
    x = y = 0
    if old_aspect > new_aspect:
        x = round((src_w - src_w * (new_aspect / old_aspect)) / 2)
    elif old_aspect < new_aspect:
        y = round((src_h - src_h * (old_aspect / new_aspect)) / 2)
    return x, y, src_w - 2 * x, src_h - 2 * y


def frames_window(frames: torch.Tensor, f0: int, f1: int) -> torch.Tensor:
    """The frames core would use for a chunk that starts at pixel frame f0: frame f0 + i, the last frame repeating
    past the end (core clamps the index the same way)."""
    t = frames.shape[0]
    if t <= 1:
        return frames
    a = min(int(f0), t - 1)
    b = max(a + 1, min(int(f1), t))
    return frames[a:b]


def control_window(frames: torch.Tensor, *, full_hw: tuple[int, int], box: tuple[int, int, int, int],
                   padded_hw: tuple[int, int], comp: int) -> torch.Tensor:
    """frames [T, C, H, W] at any size -> [T, C, padded_h * comp, padded_w * comp].

    full_hw = the whole latent (h, w); box = the tile (r0, c0, tr, tc) in latent cells; padded_hw = the size the tile is
    sampled at (pad_even). The result is the part of core's whole-frame fit under the tile; the pad strip repeats the
    edge, like the latent pad.
    """
    lat_h, lat_w = int(full_hw[0]), int(full_hw[1])
    r0, c0, tr, tc = (int(v) for v in box)
    src_h, src_w = frames.shape[-2:]
    x, y, cw, ch = _center_crop_box(src_h, src_w, lat_h * comp, lat_w * comp)
    sx0, sx1 = x + round(c0 * cw / lat_w), x + round((c0 + tc) * cw / lat_w)
    sy0, sy1 = y + round(r0 * ch / lat_h), y + round((r0 + tr) * ch / lat_h)
    crop = frames[..., sy0:max(sy1, sy0 + 1), sx0:max(sx1, sx0 + 1)]
    out = F.interpolate(crop, size=(tr * comp, tc * comp), mode="bilinear")   # = common_upscale "bilinear"
    ph, pw = (int(padded_hw[0]) - tr) * comp, (int(padded_hw[1]) - tc) * comp
    if ph > 0 or pw > 0:
        out = F.pad(out, (0, max(pw, 0), 0, max(ph, 0)), mode="replicate")
    return out


def _is_block(p) -> bool:
    return p is not None and all(hasattr(p, a) for a in ("control_patch", "block_index", "previous"))


def _hides_fun(p, depth: int = 0) -> bool:
    """A foreign block patch whose `previous` chain holds a Fun control block: cannot be rebuilt from outside."""
    while p is not None and depth < 64:
        if _is_block(p):
            return True
        p, depth = getattr(p, "previous", None), depth + 1
    return False


class FunControlNotSplittable(ValueError):
    """A control patch this module cannot rebuild per tile; the message says which and what to do."""


def fun_control_patches(model) -> list:
    """The distinct core Fun control patches on `model`, in order ([] = none).

    Raises FunControlNotSplittable for a patch that cannot be rebuilt from outside: a Fun block under another block
    patch, or a control patch that is not exactly core's MiniMaxH3FunControlPatch - a subclass such as the Suite's
    H3 Mask-Aware ControlNet carries more state (region, crop boxes, schedule) than its constructor arguments, and a
    rebuild would silently drop it."""
    try:
        reps = model.model_options.get("transformer_options", {}).get("patches_replace", {}).get("dit", {})
    except AttributeError:
        return []
    found: list = []
    for p in reps.values():
        if not _is_block(p):
            if _hides_fun(getattr(p, "previous", None)):
                raise FunControlNotSplittable(
                    f"a Fun ControlNet sits under another block patch ({type(p).__name__}); apply the Fun ControlNet "
                    "last, after the other model patches")
            continue
        while _is_block(p):
            cp = p.control_patch
            if type(cp).__name__ != "MiniMaxH3FunControlPatch" or not all(hasattr(cp, a) for a in _CP_ATTRS):
                raise FunControlNotSplittable(
                    f"'{type(cp).__name__}' cannot be split into tiles / chunks (its state is built for the whole "
                    "frame); use core's 'Apply MiniMax H3 Fun ControlNet' for a split run")
            if all(cp is not f for f in found):
                found.append(cp)
            p = p.previous
        if _hides_fun(getattr(p, "previous", None)):
            raise FunControlNotSplittable("a Fun ControlNet sits under another block patch; apply it last")
    return found


def _rebuild(p, new: dict):
    if not _is_block(p):
        return p
    return type(p)(new[id(p.control_patch)], p.block_index, _rebuild(p.previous, new))


def retile_model(model, patches: list, window_fn):
    """A clone of `model` whose Fun control patches carry window_fn(patch) = (control_video, mask, source_video)
    instead of the whole clip. The original model and its patches are not touched."""
    new = {}
    for cp in patches:
        cv, mask, src = window_fn(cp)
        new[id(cp)] = type(cp)(cp.model_patch, cp.vae, cv, mask, src, cp.strength, cp.sigma_start, cp.sigma_end)
    clone = model.clone()
    # the wrapper that prepares each patch's control latent: swap the old patch's bound method for the new one in
    # place (same wrapper type, key and order), so the whole clip is never encoded for a tile
    for keyed in clone.wrappers.values():
        for k, lst in keyed.items():
            keyed[k] = [getattr(new[id(w.__self__)], w.__func__.__name__)
                        if id(getattr(w, "__self__", None)) in new else w for w in lst]
    dit = clone.model_options.get("transformer_options", {}).get("patches_replace", {}).get("dit", {})
    for key, p in list(dit.items()):
        dit[key] = _rebuild(p, new)
    return clone


def tile_window_fn(*, f0: int, f1: int, full_hw, box, padded_hw):
    """window_fn for retile_model: the chunk's frames and the tile's pixels of control video, mask and source."""
    def fn(cp):
        comp = int(cp.vae.spacial_compression_encode())

        def sp(x):
            return control_window(frames_window(x, f0, f1), full_hw=full_hw, box=box, padded_hw=padded_hw, comp=comp)

        cv = sp(cp.control_video) if cp.control_video is not None else None
        mask = src = None
        if cp.mask is not None:
            m = cp.mask.reshape(-1, 1, cp.mask.shape[-2], cp.mask.shape[-1]).to(torch.float32)
            mask = sp(m)[:, 0]
            src = sp(cp.source_video) if cp.source_video is not None else None
        return cv, mask, src
    return fn
