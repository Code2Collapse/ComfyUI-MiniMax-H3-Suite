"""Make H3's ControlNet residual agree with the latent mask instead of fighting it.

THE DEFECT, with the lines it lives on.

ComfyUI's H3 Fun ControlNet adds its residual to every video row:

    comfy_extras/nodes_minimax_h3.py, MiniMaxH3FunControlPatch.after_block
        skip[args["layout"].audio_pos.to(skip.device)] = 0
        out["img"].add_(skip, alpha=self.strength)

Audio rows are zeroed. Nothing else is.

Latent masking, meanwhile, does not zero anything. It converts the denoise mask
into a PER-ROW TIMESTEP:

    comfy/ldm/minimax/model.py
        m = mask_row_values(denoise_mask[0, 0].to(torch.float32), latent_t, lat_h, lat_w)
        rows_t = (1.0 - m * sigma_v.to(m.device)).clamp(max=t_pin_v)

so a masked-out row is told it is already near-clean and should hold still.

Put those together and two forces act on the same rows in opposite directions:
the mask says "this row is finished", the ControlNet says "this row, move
toward my hint". Nothing reconciles them, and the result is the contradiction a
user sees as smeared or fighting detail inside the region they asked to
preserve.

The asymmetry is sharper than it first looks. The control INPUT already knows
about the mask - nodes_minimax_h3.py concatenates a visibility channel and a
masked latent into the hint before encoding. It is only the residual
APPLICATION that is mask-blind.

THE FIX. Scale the residual per row by the same mask the sampler used, so the
control steers only what is actually being generated. That makes the flow
uni-directional: mask decides where, control decides what.

This module is the arithmetic, deliberately free of any ComfyUI import so it
can be tested on CPU with no model, no weights and no H3 build. The node layer
in mmx_nodes/mask_aware_control.py is the part that touches ComfyUI.
"""

from __future__ import annotations

import torch

__all__ = [
    "row_values_from_mask",
    "soften_rows",
    "control_gate",
    "describe_gate",
]


def row_values_from_mask(
    mask: torch.Tensor,
    latent_t: int,
    lat_h: int,
    lat_w: int,
) -> torch.Tensor:
    """[T,H,W] denoise mask (1 = generate) -> one float per 2x2 patch row.

    This mirrors `mask_row_values` in comfy/ldm/minimax/model.py, which is what
    the sampler itself uses to build the per-row timesteps. The two MUST agree:
    a gate computed on a different row order would mute the control in the wrong
    places, which is a worse bug than the one being fixed because it looks like
    the control is simply weak.

    The one deliberate difference: comfy's returns None when every row fully
    generates. Returning the all-ones vector instead keeps the caller branchless.
    """
    if mask.ndim != 3:
        raise ValueError(
            f"denoise mask must be [T,H,W]; got shape {tuple(mask.shape)}"
        )
    m = mask.to(torch.float32)
    pad_w = lat_w - m.shape[-1]
    pad_h = lat_h - m.shape[-2]
    if pad_w < 0 or pad_h < 0:
        raise ValueError(
            f"denoise mask is {tuple(m.shape[-2:])} but the latent is "
            f"{(lat_h, lat_w)}; a mask larger than the latent cannot be placed"
        )
    if pad_w or pad_h:
        m = torch.nn.functional.pad(m, (0, pad_w, 0, pad_h), mode="replicate")
    # amax over each 2x2 patch: a patch that contains ANY generated pixel counts
    # as generated, matching the sampler. min() here would freeze patches that
    # straddle the mask edge and leave a hard seam.
    m = m.reshape(latent_t, lat_h // 2, 2, lat_w // 2, 2).amax(dim=(2, 4))
    return m.reshape(-1)


def soften_rows(
    rows: torch.Tensor,
    latent_t: int,
    lat_h: int,
    lat_w: int,
    softness: float,
) -> torch.Tensor:
    """Blur the gate spatially so the control does not stop at a hard edge.

    A binary gate puts a 1-patch cliff between "fully controlled" and "not
    controlled at all", and that cliff is visible in the output as a seam. The
    blur is 2-D per frame: blurring across time would let one frame's mask leak
    into the next and smear a moving boundary.
    """
    if softness <= 0:
        return rows
    gh, gw = lat_h // 2, lat_w // 2
    # one frame per batch entry: the blur is 2-D within a frame, never across
    grid = rows.reshape(latent_t, 1, gh, gw)

    radius = max(1, int(round(softness)))
    size = radius * 2 + 1
    # A box pass per axis: cost is linear in the radius, and a gate does not
    # need a gaussian's exactness.
    for dim in (2, 3):
        length = grid.shape[dim]
        pad = min(radius, max(0, length - 1))
        if pad == 0:
            continue
        weight_shape = (1, 1, size, 1) if dim == 2 else (1, 1, 1, size)
        weight = torch.full(weight_shape, 1.0 / size,
                            device=grid.device, dtype=grid.dtype)
        pad_spec = (0, 0, pad, pad) if dim == 2 else (pad, pad, 0, 0)
        padded = torch.nn.functional.pad(grid, pad_spec, mode="replicate")
        if pad < radius:
            extra = radius - pad
            extra_spec = (0, 0, extra, extra) if dim == 2 else (extra, extra, 0, 0)
            padded = torch.nn.functional.pad(padded, extra_spec, mode="replicate")
        grid = torch.nn.functional.conv2d(padded, weight)
    return grid.reshape(-1)


def control_gate(
    denoise_mask: torch.Tensor | None,
    img_update: torch.Tensor,
    latent_t: int,
    lat_h: int,
    lat_w: int,
    *,
    preserved_strength: float = 0.0,
    boundary_softness: float = 0.0,
) -> torch.Tensor:
    """Per-row multiplier for the control residual, over the layout's image rows.

    `img_update` is `layout.img_update`: True for the rows being generated,
    False for the conditioning rows (keyframes, reference images) that sit in
    the same image span. Those conditioning rows are not being denoised at all,
    so a control residual on them is pure contradiction - they get 0 regardless
    of the mask, which is also what `init_stream` does with the control INPUT.

    `preserved_strength` is the deliberate escape hatch: 0 means the control
    never touches a preserved row, 1 restores ComfyUI's current behaviour of
    ignoring the mask entirely. Anything between is a partial hold, for the case
    where a user wants the control to inform a preserved region without
    overriding it.
    """
    n = int(img_update.shape[0])
    gate = torch.zeros(n, dtype=torch.float32, device=img_update.device)
    update = img_update.to(torch.bool)
    generated = int(update.sum())

    if denoise_mask is None:
        # No latent mask: every generated row is fully controlled, which is the
        # core behaviour and the right answer when there is nothing to conflict.
        gate[update] = 1.0
        return gate

    rows = row_values_from_mask(denoise_mask, latent_t, lat_h, lat_w).to(gate.device)
    if rows.shape[0] != generated:
        raise ValueError(
            f"the denoise mask produced {rows.shape[0]} latent rows but the "
            f"layout has {generated} generated image rows. The mask and the "
            "latent describe different videos - check that the mask's frame "
            "count and resolution match the latent being sampled."
        )
    rows = soften_rows(rows, latent_t, lat_h, lat_w, boundary_softness)
    keep = float(min(max(preserved_strength, 0.0), 1.0))
    # rows == 1 -> fully generated -> full control
    # rows == 0 -> fully preserved -> `keep`, which defaults to nothing
    gate[update] = keep + (1.0 - keep) * rows.clamp(0.0, 1.0)
    return gate


def describe_gate(gate: torch.Tensor, img_update: torch.Tensor) -> str:
    """One plain sentence about what the gate will do, for the node's report."""
    update = img_update.to(torch.bool)
    total = int(update.sum())
    if total == 0:
        return "no generated image rows: the control has nothing to steer."
    active = gate[update]
    full = int((active >= 0.999).sum())
    none = int((active <= 0.001).sum())
    partial = total - full - none
    if none == 0 and partial == 0:
        return (f"all {total} generated rows are fully controlled "
                "(no latent mask, or the mask generates everywhere).")
    return (
        f"{full} of {total} rows fully controlled, {none} held back by the "
        f"latent mask, {partial} on the soft boundary. Those held rows are the "
        "ones the mask asked to preserve; without this the control would be "
        "pushing them while the sampler holds them still."
    )
