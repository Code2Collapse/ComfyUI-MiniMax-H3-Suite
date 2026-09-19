"""Build the H3 Fun ControlNet-Union's 49-channel hint correctly.

THE LAYOUT, from alibaba-pai/MiniMax-H3-Fun-Controlnet-Union's own model card:

    "the control input is widened to control_in_dim = 49
     (latent + masked latent + mask channels)"

which is 24 (control latent) + 1 (visibility) + 24 (masked latent), and that is
exactly what comfy/ldm/minimax/controlnet.py builds a 196-column patch from
(49 x 1 x 2 x 2). The card names two SEPARATE modes: structural control from
canny/depth/hed/mlsd/pose, and inpainting, which upstream runs through its own
`predict_v2v_control_inpaint.py`.

WHAT GOES WRONG IN COMFYUI, and it is the thing that makes a control-only graph
behave oddly. comfy_extras/nodes_minimax_h3.py's prepare_control_latent only
builds the visibility and masked-latent channels `if self.mask is not None`.
With no mask the hint stays 24 channels, and init_stream zero-PADS it to 196:

    target_rows = patchify_video(control_latent, self.patch_size)
    if target_rows.shape[1] < patch_dim:
        target_rows = torch.nn.functional.pad(target_rows, (0, patch_dim - target_rows.shape[1]))

Zero padding is not neutral here. Visibility is built as `1.0 - mask`, so

    visibility 1 = keep this pixel      visibility 0 = this is a hole

and a zero-filled visibility channel tells the ControlNet that THE ENTIRE FRAME
IS A HOLE, with an all-black masked latent as its context. That is the opposite
of "no inpainting requested". Every control-video-only graph is running the
Union model in a whole-frame-inpaint state it was never asked for, which is why
those graphs need the strength turned down to 0.15-0.6 before they behave.

THE OTHER TRAP. `mask` here is a PIXEL-space mask - it is upscaled to
width x height. The mask that feeds SetLatentNoiseMask is a LATENT-space mask
(MVEx_MaskToLatentSpace, or NKDMaskOps' latent_mask output). Same MASK socket,
two different spaces. Wiring the latent one in looks reasonable and is
destructive: 37 latent rows against a 124-frame clip, indexed by

    indices = torch.arange(frame_count).clamp(max=mask.shape[0] - 1)

so frames 37 to 123 all repeat row 36. The mask smears across the whole tail of
the shot.

This module is the arithmetic, deliberately free of any ComfyUI import so it
can be tested on CPU with no VAE, no weights and no H3 build.
"""

from __future__ import annotations

import torch

__all__ = [
    "CONTROL_IN_DIM",
    "LATENT_CHANNELS",
    "frame_indices",
    "looks_like_a_latent_mask",
    "visibility_from_mask",
    "assemble_hint",
    "describe_hint",
]

#: From the model card and comfy/ldm/minimax/controlnet.py's default.
CONTROL_IN_DIM = 49
#: H3's video latent width. 24 + 1 + 24 = 49.
LATENT_CHANNELS = 24


def frame_indices(available: int, wanted: int) -> torch.Tensor:
    """Map `wanted` output frames onto `available` input frames.

    ComfyUI clamps here, which repeats the last frame forever once the input
    runs out. For a mask that genuinely is shorter than the clip - and for the
    latent-space mask people wire in by mistake - that smears the last frame
    across the rest of the shot. An even resample keeps the mask's timing.

    When the counts already match this is the identity, so nothing moves in the
    normal case.
    """
    if available < 1:
        raise ValueError("cannot index into an empty sequence")
    if available == wanted:
        return torch.arange(wanted)
    if available > wanted:
        # More material than needed: take an even spread rather than the head,
        # so a mask authored for the whole clip still lines up.
        return torch.linspace(0, available - 1, wanted).round().long()
    return torch.linspace(0, available - 1, wanted).round().long()


def looks_like_a_latent_mask(mask_hw: tuple[int, int],
                             latent_hw: tuple[int, int],
                             pixel_hw: tuple[int, int]) -> bool:
    """True when a mask is at LATENT resolution and was meant for the sampler.

    The two are far enough apart to tell them apart reliably: a latent mask is
    the latent's own height and width (or that halved again, for a mask already
    reduced to H3's 2x2 token grid), while a pixel mask is 8 or 16 times larger.
    Being wrong in the safe direction matters more than being clever, so this
    only fires on an exact match to one of the two latent shapes.
    """
    lh, lw = latent_hw
    ph, pw = pixel_hw
    if (ph, pw) == (lh, lw):
        return False                      # degenerate; cannot distinguish
    return mask_hw in {(lh, lw), (lh // 2, lw // 2)}


def visibility_from_mask(
    mask: torch.Tensor | None,
    frame_count: int,
    *,
    inpaint: bool,
) -> torch.Tensor | None:
    """[F,1,H,W] visibility: 1 keeps a pixel, 0 marks it a hole.

    Returns None when there is nothing to build from, so the caller can decide
    what a missing mask means - which is the whole point, because ComfyUI's
    answer to that question (zeros, i.e. everything is a hole) is wrong.
    """
    if not inpaint:
        return None
    if mask is None:
        return None
    m = mask.reshape(-1, 1, mask.shape[-2], mask.shape[-1]).to(torch.float32)
    idx = frame_indices(m.shape[0], frame_count).to(m.device)
    m = m[idx]
    return 1.0 - (m > 0.5).to(torch.float32)


def assemble_hint(
    control_latent: torch.Tensor | None,
    visibility_latent: torch.Tensor | None,
    masked_latent: torch.Tensor | None,
    *,
    latent_shape: tuple[int, ...],
    device=None,
    dtype=torch.float32,
) -> torch.Tensor:
    """The full 49-channel hint. Never short, never zero-padded by accident.

    Any part the caller did not supply is filled with its NEUTRAL value, not
    with zeros:

      control latent   zeros  - no structural opinion, which genuinely is zero
      visibility       ONES   - nothing is a hole. This is the one ComfyUI gets
                               wrong, and it is the difference between "no
                               inpainting requested" and "inpaint everything".
      masked latent    zeros  - only ever paired with visibility 1, where the
                               model has no hole to fill and no context to want
    """
    b = latent_shape[0]
    t, h, w = latent_shape[2:]
    if control_latent is None:
        control_latent = torch.zeros(b, LATENT_CHANNELS, t, h, w,
                                     device=device, dtype=dtype)
    if visibility_latent is None:
        visibility_latent = torch.ones(b, 1, t, h, w, device=device, dtype=dtype)
    if masked_latent is None:
        masked_latent = torch.zeros(b, LATENT_CHANNELS, t, h, w,
                                    device=device, dtype=dtype)

    hint = torch.cat([
        control_latent.to(device=device, dtype=dtype),
        visibility_latent.to(device=device, dtype=dtype),
        masked_latent.to(device=device, dtype=dtype),
    ], dim=1)
    if hint.shape[1] != CONTROL_IN_DIM:
        raise ValueError(
            f"the control hint came to {hint.shape[1]} channels, not "
            f"{CONTROL_IN_DIM}. The ControlNet-Union expects "
            f"{LATENT_CHANNELS} control + 1 visibility + {LATENT_CHANNELS} "
            "masked latent; a short hint gets zero-padded by the model patch, "
            "and a zero visibility channel means 'the whole frame is a hole'."
        )
    return hint


def describe_hint(has_control: bool, inpainting: bool,
                  frames: int, latent_frames: int) -> str:
    """One plain paragraph about what was actually handed to the ControlNet."""
    lines = []
    if has_control and inpainting:
        lines.append(
            "Structural control AND inpainting: all 49 channels carry real "
            "data - the hint, the visibility mask and the masked plate.")
    elif has_control:
        lines.append(
            "Structural control only. Visibility is filled with ONES, meaning "
            "nothing is a hole. ComfyUI's own node leaves those channels at "
            "zero here, which tells the model the entire frame needs "
            "inpainting - the reason a control-only graph needs its strength "
            "turned down before it behaves.")
    elif inpainting:
        lines.append(
            "Inpainting only, no structural hint: the control channels are "
            "zero, which is genuinely neutral.")
    else:
        lines.append(
            "Neither a control video nor a mask - the ControlNet has nothing "
            "to say and will contribute nothing.")
    lines.append(f"{frames} frames -> {latent_frames} latent rows.")
    return " ".join(lines)
