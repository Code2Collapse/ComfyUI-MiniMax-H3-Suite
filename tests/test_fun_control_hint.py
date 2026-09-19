"""The ControlNet-Union's 49 channels must all mean something.

The layout is stated by alibaba-pai's own model card - "control_in_dim = 49
(latent + masked latent + mask channels)" - and matched by
comfy/ldm/minimax/controlnet.py, which patchifies 49 x 1 x 2 x 2 = 196 columns.

The defect these tests exist for: ComfyUI builds the visibility and
masked-latent channels only when a mask is connected, and `init_stream`
zero-pads the short hint otherwise. Visibility is `1 - mask`, so zero means
"this is a hole" - a control-video-only graph therefore asks the Union model to
inpaint the entire frame. Both of the user's production workflows run exactly
that way, with mask and source_video left unconnected.

CPU-only, weight-free.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

PACK = Path(__file__).resolve().parents[1]
if str(PACK) not in sys.path:
    sys.path.insert(0, str(PACK))

from mmx_utils.fun_control_hint import (  # noqa: E402
    CONTROL_IN_DIM,
    LATENT_CHANNELS,
    assemble_hint,
    describe_hint,
    frame_indices,
    looks_like_a_latent_mask,
    visibility_from_mask,
)

SHAPE = (1, LATENT_CHANNELS, 7, 8, 12)      # [B,C,T,H,W]


# ── the layout ──────────────────────────────────────────────────────────────

def test_the_channel_budget_matches_the_model_card():
    # 24 + 1 + 24 = 49. If this ever drifts, every hint is silently misread.
    assert LATENT_CHANNELS * 2 + 1 == CONTROL_IN_DIM == 49


def test_the_hint_is_always_full_width():
    for control in (None, torch.zeros(1, LATENT_CHANNELS, 7, 8, 12)):
        for vis in (None, torch.zeros(1, 1, 7, 8, 12)):
            hint = assemble_hint(control, vis, None, latent_shape=SHAPE)
            assert hint.shape[1] == CONTROL_IN_DIM


def test_a_missing_visibility_channel_is_ONES_not_zeros():
    # THE bug, stated as an assertion. Visibility 0 means "hole", so the
    # neutral value for "no inpainting requested" is 1.
    hint = assemble_hint(None, None, None, latent_shape=SHAPE)
    visibility = hint[:, LATENT_CHANNELS:LATENT_CHANNELS + 1]
    assert visibility.min().item() == 1.0, (
        "a zero visibility channel tells the ControlNet the whole frame is a "
        "hole - which is what ComfyUI's zero padding does"
    )


def test_the_control_and_masked_channels_default_to_zero():
    # Those two ARE genuinely neutral at zero: no structural opinion, and no
    # context to offer for a hole that does not exist.
    hint = assemble_hint(None, None, None, latent_shape=SHAPE)
    assert hint[:, :LATENT_CHANNELS].abs().max().item() == 0.0
    assert hint[:, LATENT_CHANNELS + 1:].abs().max().item() == 0.0


def test_supplied_channels_land_in_the_right_slots():
    control = torch.full((1, LATENT_CHANNELS, 7, 8, 12), 0.25)
    vis = torch.full((1, 1, 7, 8, 12), 0.5)
    masked = torch.full((1, LATENT_CHANNELS, 7, 8, 12), 0.75)
    hint = assemble_hint(control, vis, masked, latent_shape=SHAPE)
    assert hint[:, :LATENT_CHANNELS].unique().tolist() == [0.25]
    assert hint[:, LATENT_CHANNELS].unique().tolist() == [0.5]
    assert hint[:, LATENT_CHANNELS + 1:].unique().tolist() == [0.75]


def test_a_wrong_width_hint_is_refused_by_name():
    bad = torch.zeros(1, LATENT_CHANNELS, 7, 8, 12)
    with pytest.raises(ValueError, match="49|channels"):
        assemble_hint(bad, torch.zeros(1, 2, 7, 8, 12), bad, latent_shape=SHAPE)


# ── visibility ──────────────────────────────────────────────────────────────

def test_visibility_inverts_the_mask():
    mask = torch.zeros(4, 8, 12)
    mask[:, :4, :] = 1.0                      # top half is the region to fill
    vis = visibility_from_mask(mask, 4, inpaint=True)
    assert vis.shape == (4, 1, 8, 12)
    assert vis[:, 0, :4, :].max().item() == 0.0, "the hole must read as 0"
    assert vis[:, 0, 4:, :].min().item() == 1.0, "the kept part must read as 1"


def test_structural_mode_builds_no_visibility_at_all():
    # and the caller then fills it with ones - see assemble_hint.
    mask = torch.ones(4, 8, 12)
    assert visibility_from_mask(mask, 4, inpaint=False) is None


def test_no_mask_means_no_visibility():
    assert visibility_from_mask(None, 4, inpaint=True) is None


# ── frame mapping ───────────────────────────────────────────────────────────

def test_matching_frame_counts_are_the_identity():
    assert torch.equal(frame_indices(124, 124), torch.arange(124))


def test_a_short_mask_is_resampled_not_smeared():
    # ComfyUI clamps: arange(124).clamp(max=36) repeats row 36 for frames
    # 37..123, dragging the last frame of the mask across the whole tail.
    idx = frame_indices(37, 124)
    assert idx.shape[0] == 124
    assert idx[0].item() == 0
    assert idx[-1].item() == 36
    # the giveaway: a clamped mapping has ~87 copies of the last index
    assert int((idx == 36).sum()) < 10, "the tail is still being smeared"
    # and it must stay monotonic, or the mask plays out of order
    assert bool((idx[1:] >= idx[:-1]).all())


def test_a_long_mask_is_spread_not_truncated():
    idx = frame_indices(248, 124)
    assert idx[0].item() == 0
    assert idx[-1].item() == 247, "the end of the mask fell off"


def test_an_empty_sequence_is_named():
    with pytest.raises(ValueError, match="empty"):
        frame_indices(0, 10)


# ── the latent-mask trap ────────────────────────────────────────────────────

def test_a_latent_sized_mask_is_recognised():
    # 96x54 latent against a 768x432 picture: this is the Set Latent Noise Mask
    # mask, wired into the wrong socket.
    assert looks_like_a_latent_mask((54, 96), (54, 96), (432, 768))


def test_a_token_grid_mask_is_recognised_too():
    # already reduced to H3's 2x2 token grid
    assert looks_like_a_latent_mask((27, 48), (54, 96), (432, 768))


def test_a_pixel_mask_is_not_mistaken_for_a_latent_one():
    assert not looks_like_a_latent_mask((432, 768), (54, 96), (432, 768))


def test_a_mask_of_some_other_size_is_left_alone():
    # A half-res pixel mask is a normal thing to hand over; only an exact
    # latent match should trip the refusal.
    assert not looks_like_a_latent_mask((216, 384), (54, 96), (432, 768))


def test_the_degenerate_case_does_not_false_positive():
    # If latent and pixel dims are somehow equal there is nothing to tell apart.
    assert not looks_like_a_latent_mask((54, 96), (54, 96), (54, 96))


# ── the report ──────────────────────────────────────────────────────────────

def test_the_report_names_the_zero_visibility_trap_in_structural_mode():
    text = describe_hint(has_control=True, inpainting=False,
                         frames=124, latent_frames=37)
    assert "ONES" in text
    assert "inpainting" in text.lower()
    assert "124" in text and "37" in text


def test_the_report_says_when_both_are_live():
    text = describe_hint(True, True, 124, 37)
    assert "49" in text


def test_the_report_says_when_nothing_is_connected():
    text = describe_hint(False, False, 124, 37)
    assert "nothing" in text.lower()


# ── the node wiring ─────────────────────────────────────────────────────────

def test_the_node_exposes_the_inpaint_switch():
    from mmx_nodes.mask_aware_control import MiniMaxH3_MaskAwareControlNet

    ids = [getattr(i, "id", None)
           for i in MiniMaxH3_MaskAwareControlNet.define_schema().inputs]
    assert "inpaint" in ids
    assert ids.index("mask") > ids.index("inpaint"), (
        "the switch should sit above the sockets it governs"
    )


def test_inpaint_without_a_mask_is_refused_before_anything_loads():
    from mmx_nodes.mask_aware_control import MiniMaxH3_MaskAwareControlNet

    with pytest.raises(ValueError, match="inpaint is on but no mask"):
        MiniMaxH3_MaskAwareControlNet.execute(
            model=None, control_net=None, vae=None, strength=1.0,
            preserved_strength=0.0, boundary_softness=1.0,
            inpaint=True, control_video=torch.zeros(1, 8, 8, 3),
        )


def test_the_patch_overrides_the_hint_builder():
    # If this stops being an override, the node inherits ComfyUI's zero
    # padding again and the whole fix evaporates silently.
    from mmx_nodes.mask_aware_control import MaskAwareControlPatch

    assert "prepare_control_latent" in MaskAwareControlPatch.__dict__
    assert "after_block" in MaskAwareControlPatch.__dict__
