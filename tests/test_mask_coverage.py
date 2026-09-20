"""Coverage thresholding on the pixel -> latent mask reduction.

This is the lever the investigation into "NKD gives better output than MVEx"
turned up, reproduced here on our own node so the finding is a behaviour and
not a note.

The two upstreams agree about H3's latent GRID - both derive [1,4,4,4,4] and
both produce 37 latent rows for a 124-frame clip. They disagree about how a
PARTIALLY covered token block is resolved:

  MVEx  nodes_mask_to_latent.py:56  _token_snap -> F.max_pool2d
        any single masked pixel promotes the whole 32x32 token.

  NKD   mask_core.py:244  blockify -> F.avg_pool2d then `>= threshold`
        the block must be `threshold` covered (their default use is 0.05).

On a face or character swap that difference is the whole result. `max` grows
the mask outward by up to a full token everywhere the subject's boundary is
antialiased - which is everywhere - so a large part of the plate that did not
need regenerating gets regenerated. `coverage` keeps the mask on the subject.

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

from mmx_nodes.mask_to_latent import MiniMaxH3_MaskToLatentSpace as Node  # noqa: E402

# H3: 16 pixels per latent cell, 2x2 latent cells per DiT token -> a 32x32
# pixel block is one token, which is the unit the mask is actually decided on.
SPATIAL = 16
TOKEN = 2
BLOCK = SPATIAL * TOKEN


def reduce(mask, method, threshold=0.05, groups=None, token=TOKEN):
    return Node._reduce(mask, SPATIAL, groups, token, method, "max", 0, 0, threshold)


def one_frame(h_blocks=4, w_blocks=4):
    return torch.zeros(1, h_blocks * BLOCK, w_blocks * BLOCK)


# ── the difference, stated directly ─────────────────────────────────────────

def test_one_stray_pixel_claims_a_whole_token_under_max():
    m = one_frame()
    m[0, 40, 40] = 1.0                      # a single pixel in block (1,1)
    out = reduce(m, "max")
    assert out.max() == 1.0
    # 32x32 pixels -> 2x2 latent cells, all four claimed by that one pixel
    assert int((out > 0).sum()) == TOKEN * TOKEN


def test_one_stray_pixel_claims_nothing_under_coverage():
    m = one_frame()
    m[0, 40, 40] = 1.0
    out = reduce(m, "coverage", threshold=0.05)
    assert int((out > 0).sum()) == 0, (
        "a 1/1024 covered block is 0.1% - it must not read as masked"
    )


def test_a_block_over_the_threshold_is_fully_claimed():
    m = one_frame()
    # 10% of one 32x32 block
    m[0, 32:32 + 10, 32:32 + 11] = 1.0      # 110 / 1024 = 10.7%
    out = reduce(m, "coverage", threshold=0.05)
    assert int((out > 0).sum()) == TOKEN * TOKEN


def test_the_threshold_is_the_dial_between_the_two_behaviours():
    m = one_frame()
    m[0, 32:32 + 10, 32:32 + 11] = 1.0      # 10.7% covered
    assert int((reduce(m, "coverage", threshold=0.05) > 0).sum()) == 4
    assert int((reduce(m, "coverage", threshold=0.20) > 0).sum()) == 0


def test_threshold_zero_reproduces_max_exactly():
    # The documented escape hatch: coverage with threshold 0 must be the old
    # behaviour, not "everything masked" (every block trivially covers >= 0).
    torch.manual_seed(0)
    m = (torch.rand(3, 4 * BLOCK, 4 * BLOCK) > 0.97).float()
    assert torch.equal(reduce(m, "coverage", threshold=0.0),
                       reduce(m, "max"))


# ── the thing the user actually sees ────────────────────────────────────────

def test_an_antialiased_subject_does_not_creep_outward():
    """A soft-edged blob, as any real matte has.

    `max` grows it by a whole token wherever the feather touches a new block;
    `coverage` keeps the edge where the matte put it. The claim under test is
    that coverage masks strictly fewer cells and still covers the solid core.
    """
    size = 8 * BLOCK
    yy, xx = torch.meshgrid(torch.arange(size), torch.arange(size), indexing="ij")
    r = ((yy - size / 2) ** 2 + (xx - size / 2) ** 2).sqrt()
    blob = (1.0 - (r - size * 0.18) / (size * 0.10)).clamp(0, 1)   # feathered disc
    m = blob[None]

    hard = reduce(m, "max")
    soft = reduce(m, "coverage", threshold=0.05)

    n_hard, n_soft = int((hard > 0).sum()), int((soft > 0).sum())
    assert n_soft < n_hard, "coverage must be the tighter of the two"
    # and it must not have eaten the subject: every fully opaque pixel is kept
    core = reduce((m >= 0.999).float(), "min")
    assert bool(((core > 0) <= (soft > 0)).all()), (
        "coverage dropped part of the fully opaque core"
    )


def test_the_saved_plate_is_the_point():
    """Restated as the number that matters: how much of the frame survives."""
    size = 8 * BLOCK
    yy, xx = torch.meshgrid(torch.arange(size), torch.arange(size), indexing="ij")
    r = ((yy - size / 2) ** 2 + (xx - size / 2) ** 2).sqrt()
    m = (1.0 - (r - size * 0.18) / (size * 0.10)).clamp(0, 1)[None]

    kept_max = 1.0 - float((reduce(m, "max") > 0).float().mean())
    kept_cov = 1.0 - float((reduce(m, "coverage", threshold=0.05) > 0).float().mean())
    assert kept_cov > kept_max


# ── it stays a valid H3 mask ────────────────────────────────────────────────

def test_the_result_is_uniform_across_each_2x2_token():
    """H3 reads the mask per 2x2 latent patch (model_base._pool_masks_to_token_grid
    takes an amax over it). A mask that is not already uniform there is
    silently widened at sample time, so the preview would lie."""
    torch.manual_seed(1)
    m = (torch.rand(1, 4 * BLOCK, 4 * BLOCK) > 0.5).float()
    out = reduce(m, "coverage")
    t, h, w = out.shape
    blocks = out.reshape(t, h // TOKEN, TOKEN, w // TOKEN, TOKEN)
    assert bool((blocks.amax(dim=(2, 4)) == blocks.amin(dim=(2, 4))).all())


def test_coverage_output_is_binary():
    # A fractional value is read by H3 as a partial denoise STRENGTH
    # (model.py:645, rows_t = 1 - m * sigma), which fragments rows_t.unique()
    # and is not what a swap wants. coverage must decide, not blend.
    torch.manual_seed(2)
    m = torch.rand(2, 2 * BLOCK, 2 * BLOCK)
    out = reduce(m, "coverage")
    assert set(out.unique().tolist()) <= {0.0, 1.0}


def test_temporal_grouping_still_takes_the_union():
    """Spatial coverage decides per frame; the temporal reduce is still a max.

    That is deliberate: a subject moving through a 4-frame latent group has to
    be regenerated everywhere it goes during that group, so the union is the
    correct answer there even though it is the wrong one spatially.
    """
    m = one_frame(2, 2).repeat(4, 1, 1)
    m[0, 0:BLOCK, 0:BLOCK] = 1.0               # frame 0 only, left block
    m[3, 0:BLOCK, BLOCK:2 * BLOCK] = 1.0       # frame 3 only, right block
    out = reduce(m, "coverage", groups=lambda t: [(0, 4)])
    assert out.shape[0] == 1
    assert int((out > 0).sum()) == 2 * TOKEN * TOKEN


# ── the other methods are untouched ─────────────────────────────────────────

@pytest.mark.parametrize("method", ["max", "min", "mean", "nearest"])
def test_the_existing_methods_ignore_the_new_widget(method):
    torch.manual_seed(3)
    m = torch.rand(1, 2 * BLOCK, 2 * BLOCK)
    a = reduce(m, method, threshold=0.05)
    b = reduce(m, method, threshold=0.90)
    assert torch.equal(a, b), f"{method} changed when coverage_threshold moved"


def test_coverage_is_offered_by_the_schema_and_the_widget_is_last():
    schema = Node.define_schema()
    ids = [getattr(i, "id", None) for i in schema.inputs]
    assert ids[-1] == "coverage_threshold", (
        "widget values serialise as a flat positional array - a new widget "
        "anywhere but last shifts every saved graph's values"
    )
    method = next(i for i in schema.inputs if getattr(i, "id", None) == "spatial_method")
    assert "coverage" in method.options
    assert method.default == "max", "the default behaviour must not change under existing graphs"
