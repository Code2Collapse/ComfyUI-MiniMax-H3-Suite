"""Split / tiled upscaler — CPU-only logic tests (no weights, no GPU)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

PACK = Path(__file__).resolve().parents[1]
if str(PACK) not in sys.path:
    sys.path.insert(0, str(PACK))

from mmx_utils.h3_grid import (  # noqa: E402
    clip_tokens,
    compute_h3_segments_adaptive,
    frames_for_tokens,
    is_h3_compatible,
)
from mmx_utils.split_upscale import (  # noqa: E402
    build_spatial_param,
    build_temporal_param,
    compute_spatial_grid,
    merge_spatial_tile,
    refuse_unless_h3_av_latent,
    spatial_crossfade_weights,
    spatial_param_report,
    split_upscale_report,
    temporal_append,
    temporal_param_report,
)


class _FakeNested:
    def __init__(self, members):
        self._members = tuple(members)

    def unbind(self):
        return self._members


def _h3_av(t_lat=37, h=32, w=32, audio_t=80):
    return _FakeNested((
        torch.zeros(1, 24, t_lat, h, w),
        torch.zeros(1, 32, 2, audio_t),
    ))


# ── temporal segment planning ───────────────────────────────────────────────


def test_124_frame_clip_lands_on_h3_grid():
    assert is_h3_compatible(124)
    tokens = clip_tokens(124)
    assert tokens == 37
    assert frames_for_tokens(tokens) == 124


def test_segment_plan_for_124_frame_latent():
    total_tokens = clip_tokens(124)
    param = build_temporal_param(73, 22, 0.999, 22, 24)
    bounds, total_px = compute_h3_segments_adaptive(
        total_tokens, param["chunk_frames"], param["overlap_frames"],
    )
    assert total_px == 124
    assert bounds
    for k0, f0, k1, f1 in bounds:
        assert 0 <= k0 < k1 <= total_tokens
        assert frames_for_tokens(k1) - frames_for_tokens(k0) == f1 - f0 or k1 == total_tokens


def test_zero_overlap_produces_contiguous_non_overlapping_segments():
    total_tokens = clip_tokens(124)
    chunk_frames = 39  # snaps to legal H3 count
    param = build_temporal_param(chunk_frames, 0, 0.999, 0, 0)
    bounds, _ = compute_h3_segments_adaptive(
        total_tokens, param["chunk_frames"], param["overlap_frames"],
    )
    assert len(bounds) >= 2
    for (k0a, _, k1a, _), (k0b, _, _, _) in zip(bounds, bounds[1:]):
        assert k1a == k0b, "segments must abut with zero overlap"
    assert bounds[-1][2] == total_tokens


# ── spatial grid ────────────────────────────────────────────────────────────


def test_spatial_grid_tiles_without_gap_or_undersized_edge_tiles():
    h, w = 48, 64
    param = build_spatial_param(512, 512, 0.25, 0.5, 256, 1.0)
    rows, cols, trows, tcols, _, _ = compute_spatial_grid(
        h, w, param["th"], param["tw"], param["ol_h"], param["ol_w"], param["mt"], param["mt"],
    )
    assert rows[0] == 0 and cols[0] == 0
    assert rows[-1] + trows[-1] == h
    assert cols[-1] + tcols[-1] == w
    if param["mt"] > 0:
        assert all(tr >= param["mt"] for tr in trows)
        assert all(tc >= param["mt"] for tc in tcols)


def test_crossfade_weights_sum_to_one_across_overlap():
    for n in (1, 4, 8, 16):
        left, right = spatial_crossfade_weights(n)
        assert torch.allclose(left + right, torch.ones(n))


# ── latent validation ───────────────────────────────────────────────────────


def test_non_h3_latent_refused_with_description():
    """The refusal has to say what it GOT, not just what it wanted.

    An H3 AV latent is a nested video+audio pair. Handed an ordinary latent -
    which is what every other model in the graph produces - the message must
    name the shape received, or the user has no way to tell whether they wired
    the wrong node or the wrong output of the right node.
    """
    plain = torch.zeros(1, 4, 8, 16, 16)
    with pytest.raises(ValueError) as e:
        refuse_unless_h3_av_latent(plain)
    msg = str(e.value)
    assert "Expected a MiniMax H3 joint AV latent" in msg
    assert "16" in msg, f"the refusal does not describe what it received: {msg}"


def test_h3_av_latent_accepted():
    refuse_unless_h3_av_latent(_h3_av())


# ── no input mutation ───────────────────────────────────────────────────────


def test_merge_spatial_tile_does_not_mutate_inputs():
    chunk = torch.randn(1, 24, 3, 16, 16)
    tile = torch.randn(1, 24, 3, 8, 8)
    before_chunk = chunk.clone()
    before_tile = tile.clone()
    out = merge_spatial_tile(chunk, tile, 0, 0, 8, 8, 0, 0, 0, 0)
    assert torch.equal(chunk, before_chunk)
    assert torch.equal(tile, before_tile)
    assert out is not chunk


def test_temporal_append_does_not_mutate_accumulator():
    acc_v = torch.ones(1, 24, 4, 8, 8)
    acc_a = torch.ones(1, 32, 2, 10)
    chunk_v = torch.full((1, 24, 3, 8, 8), 2.0)
    chunk_a = torch.full((1, 32, 2, 8), 2.0)
    before_v = acc_v.clone()
    before_a = acc_a.clone()
    temporal_append(acc_v, acc_a, chunk_v, chunk_a, 1, 2, 10, color_match=False)
    assert torch.equal(acc_v, before_v)
    assert torch.equal(acc_a, before_a)


# ── reports ─────────────────────────────────────────────────────────────────


def test_report_states_segment_count_and_tile_grid():
    bounds = [(0, 0, 10, 40), (8, 30, 20, 80)]
    report = split_upscale_report(
        bounds=bounds,
        nrows=2,
        ncols=3,
        row_ovl=[0, 2],
        col_ovl=[0, 4, 4],
        seam_polish="auto",
        polish_count=0,
        color_match=True,
    )
    assert "2 temporal segment(s)" in report
    assert "2x3" in report
    assert "no seam needed repolishing" in report


def test_temporal_param_report_mentions_segment_count():
    param = build_temporal_param(73, 22, 0.999, 22, 24)
    report = temporal_param_report(param, clip_tokens(124))
    assert "segment" in report
    assert "124 pixel frames" in report


def test_spatial_param_report_mentions_grid():
    param = build_spatial_param(512, 512, 0.25, 0.5, 256, 0.8)
    report = spatial_param_report(param, h=48, w=64)
    assert "spatial grid" in report
    assert "48x64" in report
