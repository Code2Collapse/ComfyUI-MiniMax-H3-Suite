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


# -- L8.51: even tiles for odd sizes + MMH3 spatial-split options ----------

from mmx_utils.split_upscale import (  # noqa: E402
    blend_ramp,
    bright_match_tile,
    dynamic_fade_function,
    fade_schedule,
    grid_1d_equal,
    pad_even,
    spatial_grid,
)


@pytest.mark.parametrize("h,w", [(45, 80), (45, 81), (67, 120), (9, 13)])
@pytest.mark.parametrize("min_tile", [0, 256, 512])
def test_tile_starts_stay_on_the_patch_grid(h, w, min_tile):
    # 720p is 45 latent rows: min_tile used to pull the last tile to size - min_tile, an odd row.
    param = build_spatial_param(512, 512, 0.25, 0.5, min_tile, 1.0)
    rows, cols, trows, tcols, _, _ = spatial_grid(param, h, w)
    assert all(r % 2 == 0 for r in rows) and all(c % 2 == 0 for c in cols)
    assert rows[-1] + trows[-1] == h and cols[-1] + tcols[-1] == w


@pytest.mark.parametrize("size,count,ov", [(45, 2, 8), (80, 3, 8), (64, 2, 0), (67, 4, 6), (120, 9, 4)])
def test_equal_tiles(size, count, ov):
    starts, sizes, ovl = grid_1d_equal(size, count, ov)
    assert len(starts) == count and starts[0] == 0 and starts[-1] + sizes[-1] == size
    assert all(s % 2 == 0 for s in starts) and len(set(sizes[:-1])) <= 1
    assert sizes[-1] >= sizes[0]
    assert min(ovl[1:]) >= ov
    param = build_spatial_param(512, 512, 0.25, 0.5, 0, 1.0, grid_rows=count, grid_cols=count)
    assert len(spatial_grid(param, size, size)[0]) == count
    assert "equal tiles" in spatial_param_report(param, size, size)


def test_pad_even_repeats_the_edge():
    x = torch.arange(2 * 3 * 5, dtype=torch.float32).reshape(1, 1, 2, 3, 5)
    y = pad_even(x)
    assert y.shape == (1, 1, 2, 4, 6)
    assert torch.equal(y[..., :3, :5], x) and torch.equal(y[..., 3, :5], x[..., 2, :])
    assert torch.equal(y[..., :3, 5], x[..., :, 4])
    even = torch.zeros(1, 24, 2, 4, 6)
    assert pad_even(even) is even


def test_brightness_match_removes_a_per_frame_offset():
    ref = torch.randn(1, 24, 3, 6, 6)
    off = torch.tensor([0.01, -0.03, 0.04]).view(1, 1, 3, 1, 1)
    assert torch.allclose(bright_match_tile(ref + off, ref), ref, atol=1e-6)
    big = bright_match_tile(ref + 1.0, ref)           # clamped: never more than 0.05 per frame
    assert torch.allclose(big - ref, torch.full_like(ref, 0.95), atol=1e-6)


def test_linear_blend_is_the_old_cross_fade():
    for n in (1, 4, 7):
        _left, right = spatial_crossfade_weights(n)
        assert torch.allclose(blend_ramp(n, "linear"), right)
    assert torch.equal(blend_ramp(4, "midpoint"), torch.tensor([0.0, 0.0, 1.0, 1.0]))
    s = blend_ramp(5, "smoothstep")
    assert float(s[0]) == 0.0 and float(s[-1]) == 1.0 and bool((s[1:] >= s[:-1]).all())
    with pytest.raises(ValueError):
        blend_ramp(3, "cubic")


def test_merge_default_unchanged_and_smoothstep_differs():
    chunk = torch.zeros(1, 2, 1, 8, 8)
    tile = torch.ones(1, 2, 1, 6, 6)
    a = merge_spatial_tile(chunk, tile, 2, 2, 6, 6, 4, 4, 1, 1)
    b = merge_spatial_tile(chunk, tile, 2, 2, 6, 6, 4, 4, 1, 1, blend="linear")
    c = merge_spatial_tile(chunk, tile, 2, 2, 6, 6, 4, 4, 1, 1, blend="smoothstep")
    assert torch.equal(a, b) and not torch.equal(a, c)


def test_fade_schedules():
    assert fade_schedule("off", 6, 2, 5) == [6] * 5
    assert fade_schedule("narrowing", 6, 2, 5) == [6, 5, 4, 3, 2]
    assert fade_schedule("widening", 6, 2, 5) == [2, 3, 4, 5, 6]
    assert fade_schedule("narrowing", 2, 2, 5) == [2] * 5


def test_dynamic_fade_writes_the_video_part_of_the_packed_mask():
    tr, tc, tr_s, tc_s = 5, 6, 6, 6
    numel = 24 * 2 * tr_s * tc_s
    fn = dynamic_fade_function("narrowing", tr, tc, tr_s, tc_s, 4, 4, True, True, 4, 4, 0, 1.0, 0.0, numel)
    assert fn is not None
    assert dynamic_fade_function("off", tr, tc, tr_s, tc_s, 4, 4, True, True, 4, 4, 0, 1.0, 0.0, numel) is None
    sigmas = torch.tensor([1.0, 0.75, 0.5, 0.25, 0.0])
    packed = torch.ones(1, 1, numel + 64)
    first = fn(torch.tensor([1.0]), packed, extra_options={"sigmas": sigmas})
    last = fn(torch.tensor([0.25]), packed, extra_options={"sigmas": sigmas})
    assert torch.equal(first[..., numel:], packed[..., numel:])            # audio part untouched
    v0 = first[0, 0, :numel].reshape(24, 2, tr_s, tc_s)
    v3 = last[0, 0, :numel].reshape(24, 2, tr_s, tc_s)
    assert float(v0[:, :, tr:, :].max()) == 0.0                            # pad strip frozen
    assert float(v3[0, 0, :4, :4].sum()) < float(v0[0, 0, :4, :4].sum())   # narrowing: more frozen at the end


# -- node level with a stand-in sampler -------------------------------------

nt = pytest.importorskip("comfy.nested_tensor")


class _Noise:
    seed = 7

    def generate_noise(self, latent):
        return torch.zeros_like(latent["samples"])


class _Model:
    def __init__(self, options=None):
        self.model_options = dict(options or {})

    def clone(self):
        return _Model(self.model_options)

    def set_model_denoise_mask_function(self, fn):
        self.model_options["denoise_mask_function"] = fn


def _run_split(monkeypatch, h, w, **spatial):
    import mmx_nodes.split_upscale as SU
    calls = []

    def fake_guider(model, cond, negative, cfg):
        return {"model": model, "cond": cond}

    def fake_sample(piece, guider, noise_tensor, seed, sampler, sigmas, callback=None):
        v, a = piece["samples"].unbind()
        mv = piece["noise_mask"].unbind()[0]
        calls.append({"shape": tuple(v.shape), "mask": mv.clone(), "model": guider["model"],
                      "noise": tuple(noise_tensor.unbind()[0].shape)})
        return nt.NestedTensor((v + mv * 1.0, a))     # +1 where generated, untouched where frozen

    monkeypatch.setattr(SU, "build_guider", fake_guider)
    monkeypatch.setattr(SU, "sample_piece", fake_sample)
    monkeypatch.setattr(SU, "make_tile_progress", lambda *a, **k: (lambda idx: None))
    sp = SU.MiniMaxH3_SpatialSplitParams.execute(256, 256, 0.25, 0.5, 0, 1.0, **spatial)
    sp_param = (sp.result if hasattr(sp, "result") else sp.args)[0]
    lat = {"samples": nt.NestedTensor((torch.zeros(1, 24, 2, h, w), torch.zeros(1, 32, 2, 8)))}
    out = SU.MiniMaxH3_SplitUpscale.execute(lat, [[torch.zeros(1, 4, 8), {}]], _Model(), _Noise(), None,
                                            torch.tensor([1.0, 0.5, 0.0]), spatial_split_param=sp_param,
                                            color_match=False)
    res = out.result if hasattr(out, "result") else out.args
    return calls, res[0]["samples"].unbind()[0], sp_param


def test_odd_latent_tiles_are_sampled_at_even_size(monkeypatch):
    calls, video, _ = _run_split(monkeypatch, 45, 23)
    assert len(calls) > 1
    for c in calls:
        assert c["shape"][-2] % 2 == 0 and c["shape"][-1] % 2 == 0 and c["noise"] == c["shape"]
    assert video.shape == (1, 24, 2, 45, 23)
    padded = [c for c in calls if float(c["mask"][..., -1, :].max()) == 0.0]
    assert padded, "the pad strip of the odd bottom tile must be frozen"
    assert float(video[..., -1, :].min()) > 0.5          # the odd last row was generated, not left behind


def test_options_reach_the_sampler(monkeypatch):
    calls, video, sp = _run_split(monkeypatch, 32, 32, brightness_match=True, dynamic_fade="narrowing",
                                  dynamic_fade_min=0, masked_area_noise=0.1, grid_rows=2, grid_cols=2)
    assert sp["dynamic_fade"] == "narrowing" and sp["grid_rows"] == 2
    assert len(calls) == 4
    later = calls[1:]
    assert all("denoise_mask_function" in c["model"].model_options for c in later)
    assert "denoise_mask_function" not in calls[0]["model"].model_options     # first tile: nothing frozen
    assert all(float(c["mask"].min()) >= 0.1 - 1e-6 for c in later)        # masked-area noise lifts the freeze
    _c, plain, _sp = _run_split(monkeypatch, 32, 32, dynamic_fade="narrowing", dynamic_fade_min=0,
                                masked_area_noise=0.1, grid_rows=2, grid_cols=2)
    diff = (video - plain).abs()
    assert 0 < float(diff.max()) <= 0.05 * len(calls) + 1e-6    # acted; 0.05 per tile, carried into later overlaps


def test_saved_workflows_keep_their_values():
    import mmx_nodes.split_upscale as SU
    schema = SU.MiniMaxH3_SpatialSplitParams.define_schema()
    names = [i.id for i in schema.inputs]
    assert names[:6] == ["tile_width", "tile_height", "overlap_ratio", "fade_ratio", "min_tile_size", "seam_denoise"]
    assert all(i.optional for i in schema.inputs[6:])     # API prompts saved with six inputs still validate
    old = SU.MiniMaxH3_SpatialSplitParams.execute(512, 512, 0.25, 0.5, 256, 1.0)
    p = (old.result if hasattr(old, "result") else old.args)[0]
    assert (p["dynamic_fade"], p["brightness_match"], p["overlap_blend"], p["masked_area_noise"], p["grid_rows"]) == (
        "off", False, "linear", 0.0, 0)
