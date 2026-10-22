"""H3 Tiled Diffusion (ported from bbaudio-2025/Comfyui-MMH3-UltimateUpscale, MIT): tile plan on the 2x2 patch grid,
weights, global window positions against core's own PackedLayout, and the wrapper with stand-in models."""
from __future__ import annotations

import pytest
import torch

from mmx_utils import tiled_diffusion as TD
from mmx_utils.tiled_diffusion import (H3TiledDiffusion, TilingUnsupported, WindowLayout, axis_tiles, pack,
                                       tile_weight, unpack)

minimax = pytest.importorskip("comfy.ldm.minimax.model")
PackedLayout = minimax.PackedLayout

TEXT = 7


# ---- tile plan -----------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("length", [1, 2, 7, 16, 33, 44, 45, 64, 68, 80, 135, 256])
@pytest.mark.parametrize("tile,overlap", [(8, 0), (8, 2), (16, 4), (16, 8), (32, 8), (64, 16), (12, 10)])
def test_axis_tiles_cover_on_even_starts(length, tile, overlap):
    tiles = axis_tiles(length, tile, overlap)
    assert tiles[0][0] == 0 and tiles[-1][1] == length
    covered = torch.zeros(length, dtype=torch.bool)
    for s, e in tiles:
        assert s % 2 == 0, tiles
        assert e - s >= min(tile, length)
        assert e - s <= tile + 1 or len(tiles) == 1
        covered[s:e] = True
    assert bool(covered.all())
    for (s0, e0), (s1, _e1) in zip(tiles, tiles[1:]):
        assert s1 > s0 and e0 - s1 >= overlap, tiles


def test_axis_tiles_fix_upstream_odd_edge():
    # 720p: 45 latent rows. Upstream put the last tile at 45 - 16 = 29 (odd, straddling DiT patches).
    tiles = axis_tiles(45, 16, 4)
    assert all(s % 2 == 0 for s, _ in tiles) and tiles[-1] == (28, 45)


def test_axis_tiles_refuses_bad_settings():
    with pytest.raises(ValueError):
        axis_tiles(64, 16, 16)
    with pytest.raises(ValueError):
        axis_tiles(64, 15, 4)


def test_trapezoid_weights_sum_to_one():
    rows, cols = axis_tiles(45, 16, 4), axis_tiles(80, 32, 8)
    total = torch.zeros(45, 80)
    for i, (h0, h1) in enumerate(rows):
        for j, (w0, w1) in enumerate(cols):
            total[h0:h1, w0:w1] += tile_weight(rows, cols, i, j, "trapezoid")
    assert torch.allclose(total, torch.ones_like(total), atol=1e-6)


@pytest.mark.parametrize("kind", TD.WEIGHTS)
def test_weights_positive(kind):
    rows, cols = axis_tiles(40, 16, 4), axis_tiles(40, 16, 4)
    for i in range(len(rows)):
        for j in range(len(cols)):
            w = tile_weight(rows, cols, i, j, kind)
            assert w.dtype == torch.float32 and float(w.min()) > 0


# ---- window layout against core ------------------------------------------------------------------------------------
def _keyframe(h, w, t=1, frame=0):
    return {"latent": torch.randn(1, 24, t, h, w), "resolved_frame_index": frame}


def test_window_layout_is_the_full_frames_rows():
    T, H, W, A = 3, 10, 12, 5
    kfs = [_keyframe(H, W, 1, 0), _keyframe(H, W, 2, 40)]
    full = PackedLayout(TEXT, T, H, W, A, keyframes=kfs)
    h0, w0, h1, w1 = 2, 4, 8, 10
    win = WindowLayout(full, kfs, T, H, W, h0, w0, h1, w1, A, TEXT)
    grid = lambda lay, kind, t: next(lay.position_ids[a:b] for a, b, k in lay.segments if k == kind)  # noqa: E731
    fv = grid(full, "video", T).reshape(T, H // 2, W // 2, 3)
    wv = grid(win, "video", T).reshape(T, (h1 - h0) // 2, (w1 - w0) // 2, 3)
    assert torch.equal(wv, fv[:, h0 // 2:h1 // 2, w0 // 2:w1 // 2])
    assert torch.equal(grid(win, "audio", T), grid(full, "audio", T))
    conds_f = [full.position_ids[a:b] for a, b, k in full.segments if k == "cond"]
    conds_w = [win.position_ids[a:b] for a, b, k in win.segments if k == "cond"]
    for kf, cf, cw in zip(kfs, conds_f, conds_w):
        vt = kf["latent"].shape[2]
        assert torch.equal(cw.reshape(vt, 3, 3, 3), cf.reshape(vt, H // 2, W // 2, 3)[:, 1:4, 2:5])
    assert win.signature == (TEXT, T, h1 - h0, w1 - w0, A)
    assert int(win.img_update.sum()) == T * 3 * 3 and int(win.audio_update.sum()) == A * 2
    assert win.seq_len == sum(b - a for a, b, _ in win.segments)


def test_window_over_the_whole_frame_equals_core_layout():
    T, H, W, A = 2, 8, 6, 4
    full = PackedLayout(TEXT, T, H, W, A)
    win = WindowLayout(full, None, T, H, W, 0, 0, H, W, A, TEXT)
    assert torch.equal(win.position_ids, full.position_ids)
    assert win.segments == full.segments and win.signature == full.signature
    assert torch.equal(win.img_update, full.img_update) and torch.equal(win.audio_update, full.audio_update)


def test_window_odd_edge_pads_inside_the_grid():
    T, H, W, A = 1, 45, 8, 2
    full = PackedLayout(TEXT, T, 46, W, A)
    win = WindowLayout(full, None, T, 46, W, 28, 0, 45, W, A, TEXT)
    assert win.signature == (TEXT, T, 18, W, A)
    with pytest.raises(ValueError):
        WindowLayout(full, None, T, 46, W, 29, 0, 45, W, A, TEXT)


# ---- the wrapper ---------------------------------------------------------------------------------------------------
def _call_args(T=2, H=12, W=20, A=6, payload=None, **extra):
    video = torch.randn(1, 24, T, H, W)
    audio = torch.randn(1, 32, 2, A)
    x, shapes = pack([video, audio])
    c = {"c_crossattn": torch.zeros(1, TEXT, 8), "latent_shapes": shapes,
         "minimax_payload": {} if payload is None else payload, "transformer_options": {"sigmas": torch.tensor([0.5])}}
    c.update(extra)
    return {"input": x, "timestep": torch.tensor([500.0]), "c": c, "cond_or_uncond": [0]}, video, audio


def pointwise(calls=None):
    def fn(x, t, **c):
        v, a = unpack(x, c["latent_shapes"])
        if calls is not None:
            calls.append({k: c.get(k) for k in ("minimax_payload", "denoise_mask")} | {"shape": tuple(v.shape)})
        m = c.get("denoise_mask")
        out_v = 2 * v + 1 if m is None else (2 * v + 1) * m
        return pack([out_v, 3 * a])[0]
    return fn


def full_frame(args, fn):
    return fn(args["input"], args["timestep"], **args["c"])


@pytest.mark.parametrize("method", TD.WEIGHTS)
def test_pointwise_model_tiled_equals_untiled(method):
    args, _v, _a = _call_args()
    calls = []
    td = H3TiledDiffusion(method, 6, 8, 2)
    out = td(pointwise(calls), args)
    assert len(calls) == len(axis_tiles(12, 6, 2)) * len(axis_tiles(20, 8, 2)) > 1
    assert torch.allclose(out, full_frame(args, pointwise()), atol=1e-5)
    assert "tiles" in td.last_report


def position_model(x, t, **c):
    """Writes each cell's packed-layout position into the video output: tiles must see the full frame's positions."""
    v, a = unpack(x, c["latent_shapes"])
    T, H, W = v.shape[2:]
    lay = c["minimax_payload"]["layout"]
    hp, wp = H + H % 2, W + W % 2
    pos = next(lay.position_ids[s:e] for s, e, k in lay.segments if k == "video").reshape(T, hp // 2, wp // 2, 3)
    cells = pos.repeat_interleave(2, 1).repeat_interleave(2, 2)[:, :H, :W].permute(3, 0, 1, 2).float()
    out = torch.zeros_like(v)
    out[0, :3] = cells
    return pack([out, a * 0])[0]


def test_tiles_see_global_positions():
    T, H, W, A = 2, 13, 22, 6
    args, video, _ = _call_args(T, H, W, A)
    args["c"]["minimax_payload"] = {"layout": PackedLayout(TEXT, T, H + 1, W, A)}
    ref = full_frame(args, position_model)
    out = H3TiledDiffusion("gaussian", 6, 8, 2)(position_model, args)
    assert torch.allclose(out, ref, atol=1e-5)


def test_layout_built_when_payload_has_none():
    args, _v, _a = _call_args(2, 10, 10, 4)
    td = H3TiledDiffusion("trapezoid", 6, 6, 2, layout_factory=PackedLayout)
    out = td(position_model, args)
    args["c"]["minimax_payload"] = {"layout": PackedLayout(TEXT, 2, 10, 10, 4)}
    assert torch.allclose(out, full_frame(args, position_model), atol=1e-5)


def test_single_tile_calls_the_model_untouched():
    args, _v, _a = _call_args()
    seen = []

    def fn(x, t, **c):
        seen.append((x, c))
        return x
    out = H3TiledDiffusion("trapezoid", 64, 64, 8)(fn, args)
    assert len(seen) == 1 and seen[0][0] is args["input"] and out is args["input"]
    assert seen[0][1].keys() == args["c"].keys() and all(seen[0][1][k] is v for k, v in args["c"].items())


def test_kill_switch(monkeypatch):
    monkeypatch.setenv(TD.KILL_SWITCH, "off")
    args, _v, _a = _call_args()
    calls = []
    H3TiledDiffusion("trapezoid", 4, 4, 2)(pointwise(calls), args)
    assert len(calls) == 1


def test_audio_is_whole_and_averaged():
    args, _v, audio = _call_args()
    k = {"n": 0}

    def fn(x, t, **c):
        v, a = unpack(x, c["latent_shapes"])
        assert torch.equal(a, audio)
        k["n"] += 1
        return pack([v, a + k["n"]])[0]
    out = H3TiledDiffusion("trapezoid", 6, 8, 2)(fn, args)
    n = k["n"]
    _ov, oa = unpack(out, args["c"]["latent_shapes"])
    assert torch.allclose(oa, audio + (n + 1) / 2.0)


def test_denoise_mask_is_cropped_per_tile():
    T, H, W = 2, 12, 20
    mask = (torch.rand(1, 1, T, H, W) > 0.5).float()
    args, _v, _a = _call_args(T, H, W, denoise_mask=mask)
    calls = []
    out = H3TiledDiffusion("gaussian", 6, 8, 2)(pointwise(calls), args)
    assert all(cl["denoise_mask"].shape[-2:] == cl["shape"][-2:] for cl in calls)
    assert torch.allclose(out, full_frame(args, pointwise()), atol=1e-5)


def test_bf16_accumulates_in_fp32():
    args, _v, _a = _call_args(2, 16, 24)
    args["input"] = args["input"].to(torch.bfloat16)
    out = H3TiledDiffusion("gaussian", 8, 8, 4)(pointwise(), args)
    ref = full_frame(args, pointwise())
    assert out.dtype == torch.bfloat16
    assert float((out.float() - ref.float()).abs().max()) <= 0.07     # bf16 rounding of the result only


def test_cond_and_uncond_keep_their_own_plan():
    td = H3TiledDiffusion("trapezoid", 6, 8, 2)
    a1, _v, _a = _call_args()
    a2, _v, _a = _call_args()
    for _ in range(3):
        td(pointwise(), a1)
        td(pointwise(), a2)
    assert len(td._plans) == 2


def test_keyframes_cropped_and_refs_kept():
    T, H, W, A = 2, 12, 16, 4
    kf = {"latent": torch.randn(1, 24, 1, H // 2, W // 2), "resolved_frame_index": 0}   # needs the resize
    ref = {"latent": torch.randn(1, 24, 1, 4, 4)}
    payload = {"keyframes": [kf], "refs": [{"kind": "image", "latent_h": 4, "latent_w": 4, **ref}]}
    args, _v, _a = _call_args(T, H, W, A, payload=payload)
    calls = []
    H3TiledDiffusion("trapezoid", 6, 8, 2, layout_factory=PackedLayout)(pointwise(calls), args)
    for cl in calls:
        p = cl["minimax_payload"]
        th, tw = cl["shape"][-2:]
        lat = p["keyframes"][0]["latent"]
        assert lat.shape[-2:] == (th + th % 2, tw + tw % 2)
        assert p["cond_video_latents"][0] is lat and p["cond_video_latents"][1] is ref["latent"]
        assert p["refs"] is payload["refs"]


class MiniMaxH3FunControlPatch:     # stand-in with core's attribute names
    def __init__(self, full_shape):
        self.sigma_start, self.sigma_end = 1.0, 0.0
        self.control_latent = None
        self.control_latent_shape = None
        self.encodes = 0

    def prepare_control_latent(self, target_shape):
        target_shape = tuple(target_shape)
        if self.control_latent is not None and self.control_latent_shape == target_shape:
            return
        self.encodes += 1
        n = 1
        for s in target_shape:
            n *= s
        self.control_latent = torch.arange(n, dtype=torch.float32).reshape(target_shape)
        self.control_latent_shape = target_shape

    def diffusion_model_wrapper(self, executor, *a, **k):  # pragma: no cover - only its __self__ is read
        return executor(*a, **k)


def test_core_fun_control_gets_its_tile_of_the_hint():
    T, H, W = 2, 12, 20
    args, video, _ = _call_args(T, H, W)
    patch = MiniMaxH3FunControlPatch(video.shape)
    args["c"]["transformer_options"]["wrappers"] = {"diffusion_model": {"fun": [patch.diffusion_model_wrapper]}}

    def fn(x, t, **c):         # what core's patch does inside the model: fit the hint to x[0].shape, then use it
        v, a = unpack(x, c["latent_shapes"])
        patch.prepare_control_latent(tuple(v.shape))
        return pack([patch.control_latent.clone(), a])[0]
    out = H3TiledDiffusion("trapezoid", 6, 8, 2)(fn, args)
    ov, _ = unpack(out, args["c"]["latent_shapes"])
    full_hint = torch.arange(video.numel(), dtype=torch.float32).reshape(video.shape)
    assert torch.allclose(ov, full_hint) and patch.encodes == 1
    assert patch.control_latent_shape == tuple(video.shape)


def test_refusals():
    args, video, _ = _call_args()
    with pytest.raises(TilingUnsupported, match="ControlNet"):
        H3TiledDiffusion("trapezoid", 6, 8, 2)(pointwise(), dict(args, c=dict(args["c"], control=object())))
    with pytest.raises(TilingUnsupported, match="MiniMax H3"):
        H3TiledDiffusion("trapezoid", 6, 8, 2)(pointwise(), dict(args, c={k: v for k, v in args["c"].items()
                                                                           if k != "minimax_payload"}))
    opts = dict(args["c"]["transformer_options"], minimax_h3_first_block_cache=True)
    with pytest.raises(TilingUnsupported, match="cache"):
        H3TiledDiffusion("trapezoid", 6, 8, 2)(pointwise(), dict(args, c=dict(args["c"], transformer_options=opts)))

    class MaskAwareControlPatch(MiniMaxH3FunControlPatch):
        pass
    p = MaskAwareControlPatch(video.shape)
    opts = dict(args["c"]["transformer_options"], wrappers={"diffusion_model": {"m": [p.diffusion_model_wrapper]}})
    with pytest.raises(TilingUnsupported, match="whole frame"):
        H3TiledDiffusion("trapezoid", 6, 8, 2)(pointwise(), dict(args, c=dict(args["c"], transformer_options=opts)))


def test_inner_wrapper_runs_per_tile():
    args, _v, _a = _call_args()
    seen = []

    def inner(model_function, a):
        seen.append(a["cond_or_uncond"])
        return model_function(a["input"], a["timestep"], **a["c"])
    out = H3TiledDiffusion("trapezoid", 6, 8, 2, inner=inner)(pointwise(), args)
    assert len(seen) > 1 and torch.allclose(out, full_frame(args, pointwise()), atol=1e-5)


# ---- the node ------------------------------------------------------------------------------------------------------
class FakePatcher:
    def __init__(self, options=None):
        self.model_options = dict(options or {})

    def clone(self):
        return FakePatcher(self.model_options)

    def set_model_unet_function_wrapper(self, fn):
        self.model_options["model_function_wrapper"] = fn


def test_node_patches_a_clone_and_chains():
    from mmx_nodes.tiled_diffusion import MiniMaxH3_TiledDiffusion
    prev = lambda f, a: f(a["input"], a["timestep"], **a["c"])  # noqa: E731
    base = FakePatcher({"model_function_wrapper": prev})
    out = MiniMaxH3_TiledDiffusion.execute(base, "gaussian", 1000, 1000, 250)
    patched, report = out.result if hasattr(out, "result") else out.args
    wrap = patched.model_options["model_function_wrapper"]
    assert isinstance(wrap, H3TiledDiffusion) and wrap.inner is prev and base.model_options["model_function_wrapper"] is prev
    assert (wrap.tile_w, wrap.tile_h, wrap.overlap) == (62, 62, 16) and "overlap" in report
    with pytest.raises(ValueError, match="smaller than the tile"):
        MiniMaxH3_TiledDiffusion.execute(FakePatcher(), "trapezoid", 512, 512, 512)
