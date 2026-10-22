# PORTED FROM: https://github.com/bbaudio-2025/Comfyui-MMH3-UltimateUpscale@0a8d5ee nodes/tiled_diffusion.py
# (_WindowLayout, _H3TiledDiffusionImpl), MIT, Copyright (c) 2026 bbaudio-2025; upstream follows
# shiimizu/ComfyUI-TiledDiffusion (MultiDiffusion / Mixture of Diffusers). Tile placement and the trapezoid weights
# follow ComfyUI-LTX-Enhanced ltx_enhanced/tiles.py (Apache-2.0, Code2Collapse).
# Changed here (docs/h3_tiled_sampling_design.md): tile starts always even (upstream put the last tile on an odd cell
# for odd latent sizes such as 720p), fp32 accumulation, one plan per conditioning payload (positive and negative no
# longer rebuild every step), video denoise masks cropped per tile instead of refused, core's H3 Fun ControlNet patch
# given its own hint crop per tile (upstream let it squash the whole control video into every tile), step caches and
# the Mask-Aware ControlNet refused, trapezoid weights, a single tile calls the model untouched, kill switch.
"""Per-step spatial tiling for MiniMax H3: every model call is split into tiles and the predictions are blended.

All tiles advance through the same sigmas on the same noise, so there is nothing to freeze and no seam to repair. Each
tile keeps its GLOBAL frame positions (rows of the full frame's packed layout), so a tile predicts what the full-frame
model would predict inside that window, and overlap blending averages consistent predictions.
"""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass, field
from typing import Any, Callable

import torch

LOG = logging.getLogger("MiniMaxSuite.tiled_diffusion")

WEIGHTS = ("trapezoid", "gaussian", "uniform")
KILL_SWITCH = "MMX_TILED_DIFFUSION"
# DIFFUSION_MODEL step caches would take the tiles of one step for consecutive steps.
STEP_CACHE_KEYS = ("minimax_h3_first_block_cache", "minimax_h3_block_cache_t8")
MAX_PLANS = 8


class TilingUnsupported(ValueError):
    """A call that cannot be tiled correctly; the message says why and what to do."""


def disabled() -> bool:
    return os.environ.get(KILL_SWITCH, "").strip().lower() in ("off", "0", "false", "no", "disable", "disabled")


def ceil2(x: int) -> int:
    """Round up to the next even number (the H3 2x2 patch grid)."""
    return (int(x) + 1) // 2 * 2


def pack(tensors: list[torch.Tensor]) -> tuple[torch.Tensor, list[torch.Size]]:
    """comfy.utils.pack_latents."""
    shapes = [t.shape for t in tensors]
    return torch.cat([t.reshape(t.shape[0], 1, -1) for t in tensors], dim=-1), shapes


def unpack(packed: torch.Tensor, shapes) -> list[torch.Tensor]:
    """comfy.utils.unpack_latents."""
    if shapes is None or len(shapes) <= 1:
        return [packed]
    out = []
    for shape in shapes:
        n = math.prod(shape[1:])
        out.append(packed[:, :, :n].reshape([packed.shape[0]] + list(shape)[1:]))
        packed = packed[:, :, n:]
    return out


def axis_tiles(length: int, tile: int, overlap: int) -> list[tuple[int, int]]:
    """[(start, end)] covering [0, length) on one axis, in latent cells.

    The fewest tiles whose overlap is at least `overlap`, first at 0, last running to the edge, the rest spread evenly.
    Every start is even, so each tile is made of whole 2x2 DiT patches of the full frame; the last tile starts at the
    even cell at or below length - tile and may be one cell larger than `tile`.
    """
    length, tile, overlap = int(length), int(tile), int(overlap)
    if length < 1 or tile < 2:
        raise ValueError(f"length must be >= 1 and tile >= 2 latent cells, got length={length}, tile={tile}")
    if tile % 2 or overlap % 2 or overlap < 0:
        raise ValueError(f"tile and overlap must be even latent cells (32 px), got tile={tile}, overlap={overlap}")
    if tile >= length:
        return [(0, length)]
    if overlap >= tile:
        raise ValueError(f"the overlap ({overlap}) must be smaller than the tile ({tile})")
    last = (length - tile) // 2 * 2
    for n in range(2, length + 2):
        starts = sorted({2 * round(i * last / (2 * (n - 1))) for i in range(n)})
        tiles = [(s, min(s + tile, length)) for s in starts]
        tiles[-1] = (starts[-1], length)
        if all(tiles[i][1] - tiles[i + 1][0] >= overlap for i in range(len(tiles) - 1)):
            return tiles
    raise ValueError(f"no tiling of {length} cells with tile {tile} and overlap {overlap}")  # pragma: no cover


def _ramp_1d(size: int, left: int, right: int) -> torch.Tensor:
    """1 inside, linear ramps over the overlaps. A ramp of r cells takes the r interior points of a 0..1 line with r + 2
    points, so two neighbours sum to exactly 1 and a frame edge (no neighbour) keeps full weight."""
    w = torch.ones(size, dtype=torch.float64)
    left, right = max(0, min(left, size)), max(0, min(right, size))
    if left:
        w[:left] = torch.linspace(0.0, 1.0, left + 2, dtype=torch.float64)[1:-1]
    if right:
        w[size - right:] = torch.minimum(w[size - right:],
                                         torch.linspace(1.0, 0.0, right + 2, dtype=torch.float64)[1:-1])
    return w


def tile_weight(rows: list[tuple[int, int]], cols: list[tuple[int, int]], i: int, j: int, kind: str) -> torch.Tensor:
    """[th, tw] float32 blend weight of tile (row i, column j)."""
    (h0, h1), (w0, w1) = rows[i], cols[j]
    th, tw = h1 - h0, w1 - w0
    if kind == "uniform":
        return torch.ones(th, tw, dtype=torch.float32)
    if kind == "gaussian":
        # upstream: one sigma for both axes, min(th, tw) / 8
        sigma = max(min(th, tw) / 8.0, 1e-6)
        yy = torch.arange(th, dtype=torch.float64) - (th - 1) / 2.0
        xx = torch.arange(tw, dtype=torch.float64) - (tw - 1) / 2.0
        return torch.exp(-(yy[:, None] ** 2 + xx[None, :] ** 2) / (2.0 * sigma * sigma)).float()
    if kind != "trapezoid":
        raise ValueError(f"method must be one of {', '.join(WEIGHTS)}; got {kind!r}")

    def ramps(axis, k):
        a, b = axis[k]
        left = axis[k - 1][1] - a if k > 0 else 0
        right = b - axis[k + 1][0] if k + 1 < len(axis) else 0
        return left, right

    wy = _ramp_1d(th, *ramps(rows, i))
    wx = _ramp_1d(tw, *ramps(cols, j))
    return (wy[:, None] * wx[None, :]).float()


class WindowLayout:
    """PackedLayout-compatible layout of one tile window.

    Video and keyframe rows keep the GLOBAL coordinates of the window (rows of the full frame's layout) instead of
    restarting the grid at the tile origin; text, references and audio rows are the full frame's, so every tile places
    the audio and the references identically. Core reads position_ids, segments, img_update, audio_update, seq_len and
    signature (comfy/ldm/minimax/model.py, MiniMaxH3Model._forward / _embed_and_pack)."""

    def __init__(self, full, keyframes, T, Hp, Wp, h0, w0, h1, w1, audio_t, text_len):
        hp1, wp1 = h0 + ceil2(h1 - h0), w0 + ceil2(w1 - w0)
        if h0 % 2 or w0 % 2 or hp1 > Hp or wp1 > Wp:
            raise ValueError(f"window ({h0}:{h1}, {w0}:{w1}) is not on the {Hp}x{Wp} patch grid")
        idx = (torch.arange(h0 // 2, hp1 // 2, dtype=torch.long)[:, None] * (Wp // 2)
               + torch.arange(w0 // 2, wp1 // 2, dtype=torch.long)[None, :]).reshape(-1)
        frame_rows = (Hp // 2) * (Wp // 2)
        # temporal steps of each keyframe that packs a video latent, in PackedLayout's 'cond' segment order
        kf_vts = [kf["latent"].shape[2] for kf in (keyframes or ()) if kf.get("latent") is not None]
        pos, img_pos, img_upd, aud_pos, aud_upd, segments = [], [], [], [], [], []
        kf_i = 0
        row = 0
        for a, b, kind in full.segments:
            p = full.position_ids[a:b]
            if kind == "video":
                p = p.reshape(T, frame_rows, 3)[:, idx, :].reshape(-1, 3)
            elif kind == "cond":
                vt = kf_vts[kf_i]
                kf_i += 1
                p = p.reshape(vt, frame_rows, 3)[:, idx, :].reshape(-1, 3)
            n = p.shape[0]
            pos.append(p)
            segments.append((row, row + n, kind))
            if kind in ("cond", "ref_img", "video"):
                img_pos.append(torch.arange(row, row + n))
                img_upd.append(torch.full((n,), kind == "video", dtype=torch.bool))
            elif kind in ("cond_audio", "ref_audio", "audio"):
                aud_pos.append(torch.arange(row, row + n))
                aud_upd.append(torch.full((n,), kind == "audio", dtype=torch.bool))
            row += n
        self.position_ids = torch.cat(pos)
        self.img_pos = torch.cat(img_pos)
        self.img_update = torch.cat(img_upd)
        self.audio_pos = torch.cat(aud_pos)
        self.audio_update = torch.cat(aud_upd)
        self.segments = segments
        self.seq_len = row
        self.signature = (text_len, T, hp1 - h0, wp1 - w0, audio_t)
        self.window = (h0, w0, h1, w1)


def crop_keyframes(keyframes, H, W, h0, w0, h1, w1, pad: Callable[[torch.Tensor], torch.Tensor]):
    """Keyframe dicts with their video latents resized to the frame grid (if needed) and cropped to the window."""
    out, cond_v = [], []
    for kf in keyframes or ():
        nkf = dict(kf)
        lt = nkf.get("latent")
        if lt is not None:
            if lt.shape[3] != H or lt.shape[4] != W:
                b, c, t, hk, wk = lt.shape
                lt = torch.nn.functional.interpolate(lt.reshape(b * t, c, hk, wk).float(), size=(H, W),
                                                     mode="bilinear", align_corners=False
                                                     ).reshape(b, c, t, H, W).to(lt.dtype)
            nkf["latent"] = pad(lt[:, :, :, h0:h1, w0:w1].contiguous())
            cond_v.append(nkf["latent"])
        out.append(nkf)
    return out, cond_v


def _pad_to_patch(x: torch.Tensor) -> torch.Tensor:
    """comfy.ldm.common_dit.pad_to_patch_size(x, (1, 2, 2)) (circular padding)."""
    ph, pw = (-x.shape[-2]) % 2, (-x.shape[-1]) % 2
    if not ph and not pw:
        return x
    return torch.nn.functional.pad(x, (0, pw, 0, ph, 0, 0), mode="circular")


def control_patches(c: dict) -> list:
    """Core H3 Fun ControlNet patches among the DIFFUSION_MODEL wrappers of this call. Refuses the Mask-Aware
    ControlNet, whose row gate is built for the whole frame."""
    opts = c.get("transformer_options") or {}
    found = []
    for fns in (opts.get("wrappers", {}) or {}).get("diffusion_model", {}).values():
        for fn in fns:
            owner = getattr(fn, "__self__", None)
            if owner is None or not hasattr(owner, "prepare_control_latent"):
                continue
            if type(owner).__name__ != "MiniMaxH3FunControlPatch":
                raise TilingUnsupported(
                    f"H3 Tiled Diffusion cannot split '{type(owner).__name__}' into tiles: it gates the rows of the "
                    "whole frame. Use core's 'Apply MiniMax H3 Fun ControlNet' with Tiled Diffusion, or H3 Split "
                    "Upscale with the Mask-Aware ControlNet.")
            found.append(owner)
    return found


@dataclass
class Plan:
    rows: list
    cols: list
    tiles: list                      # [(h0, w0, h1, w1)]
    weights: list                    # [th, tw] float32 (CPU)
    payloads: list                   # per-tile minimax_payload
    payload: Any = field(repr=False, default=None)   # keeps id(payload) valid while cached

    @property
    def cost(self) -> float:
        h, w = self.rows[-1][1], self.cols[-1][1]
        return sum((h1 - h0) * (w1 - w0) for h0, w0, h1, w1 in self.tiles) / float(max(1, h * w))


class H3TiledDiffusion:
    """model_function_wrapper (ModelPatcher.set_model_unet_function_wrapper).

    Receives the packed AV latent ([1, 1, video + audio elements], comfy.utils.pack_latents) and the cond dict with
    latent_shapes and the minimax payload (keyframes / refs / prebuilt layout)."""

    def __init__(self, method: str, tile_h: int, tile_w: int, overlap: int, inner=None,
                 layout_factory: Callable | None = None):
        if method not in WEIGHTS:
            raise ValueError(f"method must be one of {', '.join(WEIGHTS)}; got {method!r}")
        self.method = method
        self.tile_h, self.tile_w, self.overlap = int(tile_h), int(tile_w), int(overlap)
        self.inner = inner                # a model_function_wrapper that was set before this one
        self.layout_factory = layout_factory
        self._plans: dict = {}
        self.last_report = "not run yet"

    # -- planning -------------------------------------------------------------------------------------------------
    def _factory(self):
        if self.layout_factory is None:
            from comfy.ldm.minimax.model import PackedLayout
            self.layout_factory = PackedLayout
        return self.layout_factory

    def plan(self, payload: dict, video: torch.Tensor, audio_t: int, text_len: int) -> Plan:
        _, _, T, H, W = video.shape
        key = (id(payload), T, H, W, audio_t, text_len)
        cached = self._plans.get(key)
        if cached is not None and cached.payload is payload:
            return cached
        rows = axis_tiles(H, self.tile_h, self.overlap)
        cols = axis_tiles(W, self.tile_w, self.overlap)
        Hp, Wp = ceil2(H), ceil2(W)
        keyframes, refs = payload.get("keyframes"), payload.get("refs")
        full = payload.get("layout")
        if full is None or getattr(full, "signature", None) != (text_len, T, Hp, Wp, audio_t):
            full = self._factory()(text_len, T, Hp, Wp, audio_t, keyframes=keyframes, refs=refs)
        tiles, weights, payloads = [], [], []
        for i, (h0, h1) in enumerate(rows):
            for j, (w0, w1) in enumerate(cols):
                tiles.append((h0, w0, h1, w1))
                weights.append(tile_weight(rows, cols, i, j, self.method))
                p = dict(payload)
                if keyframes:
                    kfs, cond_v = crop_keyframes(keyframes, H, W, h0, w0, h1, w1, _pad_to_patch)
                    p["keyframes"] = kfs
                    # refs keep their own grid: only keyframes are cropped, and the flat list mirrors that order
                    p["cond_video_latents"] = cond_v + [r["latent"] for r in (refs or ()) if "latent" in r]
                p["layout"] = WindowLayout(full, keyframes, T, Hp, Wp, h0, w0, h1, w1, audio_t, text_len)
                payloads.append(p)
        plan = Plan(rows, cols, tiles, weights, payloads, payload)
        if len(self._plans) >= MAX_PLANS:
            self._plans.clear()
        self._plans[key] = plan
        self.last_report = (f"{len(tiles)} tiles ({len(rows)} x {len(cols)}) on a {H * 16} x {W * 16} px frame "
                            f"({H} x {W} cells), overlap >= {self.overlap * 16} px, {self.method} weights, "
                            f"{plan.cost:.2f}x the compute of one full-frame call")
        LOG.info("[H3 Tiled Diffusion] %s", self.last_report)
        return plan

    # -- the call -------------------------------------------------------------------------------------------------
    def _call(self, model_function, x, t, c, cond_or_uncond):
        if self.inner is not None:
            return self.inner(model_function, {"input": x, "timestep": t, "c": c, "cond_or_uncond": cond_or_uncond})
        return model_function(x, t, **c)

    def __call__(self, model_function, args):
        x, t, c = args["input"], args["timestep"], args["c"]
        cou = args.get("cond_or_uncond")
        if disabled():
            return self._call(model_function, x, t, c, cou)
        payload = c.get("minimax_payload")
        if payload is None:
            raise TilingUnsupported("H3 Tiled Diffusion only works with the MiniMax H3 model.")
        if c.get("control") is not None:
            raise TilingUnsupported(
                "H3 Tiled Diffusion cannot crop a ControlNet that rides in the conditioning. Use core's 'Apply "
                "MiniMax H3 Fun ControlNet' (a model patch), or H3 Split Upscale.")
        opts = c.get("transformer_options") or {}
        for k in STEP_CACHE_KEYS:
            if opts.get(k):
                raise TilingUnsupported(
                    "H3 Tiled Diffusion cannot run with a block / step cache (First Block Cache, T8 Block Cache): the "
                    "cache would take the tiles of one step for consecutive steps. Remove the cache node.")
        shapes = c.get("latent_shapes")
        streams = unpack(x, shapes)
        video = streams[0]
        audio = streams[1] if len(streams) > 1 else None
        if video.ndim != 5:
            raise TilingUnsupported(f"expected an H3 video latent [1, 24, T, H, W], got {tuple(video.shape)}")
        if video.shape[0] != 1:
            raise TilingUnsupported("MiniMax H3 runs batch size 1; H3 Tiled Diffusion does too.")
        text = c.get("c_crossattn")
        text_len = int(text.shape[1]) if text is not None else 0
        audio_t = int(audio.shape[-1]) if audio is not None else 0
        plan = self.plan(payload, video, audio_t, text_len)
        if len(plan.tiles) == 1:
            return self._call(model_function, x, t, c, cou)

        _, _, T, H, W = video.shape
        controls = control_patches(c)
        sigmas = opts.get("sigmas")
        sigma = float(sigmas.flatten()[0]) if sigmas is not None else float(t.flatten()[0]) / 1000.0
        full_hints = []
        for patch in controls:
            active = patch.sigma_end <= sigma <= patch.sigma_start
            if active:
                patch.prepare_control_latent(tuple(video.shape))
            full_hints.append((active, patch.control_latent, patch.control_latent_shape))
        mask = c.get("denoise_mask")
        if mask is not None and tuple(mask.shape[-3:]) != (T, H, W):
            raise TilingUnsupported(f"the video denoise mask {tuple(mask.shape)} is not on the latent grid "
                                    f"{(T, H, W)}; H3 Tiled Diffusion crops it per tile")

        buf = torch.zeros(video.shape, dtype=torch.float32, device=video.device)
        wsum = torch.zeros((1, 1, 1, H, W), dtype=torch.float32, device=video.device)
        audio_acc = None
        try:
            for (h0, w0, h1, w1), w, p in zip(plan.tiles, plan.weights, plan.payloads):
                c_t = dict(c)
                c_t["minimax_payload"] = p
                if mask is not None:
                    c_t["denoise_mask"] = mask[..., h0:h1, w0:w1]
                v_t = video[:, :, :, h0:h1, w0:w1]
                for patch, (active, hint, _shape) in zip(controls, full_hints):
                    if active and hint is not None:
                        patch.control_latent = hint[..., h0:h1, w0:w1]
                        patch.control_latent_shape = tuple(v_t.shape)
                if audio is not None:
                    x_t, shapes_t = pack([v_t, audio])
                    c_t["latent_shapes"] = shapes_t
                    out = unpack(self._call(model_function, x_t, t, c_t, cou), shapes_t)
                    v_out, a_out = out[0], out[1]
                else:
                    v_out, a_out = self._call(model_function, v_t, t, c_t, cou), None
                wd = w.to(device=video.device)[None, None, None]
                buf[:, :, :, h0:h1, w0:w1] += v_out.float() * wd
                wsum[:, :, :, h0:h1, w0:w1] += wd
                if a_out is not None:
                    audio_acc = a_out.float() if audio_acc is None else audio_acc + a_out.float()
        finally:
            for patch, (_active, hint, shape) in zip(controls, full_hints):
                patch.control_latent, patch.control_latent_shape = hint, shape
        out_v = (buf / wsum.clamp(min=1e-12)).to(video.dtype)
        if audio is None:
            return out_v
        # every tile saw the whole audio stream; its predictions are averaged
        return pack([out_v, (audio_acc / len(plan.tiles)).to(audio.dtype)])[0]


def snap_px(px: int) -> int:
    """Pixels -> latent cells on the 32 px patch grid."""
    return int(round(int(px) / 32.0)) * 2


def describe_settings(method: str, tile_w_px: int, tile_h_px: int, overlap_px: int) -> str:
    tw, th, ov = snap_px(tile_w_px), snap_px(tile_h_px), snap_px(overlap_px)
    return (f"tiles {tw * 16} x {th * 16} px ({tw} x {th} latent cells), overlap >= {ov * 16} px, {method} weights; "
            f"the frame is split when it is larger than a tile. Kill switch: {KILL_SWITCH}=off.")
