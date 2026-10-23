"""H3 context-window executor (L8.36): every model call runs as overlapping time windows inside one sampling pass, and
their predictions are fused. Design: docs/h3_context_window_executor_design.md. Original code (GPL-3.0); the window
layout follows the same rule as H3 Tiled Diffusion (mmx_utils/tiled_diffusion.py): a window's rows are the full clip's
layout rows, so it predicts what the whole clip would predict there.
"""

from __future__ import annotations

import logging
import os
from typing import Callable

import torch

from .tiled_diffusion import (STEP_CACHE_KEYS, TilingUnsupported, _ramp_1d, ceil2, control_patches, pack,
                              unpack)

LOG = logging.getLogger("MiniMaxSuite.context_executor")
KILL_SWITCH = "MMX_CONTEXT_WINDOWS"
POSITIONS = ("global", "local (experimental)")
FPT = (1, 4, 4, 4, 4)
_PREFIX = (0, 1, 5, 9, 13)              # frames before step k % 5 inside a cycle
MAX_PLANS = 8


def disabled() -> bool:
    return os.environ.get(KILL_SWITCH, "").strip().lower() in ("off", "0", "false", "no", "disable", "disabled")


def frames_before(k: int) -> int:
    """Pixel frames before latent step k (FRAME_PER_TOKEN 1, 4, 4, 4, 4)."""
    k = int(k)
    return (k // 5) * 17 + _PREFIX[k % 5]


def steps_for_frames(n: int) -> int:
    """Latent steps of a legal (17k + 5) clip."""
    n = int(n)
    if n % 17 != 5:
        raise ValueError(f"{n} frames is not a legal H3 length (17k + 5)")
    return 2 + 5 * ((n - 5) // 17)


def snap_legal(n: int) -> int:
    n = max(5, int(n))
    return n + ((5 - n) % 17)


def tick_for_frame(f: int) -> int:
    """The 40 Hz audio tick at a 24 fps frame (exact when f % 3 == 0)."""
    return int(round(int(f) * 5 / 3))


def plan_windows(total_steps: int, window_frames: int, overlap_frames: int):
    """[(s, e)] latent-step windows covering [0, total_steps): starts on cycles, every window a legal clip, overlap at
    least `overlap_frames` (snapped to 17m + 5). Returns (windows, window_steps, overlap_steps)."""
    T = int(total_steps)
    lw = steps_for_frames(snap_legal(window_frames))
    ow = steps_for_frames(snap_legal(overlap_frames))
    if ow >= lw:
        raise ValueError(f"the overlap ({frames_before(ow)} frames) must be shorter than the window "
                         f"({frames_before(lw)} frames)")
    if T <= lw:
        return [(0, T)], lw, ow
    stride = lw - ow                                  # a multiple of 5: (2 + 5a) - (2 + 5b)
    starts = [0]
    while starts[-1] + lw < T:
        starts.append(starts[-1] + stride)
    # every start stays on the stride (exact audio edges on 51-frame strides); the last window is cut at the clip's
    # end, so it is shorter, never shifted, and still 2 + 5k steps long on a legal clip (more than the overlap)
    return [(s, min(s + lw, T)) for s in starts], lw, ow


def audio_ranges(windows, audio_t: int):
    """[(a0, a1)] tick range of each window; the first starts at 0, the last ends at the clip's last tick."""
    out = []
    for i, (s, e) in enumerate(windows):
        a0 = 0 if i == 0 else min(audio_t, tick_for_frame(frames_before(s)))
        a1 = audio_t if i == len(windows) - 1 else min(audio_t, tick_for_frame(frames_before(e)))
        out.append((a0, max(a0, a1)))
    return out


def axis_weights(ranges, i: int) -> torch.Tensor:
    """Trapezoid weight of range i over its overlaps with its neighbours (float32)."""
    a, b = ranges[i]
    left = ranges[i - 1][1] - a if i > 0 else 0
    right = b - ranges[i + 1][0] if i + 1 < len(ranges) else 0
    return _ramp_1d(b - a, max(0, left), max(0, right)).float()


def describe_settings(window_frames: int, overlap_frames: int, positions: str) -> str:
    wf, of = snap_legal(window_frames), snap_legal(overlap_frames)
    lw, ow = steps_for_frames(wf), steps_for_frames(of)
    if ow >= lw:
        raise ValueError(f"overlap_frames ({of} after snapping) must be shorter than window_frames ({wf} after "
                         "snapping) so the windows can advance.")
    stride = lw - ow
    exact = frames_before(stride) % 3 == 0 and wf % 3 == 0
    return (f"windows of {wf} frames ({lw} latent steps), overlap {of} frames, stride {frames_before(stride)} frames; "
            f"{positions} positions; audio edges "
            + ("exact" if exact else "within half a 1/40 s tick (exact when the stride is a multiple of 51 frames "
                                     "and the window a multiple of 3)")
            + f". A clip of {wf} frames or less runs untouched. Kill switch: {KILL_SWITCH}=off.")


class WindowLayoutT:
    """PackedLayout-compatible layout of one time window with the clip's own (global) rows: video steps [s, e), audio
    ticks [a0, a1) of both channels; text, references and keyframe conditioning unchanged."""

    def __init__(self, full, T, Hp, Wp, audio_t, s, e, a0, a1, text_len):
        frame_rows = (Hp // 2) * (Wp // 2)
        pos, img_pos, img_upd, aud_pos, aud_upd, segments = [], [], [], [], [], []
        row = 0
        for a, b, kind in full.segments:
            p = full.position_ids[a:b]
            if kind == "video":
                p = p.reshape(T, frame_rows, 3)[s:e].reshape(-1, 3)
            elif kind == "audio":
                p = torch.cat([p[a0:a1], p[audio_t + a0:audio_t + a1]])
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
        self.signature = (text_len, e - s, Hp, Wp, a1 - a0)
        self.window = (s, e, a0, a1)


def local_keyframes(keyframes, f0: int, f1: int):
    """Keyframes inside frames [f0, f1), re-indexed to the window (local positions)."""
    out = []
    for kf in keyframes or ():
        idx = int(kf.get("resolved_frame_index", 0))
        if f0 <= idx < f1:
            out.append(dict(kf, resolved_frame_index=idx - f0))
    return out


class H3ContextWindows:
    """model_function_wrapper (ModelPatcher.set_model_unet_function_wrapper) running every call as time windows."""

    def __init__(self, window_frames: int, overlap_frames: int, positions: str = POSITIONS[0], inner=None,
                 layout_factory: Callable | None = None):
        if positions not in POSITIONS:
            raise ValueError(f"positions must be one of {POSITIONS}")
        self.window_frames, self.overlap_frames = int(window_frames), int(overlap_frames)
        self.positions = positions
        self.inner = inner
        self.layout_factory = layout_factory
        self._plans: dict = {}
        self.last_report = "not run yet"

    def _factory(self):
        if self.layout_factory is None:
            from comfy.ldm.minimax.model import PackedLayout
            self.layout_factory = PackedLayout
        return self.layout_factory

    def plan(self, payload, video, audio_t, text_len):
        _, _, T, H, W = video.shape
        key = (id(payload), T, H, W, audio_t, text_len)
        hit = self._plans.get(key)
        if hit is not None and hit["payload"] is payload:
            return hit
        wins, lw, ow = plan_windows(T, self.window_frames, self.overlap_frames)
        aud = audio_ranges(wins, audio_t)
        Hp, Wp = ceil2(H), ceil2(W)
        keyframes, refs = payload.get("keyframes"), payload.get("refs")
        full = payload.get("layout")
        if full is None or getattr(full, "signature", None) != (text_len, T, Hp, Wp, audio_t):
            full = self._factory()(text_len, T, Hp, Wp, audio_t, keyframes=keyframes, refs=refs)
        payloads = []
        for (s, e), (a0, a1) in zip(wins, aud):
            p = dict(payload)
            if self.positions == POSITIONS[0]:
                p["layout"] = WindowLayoutT(full, T, Hp, Wp, audio_t, s, e, a0, a1, text_len)
            else:
                kfs = local_keyframes(keyframes, frames_before(s), frames_before(e))
                p["keyframes"] = kfs
                p["cond_video_latents"] = [k["latent"] for k in kfs if k.get("latent") is not None] + \
                    [r["latent"] for r in (refs or ()) if "latent" in r]
                p["cond_audio_latents"] = [k["audio_latent"] for k in kfs if k.get("audio_latent") is not None] + \
                    [r["audio_latent"] for r in (refs or ()) if r.get("audio_latent") is not None]
                p["layout"] = self._factory()(text_len, e - s, Hp, Wp, a1 - a0, keyframes=kfs, refs=refs)
            payloads.append(p)
        edges = [s for s, _e in wins[1:]] + [e for _s, e in wins[:-1]]
        exact = all(frames_before(k) % 3 == 0 for k in edges)
        plan = {"payload": payload, "windows": wins, "audio": aud, "payloads": payloads,
                "vw": [axis_weights(wins, i) for i in range(len(wins))],
                "aw": [axis_weights(aud, i) for i in range(len(aud))]}
        if len(self._plans) >= MAX_PLANS:
            self._plans.clear()
        self._plans[key] = plan
        self.last_report = (
            f"{len(wins)} window(s) of {frames_before(lw)} frames ({lw} latent steps), overlap >= "
            f"{frames_before(ow)} frames, on a {frames_before(T)}-frame clip, {self.positions} positions; audio "
            "windows " + ("exact" if exact else "within half a 1/40 s tick (window edges not on multiples of 3 frames)"))
        if T % 5 != 2:
            self.last_report += f"; the clip's {T} latent steps are not a legal H3 length, so the last window is not one"
        LOG.info("[H3 Context Windows] %s", self.last_report)
        return plan

    def _call(self, fn, x, t, c, cou):
        if self.inner is not None:
            return self.inner(fn, {"input": x, "timestep": t, "c": c, "cond_or_uncond": cou})
        return fn(x, t, **c)

    def __call__(self, model_function, args):
        x, t, c = args["input"], args["timestep"], args["c"]
        cou = args.get("cond_or_uncond")
        if disabled():
            return self._call(model_function, x, t, c, cou)
        payload = c.get("minimax_payload")
        if payload is None:
            raise TilingUnsupported("H3 Context Window Sampling only works with the MiniMax H3 model.")
        if c.get("control") is not None:
            raise TilingUnsupported("H3 Context Window Sampling cannot cut a ControlNet that rides in the "
                                    "conditioning. Use core's 'Apply MiniMax H3 Fun ControlNet' (a model patch).")
        opts = c.get("transformer_options") or {}
        for k in STEP_CACHE_KEYS:
            if opts.get(k):
                raise TilingUnsupported("H3 Context Window Sampling cannot run with a block / step cache: the cache "
                                        "would take the windows of one step for consecutive steps.")
        shapes = c.get("latent_shapes")
        streams = unpack(x, shapes)
        if len(streams) < 2 or streams[0].ndim != 5:
            raise TilingUnsupported("H3 Context Window Sampling needs the joint H3 video + audio latent.")
        video, audio = streams[0], streams[1]
        if video.shape[0] != 1:
            raise TilingUnsupported("MiniMax H3 runs batch size 1.")
        text = c.get("c_crossattn")
        plan = self.plan(payload, video, int(audio.shape[-1]), int(text.shape[1]) if text is not None else 0)
        if len(plan["windows"]) == 1:
            return self._call(model_function, x, t, c, cou)

        _, _, T, H, W = video.shape
        At = int(audio.shape[-1])
        controls = control_patches(c, "H3 Context Window Sampling", "time windows")
        sigmas = opts.get("sigmas")
        sigma = float(sigmas.flatten()[0]) if sigmas is not None else float(t.flatten()[0]) / 1000.0
        hints = []
        for patch in controls:
            active = patch.sigma_end <= sigma <= patch.sigma_start
            if active:
                patch.prepare_control_latent(tuple(video.shape))
            hints.append((active, patch.control_latent, patch.control_latent_shape))
        vmask, amask = c.get("denoise_mask"), c.get("audio_denoise_mask")
        vbuf = torch.zeros(video.shape, dtype=torch.float32, device=video.device)
        vsum = torch.zeros((1, 1, T, 1, 1), dtype=torch.float32, device=video.device)
        abuf = torch.zeros(audio.shape, dtype=torch.float32, device=audio.device)
        asum = torch.zeros((1, 1, 1, At), dtype=torch.float32, device=audio.device)
        try:
            for (s, e), (a0, a1), p, vw, aw in zip(plan["windows"], plan["audio"], plan["payloads"], plan["vw"],
                                                   plan["aw"]):
                c_w = dict(c)
                c_w["minimax_payload"] = p
                if vmask is not None and vmask.ndim == 5 and vmask.shape[2] == T:
                    c_w["denoise_mask"] = vmask[:, :, s:e]           # a one-frame mask broadcasts: left whole
                if amask is not None and amask.shape[-1] == At:
                    c_w["audio_denoise_mask"] = amask[..., a0:a1]
                v_w, a_w = video[:, :, s:e], audio[..., a0:a1]
                for patch, (active, hint, _shape) in zip(controls, hints):
                    if active and hint is not None:
                        patch.control_latent = hint[:, :, s:e]
                        patch.control_latent_shape = tuple(v_w.shape)
                x_w, shapes_w = pack([v_w, a_w])
                c_w["latent_shapes"] = shapes_w
                ov, oa = unpack(self._call(model_function, x_w, t, c_w, cou), shapes_w)[:2]
                vwd = vw.to(video.device).view(1, 1, -1, 1, 1)
                awd = aw.to(audio.device).view(1, 1, 1, -1)
                vbuf[:, :, s:e] += ov.float() * vwd
                vsum[:, :, s:e] += vwd
                abuf[..., a0:a1] += oa.float() * awd
                asum[..., a0:a1] += awd
        finally:
            for patch, (_active, hint, shape) in zip(controls, hints):
                patch.control_latent, patch.control_latent_shape = hint, shape
        out_v = (vbuf / vsum.clamp(min=1e-12)).to(video.dtype)
        out_a = (abuf / asum.clamp(min=1e-12)).to(audio.dtype)
        return pack([out_v, out_a])[0]
