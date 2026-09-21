"""Split / tiled latent upscaling nodes for MiniMax H3.

PORTED FROM: Comfyui_Minimax_h3_latent_Upscaler (author LBH-123-AI, MIT License).
This is a direct port of the upstream split-upscale sampler, not a clean-room
reimplementation. Pure maths lives in mmx_utils/split_upscale.py; temporal grid
helpers are in mmx_utils/h3_grid.py.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

import comfy.model_management
import comfy.sample
import comfy.samplers
import comfy.utils
import latent_preview
from comfy_api.latest import io

_PKG = Path(__file__).resolve().parents[1]
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from mmx_utils.h3_grid import (  # noqa: E402
    audio_range,
    compute_h3_segments_adaptive,
    frames_for_tokens,
    tokens_for_frames,
)
from mmx_utils.split_upscale import (  # noqa: E402
    POLISH_HALO,
    build_spatial_param,
    build_temporal_param,
    compute_spatial_grid,
    dc_correct,
    grade_pin,
    identity_keyframes,
    merge_spatial_tile,
    motion_keyframes,
    refuse_unless_h3_av_latent,
    should_polish_seam,
    spatial_fade_mask,
    spatial_param_report,
    split_upscale_report,
    temporal_append,
    temporal_param_report,
    trim_keyframe,
    unpack_av_samples,
)

_LOG = logging.getLogger(__name__)

CATEGORY = "MiniMax H3/Sampling"

H3_TEMPORAL_SPLIT = io.Custom("H3_TEMPORAL_SPLIT_PARAM")
H3_SPATIAL_SPLIT = io.Custom("H3_SPATIAL_SPLIT_PARAM")

_H3_MISSING = (
    "MiniMax H3 support is missing from this ComfyUI build: comfy.ldm.minimax "
    "does not import ({err}). AV latents are NestedTensors defined there, so "
    "this node cannot run without it."
)


def _nested_tensor_factory():
    try:
        import comfy.nested_tensor as nested_tensor
        return nested_tensor.NestedTensor
    except Exception:
        try:
            from comfy.ldm.minimax.model import NestedTensor  # type: ignore
            return NestedTensor
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(_H3_MISSING.format(err=exc)) from exc


# ── conditioning helpers ────────────────────────────────────────────────────


def reanchor_conditioning(cond, f0, f1, spatial):
    out = []
    for tensor, d in cond:
        nd = dict(d)
        kfs = nd.get("minimax_keyframes")
        if kfs:
            trimmed = [
                kf for kf in (trim_keyframe(kf, f0, f1) for kf in kfs) if kf is not None
            ]
            if trimmed:
                if spatial is not None:
                    for kf in trimmed:
                        lt = kf.get("latent")
                        if lt is not None and (lt.shape[3] != spatial[0] or lt.shape[4] != spatial[1]):
                            b, c, t, h, w = lt.shape
                            kf["latent"] = F.interpolate(
                                lt.view(b * t, c, h, w),
                                size=spatial,
                                mode="bilinear",
                                align_corners=False,
                            ).view(b, c, t, spatial[0], spatial[1])
                nd["minimax_keyframes"] = trimmed
            else:
                nd.pop("minimax_keyframes", None)
        out.append([tensor, nd])
    return out


def prepend_keyframes(cond, kfs):
    if not kfs:
        return cond
    out = []
    for tensor, d in cond:
        nd = dict(d)
        nd["minimax_keyframes"] = kfs + (nd.get("minimax_keyframes") or [])
        out.append([tensor, nd])
    return out


def anchor_conditioning(cond, prev_video, f0, strength):
    t = tokens_for_frames(f0)
    if t >= prev_video.shape[2]:
        raise ValueError("previous result does not reach the current segment start")
    anchor_kf = {"resolved_frame_index": 0, "latent": prev_video[:, :, t:t + 1].contiguous()}
    aug = max(0.0, min(1.0, float(strength)))
    out = []
    for tensor, d in cond:
        nd = dict(d)
        kfs = nd.get("minimax_keyframes")
        if kfs:
            kept = [
                kf for kf in kfs
                if kf.get("resolved_frame_index") != 0 or "latent" not in kf
            ]
            nd["minimax_keyframes"] = [anchor_kf] + kept
        else:
            nd["minimax_keyframes"] = [anchor_kf]
        nd["minimax_visual_cond_noise_aug"] = aug
        out.append([tensor, nd])
    return out


def crop_keyframes_to_tile(cond, src_h, src_w, r0, c0, tr, tc):
    out = []
    for tensor, d in cond:
        nd = dict(d)
        kfs = nd.get("minimax_keyframes")
        if kfs:
            cropped = []
            for kf in kfs:
                nkf = dict(kf)
                lt = kf.get("latent")
                if lt is not None:
                    if lt.shape[3] == src_h and lt.shape[4] == src_w:
                        nkf["latent"] = lt[:, :, :, r0:r0 + tr, c0:c0 + tc].contiguous()
                    else:
                        lt_r = F.interpolate(
                            lt.to(torch.float32),
                            size=(src_h, src_w),
                            mode="bilinear",
                            align_corners=False,
                        )
                        nkf["latent"] = lt_r[:, :, :, r0:r0 + tr, c0:c0 + tc].contiguous()
                cropped.append(nkf)
            nd["minimax_keyframes"] = cropped
        out.append([tensor, nd])
    return out


# ── sampling helpers ────────────────────────────────────────────────────────


def build_guider(model, cond, negative, cfg):
    guider = comfy.samplers.CFGGuider(model)
    if negative is not None:
        guider.set_conds(cond, negative)
        guider.set_cfg(cfg)
    else:
        guider.inner_set_conds({"positive": cond})
    return guider


def sample_piece(piece, guider, noise_tensor, seed, sampler, sigmas, callback=None):
    latent = dict(piece)
    latent_image = latent["samples"]
    latent_image = comfy.sample.fix_empty_latent_channels(
        model=guider.model_patcher.model,
        latent_image=latent_image,
    )
    latent["samples"] = latent_image
    if callback is None:
        x0_output = {}
        callback = latent_preview.prepare_callback(
            guider.model_patcher, sigmas.shape[-1] - 1, x0_output,
        )
    samples = guider.sample(
        noise_tensor,
        latent_image,
        sampler,
        sigmas,
        denoise_mask=latent.get("noise_mask"),
        callback=callback,
        disable_pbar=not comfy.utils.PROGRESS_BAR_ENABLED,
        seed=seed,
    )
    return samples.to(comfy.model_management.intermediate_device())


def make_tile_progress(model_patcher, steps, n_tiles):
    previewer = latent_preview.get_previewer(
        model_patcher.load_device, model_patcher.model.latent_format,
    )
    total = steps * n_tiles
    pbar = comfy.utils.ProgressBar(total)

    def for_tile(idx):
        def callback(step, x0, x, total_steps):
            preview = None
            if previewer is not None and x0 is not None:
                px0 = x0.tensors[0] if getattr(x0, "is_nested", False) else x0
                try:
                    preview = previewer.decode_latent_to_preview_image("JPEG", px0)
                except Exception:  # noqa: BLE001
                    preview = None
            pbar.update_absolute(idx * steps + step + 1, total, preview)

        return callback

    return for_tile


# ── parameter nodes ─────────────────────────────────────────────────────────


class MiniMaxH3_TemporalSplitParams(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3_TemporalSplitParams",
            display_name="H3 Temporal Split Params",
            category=CATEGORY,
            description=(
                "Temporal chunking on H3's native grid (5-frame head + 17-frame "
                "cycles). Feeds Split Upscale with legal segment sizes, overlap "
                "and triple-anchor settings."
            ),
            inputs=[
                io.Int.Input(
                    "chunk_frames", default=73, min=5, max=100_000, step=1,
                    tooltip="Frames per temporal pass. Off-grid values snap to "
                            "17n+5 — a wrong length produces a temporal POP at "
                            "the chunk boundary because the latent rows no longer "
                            "align with the model's token clock."),
                io.Int.Input(
                    "temporal_overlap_frames", default=22, min=0, max=100_000, step=1,
                    tooltip="Shared frames between chunks — what gets cross-faded "
                            "and colour-matched. Too little looks like a hard cut; "
                            "too much doubles render time. Snapped to the H3 grid."),
                io.Float.Input(
                    "anchor_strength", default=0.999, min=0.0, max=1.0, step=0.001,
                    tooltip="How tightly the new chunk is pinned to the previous "
                            "segment's last latent row. Low values let the seam "
                            "drift and look like two different takes spliced together."),
                io.Combo.Input(
                    "motion_anchor_frames",
                    options=["0", "5", "22", "39"],
                    default="22",
                    tooltip="Motion keyframes borrowed from the accumulated result "
                            "to carry fast movement across the seam. 0 disables; "
                            "22 is one H3 cycle — use when limbs cross the boundary."),
                io.Int.Input(
                    "identity_anchor_frames", default=24, min=0, max=240, step=1,
                    tooltip="Spacing of identity anchors sampled from the ORIGINAL "
                            "latent inside each chunk. Stops faces or wardrobe "
                            "from slowly morphing across a long upscale. 0 disables."),
            ],
            outputs=[
                H3_TEMPORAL_SPLIT.Output(display_name="temporal_split_param"),
                io.String.Output(display_name="report"),
            ],
        )

    @classmethod
    def execute(
        cls,
        chunk_frames,
        temporal_overlap_frames,
        anchor_strength,
        motion_anchor_frames,
        identity_anchor_frames,
    ) -> io.NodeOutput:
        param = build_temporal_param(
            chunk_frames,
            temporal_overlap_frames,
            anchor_strength,
            int(motion_anchor_frames),
            identity_anchor_frames,
        )
        # Preview report assumes a typical 124-frame / 37-row clip; actual Split
        # Upscale recomputes segments from the connected latent.
        report = temporal_param_report(param, 37)
        return io.NodeOutput(param, report)


class MiniMaxH3_SpatialSplitParams(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3_SpatialSplitParams",
            display_name="H3 Spatial Split Params",
            category=CATEGORY,
            description=(
                "Spatial tiling for Split Upscale: tile size, overlap, fade and "
                "seam-denoise cap. Converts pixel sizes to latent cells (÷16, "
                "aligned to 2)."
            ),
            inputs=[
                io.Int.Input(
                    "tile_width", default=512, min=64, max=16_384, step=32,
                    tooltip="Tile width in PIXELS. Too small and each tile has no "
                            "context — edges look like postage stamps. Too large "
                            "and VRAM blows up before you see a seam."),
                io.Int.Input(
                    "tile_height", default=512, min=64, max=16_384, step=32,
                    tooltip="Tile height in pixels — same trade-off as width."),
                io.Float.Input(
                    "overlap_ratio", default=0.25, min=0.0, max=0.90, step=0.05,
                    tooltip="Fraction of each tile that overlaps its neighbour. "
                            "Zero overlap is fastest but shows a hard grid; 25% "
                            "is a typical compositor default. Above ~50% you pay "
                            "for tiles you mostly throw away."),
                io.Float.Input(
                    "fade_ratio", default=0.50, min=0.0, max=1.0, step=0.05,
                    tooltip="How much of the overlap is a gradual freeze-to-free "
                            "ramp vs a hard frozen band. Too little fade leaves a "
                            "visible step in denoise strength — a brightness ridge "
                            "along the seam."),
                io.Int.Input(
                    "min_tile_size", default=256, min=0, max=16_384, step=32,
                    tooltip="Smallest edge tile allowed in pixels. Stops the "
                            "planner from leaving a skinny sliver at the frame "
                            "edge that has almost no context and blooms differently."),
                io.Float.Input(
                    "seam_denoise", default=1.0, min=0.1, max=1.0, step=0.05,
                    tooltip="Cap on denoise in the seam neighbourhood. Below 1.0 "
                            "lets the sampler 'continue' frozen neighbour content at "
                            "moderate strength — stops fast motion being cut in half "
                            "at the tile edge under high denoise. 1.0 = classic "
                            "behaviour (off). Try 0.5–0.8 for sports or hair."),
            ],
            outputs=[
                H3_SPATIAL_SPLIT.Output(display_name="spatial_split_param"),
                io.String.Output(display_name="report"),
            ],
        )

    @classmethod
    def execute(
        cls,
        tile_width,
        tile_height,
        overlap_ratio,
        fade_ratio,
        min_tile_size,
        seam_denoise,
    ) -> io.NodeOutput:
        param = build_spatial_param(
            tile_width, tile_height, overlap_ratio, fade_ratio, min_tile_size, seam_denoise,
        )
        # Grid preview uses a representative 64x64 latent canvas; Split Upscale
        # recomputes from the real latent H×W.
        report = spatial_param_report(param, h=64, w=64)
        return io.NodeOutput(param, report)


class MiniMaxH3_SplitUpscale(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3_SplitUpscale",
            display_name="H3 Split Upscale",
            category=CATEGORY,
            description=(
                "Upscale a MiniMax H3 AV latent by tiling it in time and/or "
                "space, sampling each tile with frozen overlap bands, colour "
                "matching and optional seam repolish.\n\n"
                "Prevention: pre-fill overlap from neighbours + triple temporal "
                "anchors. Correction: spatial/temporal DC match + per-chunk grade "
                "pin. Anti-fork: seam_denoise cap on the freeze mask."
            ),
            inputs=[
                io.Model.Input("model"),
                io.Conditioning.Input("conditioning"),
                io.Conditioning.Input("negative", optional=True),
                io.Latent.Input("latent"),
                io.Noise.Input("noise"),
                io.Sampler.Input("sampler"),
                io.Sigmas.Input("sigmas"),
                io.Float.Input("cfg", default=1.0, min=0.0, max=100.0, step=0.1, round=0.01),
                H3_TEMPORAL_SPLIT.Input("temporal_split_param", optional=True),
                H3_SPATIAL_SPLIT.Input("spatial_split_param", optional=True),
                io.Combo.Input(
                    "seam_polish", options=["off", "auto", "all"], default="off",
                    tooltip="Second-pass resample on seams that fail a colour/"
                            "correlation probe. 'auto' only polishes bad seams; "
                            "'all' is slow but catches subtle colour steps. 'off' "
                            "if you will fix seams in comp."),
                io.Boolean.Input(
                    "color_match", default=True,
                    tooltip="Median DC match on overlaps and per-chunk grade pin "
                            "to the source. Off saves a little time but seams "
                            "often show a visible colour step between tiles."),
            ],
            outputs=[
                io.Latent.Output(display_name="latent"),
                io.String.Output(display_name="report"),
            ],
        )

    @classmethod
    def execute(
        cls,
        latent,
        conditioning,
        model,
        noise,
        sampler,
        sigmas,
        negative=None,
        cfg=1.0,
        temporal_split_param=None,
        spatial_split_param=None,
        seam_polish="off",
        color_match=True,
    ) -> io.NodeOutput:
        NestedTensor = _nested_tensor_factory()
        samples = latent["samples"]
        refuse_unless_h3_av_latent(samples)
        video, audio = unpack_av_samples(samples)
        if video.shape[0] != 1:
            raise ValueError("Split Upscale supports batch size 1 only.")
        _, _, t, h, w = video.shape

        if temporal_split_param is not None:
            tp = temporal_split_param.get("p", temporal_split_param)
            if isinstance(tp, dict):
                cl = tp["chunk_frames"]
                ov = tp["overlap_frames"]
                anchor_strength = tp["anchor_strength"]
                motion_n = tp["motion_anchor_frames"]
                identity_n = tp["identity_anchor_frames"]
            else:
                cl, ov, anchor_strength, motion_n, identity_n = tp
            bounds, _ = compute_h3_segments_adaptive(t, cl, ov)
        else:
            bounds = [(0, 0, t, frames_for_tokens(t))]
            anchor_strength, motion_n, identity_n = 0.999, 0, 0

        if spatial_split_param is not None:
            sp = spatial_split_param
            rows, cols, trows, tcols, row_ovl, col_ovl = compute_spatial_grid(
                h, w, sp["th"], sp["tw"], sp["ol_h"], sp["ol_w"], sp["mt"], sp["mt"],
            )
            seam_cap = sp.get("cap", 1.0)
        else:
            rows, cols, trows, tcols, row_ovl, col_ovl = [0], [0], [h], [w], [0], [0]
            seam_cap = 1.0
        nrows, ncols = len(rows), len(cols)

        steps = max(int(sigmas.shape[-1]) - 1, 1)
        n_tiles = len(bounds) * nrows * ncols
        patcher = model.model_patcher if hasattr(model, "model_patcher") else model
        for_tile = make_tile_progress(patcher, steps, n_tiles)

        source = video
        noise_v = noise.generate_noise({"samples": torch.zeros_like(video, dtype=torch.float32)})
        noise_a = noise.generate_noise({"samples": torch.zeros_like(audio, dtype=torch.float32)})

        acc_v = acc_a = None
        polish_queue: dict = {}
        tile_idx = 0
        polish_count = 0

        for i, (k0, f0, k1, f1) in enumerate(bounds):
            chunk_v = video[:, :, k0:k1].contiguous()
            a0, a1 = audio_range(f0, f1)
            a1 = min(a1, audio.shape[-1])
            chunk_a = audio[:, :, :, a0:a1].contiguous()

            cond_i = reanchor_conditioning(conditioning, f0, f1, (h, w))
            if i > 0 and acc_v is not None:
                if motion_n > 0:
                    cond_i = prepend_keyframes(
                        cond_i, motion_keyframes(acc_v, k0, f0, motion_n),
                    )
                if identity_n > 0:
                    cond_i = prepend_keyframes(
                        cond_i, identity_keyframes(source, f0, f1, identity_n),
                    )
                if anchor_strength > 0.0:
                    cond_i = anchor_conditioning(cond_i, acc_v, f0, anchor_strength)
            elif identity_n > 0:
                cond_i = prepend_keyframes(
                    cond_i, identity_keyframes(source, f0, f1, identity_n),
                )

            chunk_out = chunk_v.clone()
            noise_vc = noise_v[:, :, k0:k1]
            noise_ac = noise_a[:, :, :, a0:a1]

            for ri in range(nrows):
                for cj in range(ncols):
                    comfy.model_management.throw_exception_if_processing_interrupted()
                    r0, c0 = rows[ri], cols[cj]
                    tr, tc = trows[ri], tcols[cj]
                    ovh, ovw = row_ovl[ri], col_ovl[cj]

                    tile = chunk_out[:, :, :, r0:r0 + tr, c0:c0 + tc].clone()

                    if spatial_split_param is not None:
                        fh, fw = spatial_split_param["fh"], spatial_split_param["fw"]
                    else:
                        fh = fw = 0
                    m = spatial_fade_mask(
                        tr, tc, ovh, ovw,
                        done_top=(ri > 0), done_left=(cj > 0),
                        fade_h=fh, fade_w=fw, seam_cap=seam_cap,
                    )
                    mv = m[None, None, None].to(chunk_out.device)
                    ma = torch.zeros(
                        (1, 32, 2, chunk_a.shape[-1]),
                        device=chunk_a.device, dtype=torch.float32,
                    )
                    piece = {
                        "samples": NestedTensor((tile, chunk_a)),
                        "noise_mask": NestedTensor((mv, ma)),
                    }
                    tile_noise = NestedTensor((
                        noise_vc[:, :, :, r0:r0 + tr, c0:c0 + tc].contiguous(),
                        noise_ac.contiguous(),
                    ))

                    cond_tile = crop_keyframes_to_tile(cond_i, h, w, r0, c0, tr, tc)
                    guider = build_guider(model, cond_tile, negative, cfg)
                    out = sample_piece(
                        piece, guider, tile_noise, noise.seed, sampler, sigmas,
                        for_tile(tile_idx),
                    )
                    tile_v = (out.tensors[0] if out.is_nested else out).to(chunk_out.device)

                    region = chunk_out[:, :, :, r0:r0 + tr, c0:c0 + tc].clone()

                    if color_match:
                        tile_v, _ = dc_correct(tile_v, [
                            (tile_v[:, :, :, :, :ovw], region[:, :, :, :, :ovw])
                            if (cj > 0 and ovw > 0) else None,
                            (tile_v[:, :, :, :ovh, :], region[:, :, :, :ovh, :])
                            if (ri > 0 and ovh > 0) else None,
                            (tile_v, chunk_v[:, :, :, r0:r0 + tr, c0:c0 + tc])
                            if (ri == 0 and cj == 0) else None,
                        ])

                    if seam_polish != "off":
                        if cj > 0 and ovw > 0 and should_polish_seam(
                            seam_polish, tile_v[:, :, :, :, :ovw], region[:, :, :, :, :ovw],
                        ):
                            polish_queue[(i, ri, cj, "W")] = (c0, ovw, (k0, k1))
                        if ri > 0 and ovh > 0 and should_polish_seam(
                            seam_polish, tile_v[:, :, :, :ovh, :], region[:, :, :, :ovh, :],
                        ):
                            polish_queue[(i, ri, cj, "H")] = (r0, ovh, (k0, k1))

                    chunk_out = merge_spatial_tile(
                        chunk_out, tile_v, r0, c0, tr, tc, ovh, ovw, ri, cj,
                    )

                    tile_idx += 1
                    comfy.model_management.soft_empty_cache()

            if color_match:
                chunk_out, dcg = grade_pin(chunk_out, chunk_v)
                if dcg is not None and dcg.abs().max().item() > 1e-4:
                    _LOG.info(
                        "Split Upscale chunk %d grade pin |dc|max=%.4f",
                        i, dcg.abs().max().item(),
                    )

            chunk_polish = [(k, v) for k, v in polish_queue.items() if k[0] == i]
            if chunk_polish:
                polish_count += len(chunk_polish)
                _LOG.info("Split Upscale chunk %d: polishing %d seam(s)", i, len(chunk_polish))
                pbar2 = comfy.utils.ProgressBar(steps * len(chunk_polish))
                for pi, (key, (s0, band_w, (tk0, tk1))) in enumerate(chunk_polish):
                    comfy.model_management.throw_exception_if_processing_interrupted()
                    axis = key[3]
                    t_len = tk1 - tk0
                    if axis == "W":
                        w0 = max(0, s0 - POLISH_HALO)
                        w1 = min(w, s0 + band_w + POLISH_HALO)
                        win = chunk_out[:, :, :, :, w0:w1].clone()
                        b0, b1 = s0 - w0, s0 - w0 + band_w
                        mv = torch.zeros(
                            (1, 1, t_len, h, w1 - w0), dtype=torch.float32, device=win.device,
                        )
                        mv[:, :, :, :, b0:b1] = 1.0
                        r0c, c0c, trc, tcc, ax = 0, w0, h, w1 - w0, 4
                        nsl = (slice(None), slice(None), slice(tk0, tk1), slice(None), slice(w0, w1))
                    else:
                        h0 = max(0, s0 - POLISH_HALO)
                        h1 = min(h, s0 + band_w + POLISH_HALO)
                        win = chunk_out[:, :, :, h0:h1, :].clone()
                        b0, b1 = s0 - h0, s0 - h0 + band_w
                        mv = torch.zeros(
                            (1, 1, t_len, h1 - h0, w), dtype=torch.float32, device=win.device,
                        )
                        mv[:, :, :, b0:b1, :] = 1.0
                        r0c, c0c, trc, tcc, ax = h0, 0, h1 - h0, w, 3
                        nsl = (slice(None), slice(None), slice(tk0, tk1), slice(h0, h1), slice(None))
                    ma = torch.zeros(
                        (1, 32, 2, chunk_a.shape[-1]),
                        device=chunk_a.device, dtype=torch.float32,
                    )
                    piece = {
                        "samples": NestedTensor((win, chunk_a)),
                        "noise_mask": NestedTensor((mv, ma)),
                    }
                    tn = NestedTensor((noise_v[nsl].contiguous(), noise_ac.contiguous()))

                    cond_p = crop_keyframes_to_tile(cond_i, h, w, r0c, c0c, trc, tcc)
                    g = build_guider(model, cond_p, negative, cfg)

                    def _cb(step, x0, x, ts, _pi=pi):
                        pbar2.update_absolute(_pi * steps + step + 1, steps * len(chunk_polish))

                    out = sample_piece(piece, g, tn, noise.seed, sampler, sigmas, callback=_cb)
                    pv = (out.tensors[0] if out.is_nested else out).to(chunk_out.device)

                    nlen = (w1 - w0) if ax == 4 else (h1 - h0)
                    alpha = torch.ones(nlen, dtype=torch.float32)
                    if b0 > 0:
                        alpha[:b0] = torch.linspace(0.0, 1.0, b0)
                    if nlen - b1 > 0:
                        alpha[b1:] = torch.linspace(1.0, 0.0, nlen - b1)
                    view = [1, 1, 1, 1, 1]
                    view[ax] = nlen
                    avv = alpha.view(view).to(chunk_out.device)
                    blended = avv * pv + (1.0 - avv) * win
                    if ax == 4:
                        chunk_out = chunk_out.clone()
                        chunk_out[:, :, :, :, w0:w1] = blended
                    else:
                        chunk_out = chunk_out.clone()
                        chunk_out[:, :, :, h0:h1, :] = blended
                    comfy.model_management.soft_empty_cache()

            acc_v, acc_a = temporal_append(
                acc_v, acc_a, chunk_out, chunk_a, i, k0, f0, color_match=color_match,
            )

        if hasattr(model, "clone_base_uuid"):
            comfy.model_management.unload_model_and_clones(model, unload_additional_models=False)
            comfy.model_management.soft_empty_cache()

        report = split_upscale_report(
            bounds=bounds,
            nrows=nrows,
            ncols=ncols,
            row_ovl=row_ovl,
            col_ovl=col_ovl,
            seam_polish=seam_polish,
            polish_count=polish_count,
            color_match=color_match,
        )
        return io.NodeOutput(
            {"samples": NestedTensor((acc_v, acc_a))},
            report,
        )
