"""Latent colour anchor — remove per-step temporal colour drift on preserved regions (N1).

Pure torch logic; ComfyUI pack/unpack helpers are injected by the node or tests.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Sequence

import torch

_LOG = logging.getLogger(__name__)


def _is_nested(x: Any) -> bool:
    return getattr(x, "is_nested", False)


def _unbind(x: Any) -> list[torch.Tensor]:
    if _is_nested(x):
        return list(x.unbind())
    return [x]


def build_preserved_packed(
    noise_mask: Any,
    latent_shapes: Sequence[tuple[int, ...]],
    prepare_mask_fn: Callable[..., torch.Tensor],
    pack_fn: Callable[..., tuple[torch.Tensor, list]],
    device: torch.device,
) -> torch.Tensor:
    """Build a packed preserved mask (1 = preserved) mirroring ``CFGGuider.sample``.

    Missing per-stream masks default to ones (fully denoised, not preserved).
    ``prepare_mask`` is applied per stream; results are packed when there is more
    than one stream.  Preserved = 1 - denoise_mask.
    """
    shapes = list(latent_shapes)
    if _is_nested(noise_mask):
        denoise_masks = list(noise_mask.unbind())[: len(shapes)]
    else:
        denoise_masks = [noise_mask]

    for i in range(len(denoise_masks), len(shapes)):
        denoise_masks.append(torch.ones(shapes[i], device=device))

    for i in range(len(denoise_masks)):
        denoise_masks[i] = prepare_mask_fn(denoise_masks[i], shapes[i], device)

    if len(denoise_masks) > 1:
        denoise_mask, _ = pack_fn(denoise_masks)
    else:
        denoise_mask = denoise_masks[0]

    return (1.0 - denoise_mask.float()).to(device=device)


def split_video(
    packed: torch.Tensor,
    latent_shapes: Sequence[tuple[int, ...]],
    unpack_fn: Callable[..., list[torch.Tensor]],
    pack_fn: Callable[..., tuple[torch.Tensor, list]],
) -> tuple[torch.Tensor, Callable[[torch.Tensor], torch.Tensor]]:
    """Unpack stream 0 as ``[B, C, T, h, w]`` and return a writer for ``packed``."""
    shapes = list(latent_shapes)
    tensors = list(unpack_fn(packed, shapes))
    target = tuple(shapes[0])
    video = tensors[0]
    if tuple(video.shape) != target:
        video = video.reshape(target)
    video = video.clone()
    tail = tensors[1:]
    packed_shape = tuple(packed.shape)
    packed_dtype = packed.dtype

    def write_back(new_video: torch.Tensor) -> torch.Tensor:
        nv = new_video
        if tuple(nv.shape) != target:
            nv = nv.reshape(target)
        if len(shapes) == 1:
            return nv.reshape(packed_shape).to(dtype=packed_dtype)
        streams = [nv] + [t for t in tail]
        repacked, _ = pack_fn(streams)
        return repacked.to(dtype=packed_dtype)

    return video, write_back


# A cell is EVIDENCE only if it is preserved at every step. `preserved > 0.5`
# counted any cell with a mask value under 0.5, and under differential
# diffusion (MVEx_DifferentialDiffusionSoft, MiniMaxH3_DifferentialDenoise) a
# cell at 0.3 starts being EDITED about 70% of the way through the schedule -
# while this runs to 85% by default. For that stretch the model's intended
# edits would have been measured as drift and subtracted. Only cells that
# stay kept throughout can say what "unchanged" looks like.
_EVIDENCE_PRESERVED = 0.98


def anchor_video(
    den_video: torch.Tensor,
    src_video: torch.Tensor,
    preserved_video: torch.Tensor,
    strength: float,
    mode: str,
    per_frame: bool,
    min_preserved: float,
) -> tuple[torch.Tensor, list[float]]:
    """Correct colour drift on the video latent; return (corrected, per-frame stats).

    Per frame *t* (or all *T* when ``per_frame`` is False) and channel *c*, over
    preserved cells: ``d = mean(src) - mean(den)``.  ``mean_std`` also rescales spread
    about the mean by ``std(src)/std(den)`` clamped to [0.5, 2.0], lerped by
    ``strength``.  The correction is applied to every cell of that frame.  Frames whose
    preserved fraction is below ``min_preserved`` are skipped.  ``stats`` holds the
    per-frame mean ``|d|`` (mean over channels).
    """
    den = den_video.float()
    src = src_video.float()
    pres = preserved_video.float()
    out = den.clone()
    stats: list[float] = []

    b, c, t, _, _ = den.shape
    strength_f = float(strength)

    def _apply_slice(
        den_sl: torch.Tensor,
        src_sl: torch.Tensor,
        pres_sl: torch.Tensor,
        out_sl: torch.Tensor,
    ) -> float:
        pres_frac = (pres_sl > _EVIDENCE_PRESERVED).float().mean().item()
        if pres_frac < min_preserved:
            return 0.0

        drift_mag = 0.0
        for ch in range(c):
            den_c = den_sl[:, ch]
            src_c = src_sl[:, ch]
            pres_c = pres_sl[:, ch] > _EVIDENCE_PRESERVED
            if not pres_c.any():
                continue

            den_vals = den_c[pres_c]
            src_vals = src_c[pres_c]
            mean_den = den_vals.mean()
            mean_src = src_vals.mean()
            d = mean_src - mean_den
            drift_mag += float(d.abs().item())

            if mode == "mean_std":
                std_den = den_vals.std(unbiased=False).clamp(min=1e-8)
                std_src = src_vals.std(unbiased=False).clamp(min=1e-8)
                ratio = (std_src / std_den).clamp(0.5, 2.0)
                target = (den_c - mean_den) * ratio + mean_src
                out_sl[:, ch] = den_c + strength_f * (target - den_c)
            else:
                out_sl[:, ch] = den_c + strength_f * d

        return drift_mag / max(c, 1)

    if per_frame:
        for ti in range(t):
            mag = _apply_slice(
                den[:, :, ti],
                src[:, :, ti],
                pres[:, :, ti],
                out[:, :, ti],
            )
            stats.append(mag)
    else:
        mag = _apply_slice(den, src, pres, out)
        stats.extend([mag] * t)

    return out.to(dtype=den_video.dtype), stats


def _pack_source(
    source: Any,
    pack_fn: Callable[..., tuple[torch.Tensor, list]],
) -> torch.Tensor:
    if _is_nested(source):
        return pack_fn(list(source.unbind()))[0]
    if isinstance(source, torch.Tensor) and source.ndim == 5:
        return pack_fn([source])[0]
    return source


def _sigma_active(
    sigma: torch.Tensor,
    model: Any,
    start_percent: float,
    end_percent: float,
) -> bool:
    sampling = model.model_sampling
    sigma_hi = sampling.percent_to_sigma(start_percent)
    sigma_lo = sampling.percent_to_sigma(end_percent)
    sigma_v = float(sigma[0].item())
    return sigma_lo <= sigma_v <= sigma_hi


def make_hook(
    source_packed_or_nested: Any,
    noise_mask: Any,
    strength: float,
    mode: str,
    per_frame: bool,
    min_preserved: float,
    start_percent: float,
    end_percent: float,
    prepare_mask_fn: Callable[..., torch.Tensor],
    pack_fn: Callable[..., tuple[torch.Tensor, list]],
    unpack_fn: Callable[..., list[torch.Tensor]],
    shapes_hint: Sequence[tuple[int, ...]] | None = None,
    logger: logging.Logger | None = None,
) -> Callable[[dict], torch.Tensor]:
    """Return a ``sampler_post_cfg_function`` that anchors video colour drift."""
    log = logger or _LOG
    warned: set[str] = set()
    preserved_cache: dict[tuple[tuple[int, ...], ...], torch.Tensor] = {}

    def _warn_once(key: str, msg: str) -> None:
        if key not in warned:
            warned.add(key)
            log.warning(msg)

    def hook(args: dict) -> torch.Tensor:
        denoised = args["denoised"]
        try:
            model = args["model"]
            if not hasattr(model, "model_sampling"):
                _warn_once("no_sampling", "LatentColourAnchor: model has no model_sampling; skipping.")
                return denoised

            sigma = args["sigma"]
            if not _sigma_active(sigma, model, start_percent, end_percent):
                return denoised

            shapes = getattr(model, "latent_shapes", None)
            if shapes is None:
                if shapes_hint:
                    shapes = list(shapes_hint)
                else:
                    shapes = [tuple(denoised.shape)]

            shapes_key = tuple(tuple(s) for s in shapes)
            device = denoised.device

            if shapes_key not in preserved_cache:
                preserved_cache[shapes_key] = build_preserved_packed(
                    noise_mask, shapes, prepare_mask_fn, pack_fn, device,
                )

            preserved_packed = preserved_cache[shapes_key]
            if preserved_packed.device != device:
                preserved_packed = preserved_packed.to(device)

            src_proc = model.process_latent_in(source_packed_or_nested)
            src_packed = _pack_source(src_proc, pack_fn)

            den_batch = denoised.shape[0]
            src_batch = src_packed.shape[0]
            if src_batch != den_batch:
                if src_batch == 1:
                    src_packed = src_packed.expand(den_batch, *src_packed.shape[1:])
                    if preserved_packed.shape[0] == 1:
                        preserved_packed = preserved_packed.expand(
                            den_batch, *preserved_packed.shape[1:],
                        )
                else:
                    _warn_once(
                        "batch_mismatch",
                        f"LatentColourAnchor: batch mismatch (denoised={den_batch}, "
                        f"source={src_batch}); skipping colour anchor.",
                    )
                    return denoised

            den_video, write_den = split_video(denoised, shapes, unpack_fn, pack_fn)
            src_video, _ = split_video(src_packed, shapes, unpack_fn, pack_fn)
            pres_video, _ = split_video(preserved_packed, shapes, unpack_fn, pack_fn)

            corrected, _stats = anchor_video(
                den_video,
                src_video,
                pres_video,
                strength,
                mode,
                per_frame,
                min_preserved,
            )
            return write_den(corrected)
        except Exception as exc:
            _warn_once("hook_error", f"LatentColourAnchor: {exc} — returning denoised unchanged.")
            return denoised

    return hook
