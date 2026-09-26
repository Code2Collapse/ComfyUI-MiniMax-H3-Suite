"""N1 — H3 Latent Colour Anchor (stop red drift during sampling)."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import torch

from comfy_api.latest import io

_PKG = Path(__file__).resolve().parents[1]
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from mmx_utils.latent_color_anchor import make_hook  # noqa: E402

_LOG = logging.getLogger(__name__)

# H3 manual compression — same defaults as Mask To Latent Space for ref2va.
_H3_MASK_COMPRESSION = {
    "compression": "manual",
    "spatial": 16,
    "token_spatial": 2,
    "frames_per_latent": "1,4,4,4,4",
}


def _samples_shapes(samples: Any) -> list[tuple[int, ...]]:
    if getattr(samples, "is_nested", False):
        return [tuple(t.shape) for t in samples.unbind()]
    if isinstance(samples, torch.Tensor):
        return [tuple(samples.shape)]
    raise ValueError("latent samples must be a tensor or NestedTensor")


def _video_tensor(samples: Any) -> torch.Tensor:
    if getattr(samples, "is_nested", False):
        return samples.unbind()[0]
    return samples


def _pixel_mask_to_denoise_mask(pixel_mask: torch.Tensor, video_latent: torch.Tensor) -> torch.Tensor:
    """Reduce a pixel-space MASK to the video latent grid (Mask To Latent Space path)."""
    from mmx_nodes.mask_to_latent import MiniMaxH3_MaskToLatentSpace, _resolve_geometry

    spatial, groups, _, token = _resolve_geometry(_H3_MASK_COMPRESSION, None)
    reduced = MiniMaxH3_MaskToLatentSpace._reduce(
        pixel_mask,
        spatial,
        groups,
        token,
        "max",
        "max",
        0,
        0,
    )
    target_t, target_h, target_w = int(video_latent.shape[2]), int(video_latent.shape[3]), int(video_latent.shape[4])
    got_t, got_h, got_w = int(reduced.shape[0]), int(reduced.shape[1]), int(reduced.shape[2])
    if (got_t, got_h, got_w) != (target_t, target_h, target_w):
        raise ValueError(
            f"pixel mask reduced to {got_t}x{got_h}x{got_w} but video latent is "
            f"{target_t}x{target_h}x{target_w} — check frame count and resolution"
        )
    mask = reduced.to(device=video_latent.device, dtype=video_latent.dtype)
    return mask.unsqueeze(0).unsqueeze(0).expand_as(video_latent)


def _resolve_noise_mask(latent: dict, pixel_mask: torch.Tensor | None) -> Any | None:
    if pixel_mask is not None:
        video = _video_tensor(latent["samples"])
        denoise = _pixel_mask_to_denoise_mask(pixel_mask, video)
        if getattr(latent.get("samples"), "is_nested", False):
            import comfy.nested_tensor

            audio_shape = latent["samples"].unbind()[1].shape
            audio_mask = torch.zeros(audio_shape, device=denoise.device, dtype=denoise.dtype)
            return comfy.nested_tensor.NestedTensor((denoise, audio_mask))
        return denoise

    noise_mask = latent.get("noise_mask")
    if noise_mask is None:
        return None
    return noise_mask


class MiniMaxH3_LatentColorAnchor(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MiniMaxH3_LatentColorAnchor",
            display_name="H3 Latent Colour Anchor (stop red drift)",
            category="MiniMax H3/Sampling",
            description=(
                "Patches the sampler to remove temporal colour drift every step, before "
                "it compounds into the red/blush segments seen on ref2va (Comfy-Org #55).\n\n"
                "The model is told which latent cells to KEEP via the noise mask; this node "
                "measures how far the denoised prediction has drifted from the source latent "
                "on those preserved cells, then removes that offset from the whole frame. "
                "Late steps are left alone — they are texture, not colour."
            ),
            inputs=[
                io.Model.Input("model"),
                io.Latent.Input(
                    "latent",
                    tooltip=(
                        "The LATENT wired into the sampler (with Set Latent Noise Mask). "
                        "Its noise_mask defines preserved vs regenerated regions."
                    ),
                ),
                io.Mask.Input(
                    "mask",
                    optional=True,
                    tooltip=(
                        "Optional pixel-space mask override (white = regenerate). Converted "
                        "to the video latent grid with the same geometry as H3 Mask To Latent "
                        "Space. When connected, replaces the latent noise_mask."
                    ),
                ),
                io.Float.Input(
                    "strength",
                    default=0.8,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip="How much of the measured drift to remove each active step. 1 = full correction.",
                ),
                io.Combo.Input(
                    "mode",
                    options=["mean", "mean_std"],
                    default="mean",
                    tooltip=(
                        "mean — subtract the per-channel offset measured on preserved cells.\n"
                        "mean_std — also match spread (contrast) on preserved cells, clamped."
                    ),
                ),
                io.Boolean.Input(
                    "per_frame",
                    default=True,
                    tooltip="Correct each latent frame independently. Off = one correction vector for all T.",
                ),
                io.Float.Input(
                    "start_percent",
                    default=0.0,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip="Earliest sampling step (high sigma) where anchoring is active.",
                ),
                io.Float.Input(
                    "end_percent",
                    default=0.85,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip=(
                        "Last active step (low sigma). Steps below this are untouched so "
                        "fine detail and grain are not smeared."
                    ),
                ),
                io.Float.Input(
                    "min_preserved",
                    default=0.05,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip="Skip frames where fewer than this fraction of cells are preserved.",
                ),
            ],
            outputs=[
                io.Model.Output(
                    tooltip="MODEL with post-CFG colour anchor installed. Input model is never mutated.",
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        model,
        latent,
        strength=0.8,
        mode="mean",
        per_frame=True,
        start_percent=0.0,
        end_percent=0.85,
        min_preserved=0.05,
        mask=None,
    ) -> io.NodeOutput:
        noise_mask = _resolve_noise_mask(latent, mask)
        if noise_mask is None:
            msg = (
                "Latent Colour Anchor: no noise_mask on the latent and no mask input — "
                "nothing defines a preserved region to anchor on. Model returned unchanged."
            )
            _LOG.info(msg)
            return io.NodeOutput(model)

        import comfy.sampler_helpers
        import comfy.utils

        shapes_hint = _samples_shapes(latent["samples"])
        hook = make_hook(
            latent["samples"],
            noise_mask,
            float(strength),
            mode,
            bool(per_frame),
            float(min_preserved),
            float(start_percent),
            float(end_percent),
            comfy.sampler_helpers.prepare_mask,
            comfy.utils.pack_latents,
            comfy.utils.unpack_latents,
            shapes_hint=shapes_hint,
            logger=_LOG,
        )

        patched = model.clone()
        patched.set_model_sampler_post_cfg_function(hook)
        return io.NodeOutput(patched)
