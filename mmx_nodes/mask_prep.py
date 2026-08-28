"""N2 — H3 Mask Prep (video-side latent mask only)."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

from comfy_api.latest import io, ui

_PKG = Path(__file__).resolve().parents[1]
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from mmx_utils.device import run_with_cpu_fallback
from mmx_utils.h3_constants import video_latent_t
from mmx_utils.h3_grid import snap_frame_count
from mmx_utils.mask_to_token import pixel_mask_to_h3_latent, token_preview_upsample

try:
    import comfy.model_management as mm
except Exception:  # not just ImportError: the comfy_kitchen skew raises AttributeError
    mm = None


class MiniMaxH3_MaskPrep(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MiniMaxH3_MaskPrep",
            display_name="H3 Mask Prep",
            category="MiniMax H3/Spine",
            description=(
                "Clean a pixel mask and reduce it to H3 video-latent resolution with "
                "max-reduction on the 16x VAE grid, 2x2 DiT token patches, and the "
                "(1,4,4,4,4) temporal cycle. Output is the VIDEO side only at latent "
                "resolution — coarse regional denoise (32 source px per token). "
                "Fine edge feathering belongs in H3 Stitch Back (pixel space).\n\n"
                "Downstream N4 assembles latent['noise_mask'] as "
                "comfy.nested_tensor.NestedTensor((video_mask, audio_mask)). "
                "N2 does not build that NestedTensor.\n\n"
                "The sampler re-pools via model_base._pool_masks_to_token_grid and "
                "_token_grid_masks (ceil(m*256)/256). Pre-applying those steps here "
                "is idempotent — no double-shrink.\n\n"
                "P5 pattern (no separate node): N2 mask → N4 sets latent['noise_mask'] "
                "NestedTensor((video_mask, audio_mask)). The DiT applies per-token "
                "noise_mask during sampling — regional strength without SplitSigmas."
            ),
            inputs=[
                io.Mask.Input("mask"),
                io.Int.Input("width", default=768, min=32, max=4096, step=32),
                io.Int.Input("height", default=768, min=32, max=4096, step=32),
                io.Int.Input("frame_count", default=22, min=5, max=400),
                io.Float.Input("denoise_strength", default=1.0, min=0.0, max=1.0, step=0.01),
                io.Int.Input("temporal_clean", default=2, min=0, max=32),
                io.Int.Input("spatial_dilate", default=0, min=0, max=64),
                io.Boolean.Input("quantize_soft", default=True),
            ],
            outputs=[
                io.Mask.Output("latent_mask"),
                io.Mask.Output("token_preview"),
                io.String.Output("report"),
            ],
        )

    @classmethod
    def fingerprint_inputs(
        cls,
        mask,
        width=768,
        height=768,
        frame_count=22,
        denoise_strength=1.0,
        temporal_clean=2,
        spatial_dilate=0,
        quantize_soft=True,
    ):
        return hashlib.md5(
            (
                hashlib.md5(mask.cpu().numpy().tobytes()).hexdigest()
                + f"|{width}|{height}|{frame_count}|{denoise_strength}|"
                f"{temporal_clean}|{spatial_dilate}|{quantize_soft}"
            ).encode()
        ).hexdigest()

    @classmethod
    def execute(
        cls,
        mask,
        width=768,
        height=768,
        frame_count=22,
        denoise_strength=1.0,
        temporal_clean=2,
        spatial_dilate=0,
        quantize_soft=True,
    ):
        fc = snap_frame_count(frame_count)
        dev = mask.device
        if mm is not None:
            try:
                dev = mm.get_torch_device()
            except Exception:
                dev = mask.device

        def _work(dev):
            m = mask.to(dev)
            lat = pixel_mask_to_h3_latent(
                m,
                width=int(width),
                height=int(height),
                frame_count=fc,
                spatial_dilate=int(spatial_dilate),
                temporal_clean=int(temporal_clean),
                denoise_strength=float(denoise_strength),
                quantize=bool(quantize_soft),
            )
            prev = token_preview_upsample(lat, int(height), int(width))
            return lat.to(mask.device), prev.to(mask.device)

        latent, preview = run_with_cpu_fallback(
            _work, device=dev, label="MiniMaxH3_MaskPrep"
        )
        t_lat = video_latent_t(fc)
        report = (
            f"video-side latent mask only; NestedTensor assembly is N4's job.\n"
            f"frame_count={fc} (snapped) latent_t={t_lat} shape={tuple(latent.shape)} "
            f"spatial={width//16}x{height//16}\n"
            f"quantize_soft={quantize_soft} denoise_strength={denoise_strength}\n"
            f"Sampler re-pool is idempotent with this output."
        )
        # First frame only — PreviewMask writes one PNG per batch frame and the widget
        # shows a single side-by-side pair. The MASK sockets keep the full batch.
        pixel_ui = ui.PreviewMask(mask.float()[:1], cls=cls).as_dict()
        token_ui = ui.PreviewMask(preview.float()[:1], cls=cls).as_dict()
        return io.NodeOutput(
            latent,
            preview,
            report,
            ui={
                "mmx_pixel_mask": pixel_ui["images"],
                "mmx_token_preview": token_ui["images"],
                "animated": pixel_ui.get("animated", (False,)),
            },
        )
