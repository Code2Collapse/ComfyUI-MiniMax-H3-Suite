"""P6 — Per-frame denoise strength from face size (MIT port)."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

from comfy_api.latest import io

_PKG = Path(__file__).resolve().parents[1]
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from mmx_utils.av_latent import assert_nested_samples
from mmx_utils.per_frame_denoise import (
    face_heights_from_transform,
    merge_per_frame_video_mask,
    scale_video_noise_mask,
    strength_curve_from_faces,
)
from mmx_utils.transform_types import H3Transform, H3TransformType

try:
    import comfy.nested_tensor as nested_tensor
except Exception:  # not just ImportError: the comfy_kitchen skew raises AttributeError
    nested_tensor = None


class MiniMaxH3_PerFrameDenoise(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MiniMaxH3_PerFrameDenoise",
            display_name="H3 Per-Frame Denoise",
            category="MiniMax H3/Sampling",
            description=(
                "Scale denoise per latent frame inversely to face size (small face → stronger). "
                "Uses noise_mask on the video stream; preserves audio-side zeros from N4 lipsync lock. "
                "Place after N4 Masked Replace. One sigma schedule, per-frame strength via noise_mask."
            ),
            inputs=[
                io.Latent.Input("av_latent"),
                H3TransformType.Input("transform"),
                io.Float.Input("strength_small_face", default=1.0, min=0.0, max=1.0, step=0.05),
                io.Float.Input("strength_large_face", default=0.35, min=0.0, max=1.0, step=0.05),
                io.Combo.Input(
                    "scale_mode",
                    options=["absolute_px", "relative_to_clip"],
                    default="absolute_px",
                ),
                io.Float.Input("face_px_small", default=30.0, min=4.0, max=400.0, step=1.0),
                io.Float.Input("face_px_large", default=120.0, min=8.0, max=800.0, step=1.0),
                io.Float.Input("gamma", default=1.0, min=0.2, max=4.0, step=0.1),
                io.Int.Input("smooth_frames", default=9, min=1, max=61, step=2),
            ],
            outputs=[
                io.Latent.Output("av_latent"),
                io.String.Output("report"),
            ],
        )

    @classmethod
    def fingerprint_inputs(
        cls,
        av_latent,
        transform,
        strength_small_face=1.0,
        strength_large_face=0.35,
        scale_mode="absolute_px",
        face_px_small=30.0,
        face_px_large=120.0,
        gamma=1.0,
        smooth_frames=9,
    ):
        return (
            transform.fingerprint()
            + f"|{strength_small_face}|{strength_large_face}|{scale_mode}|"
            f"{face_px_small}|{face_px_large}|{gamma}|{smooth_frames}"
        )

    @classmethod
    def execute(
        cls,
        av_latent,
        transform: H3Transform,
        strength_small_face,
        strength_large_face,
        scale_mode="absolute_px",
        face_px_small=30.0,
        face_px_large=120.0,
        gamma=1.0,
        smooth_frames=9,
    ) -> io.NodeOutput:
        members = assert_nested_samples(av_latent.get("samples"))
        video, audio = members[0], members[1]
        latent_t = int(video.shape[-3])

        face = face_heights_from_transform(transform)
        strength = strength_curve_from_faces(
            face,
            strength_small_face=float(strength_small_face),
            strength_large_face=float(strength_large_face),
            face_px_small=float(face_px_small),
            face_px_large=float(face_px_large),
            gamma=float(gamma),
            smooth_frames=int(smooth_frames),
            scale_mode=scale_mode,
        )

        prev_vmask = None
        prev = av_latent.get("noise_mask")
        if prev is not None and hasattr(prev, "unbind"):
            prev_vmask = list(prev.unbind())[0]

        vmask, _ = scale_video_noise_mask(video, prev_vmask, strength)
        out, _ = merge_per_frame_video_mask(av_latent, vmask, video, audio)

        lo = float(face_px_small) if scale_mode == "absolute_px" else float(face.min())
        hi = float(face_px_large) if scale_mode == "absolute_px" else float(face.max())
        report = (
            f"per-frame denoise: face {face.min():.0f}-{face.max():.0f}px, ramp "
            f"{lo:.0f}-{hi:.0f}px ({scale_mode}) -> strength "
            f"{strength.max():.2f} (smallest) .. {strength.min():.2f} (largest)\n"
            f"mean {strength.mean():.2f} over {len(strength)} frames, "
            f"{latent_t} latent steps, gamma={gamma}"
        )
        return io.NodeOutput(out, report)
