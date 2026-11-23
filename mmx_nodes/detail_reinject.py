"""N6 — H3 Detail Reinject (texture transfer inside mask)."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import torch

from comfy_api.latest import io

_PKG = Path(__file__).resolve().parents[1]
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from mmx_utils.device import run_with_cpu_fallback
from mmx_utils.frequency import detail_reinject_frame

try:
    import comfy.model_management as mm
except Exception:  # not just ImportError: the comfy_kitchen skew raises AttributeError
    mm = None


class MiniMaxH3_DetailReinject(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MiniMaxH3_DetailReinject",
            display_name="H3 Detail Reinject",
            category="MiniMax H3/Spine",
            description=(
                "Transfer plate micro-texture onto the regenerated region so the texture seam "
                "dies even when geometry is perfect.\n\n"
                "Frequency separation: the region keeps the generated image's own low band (lighting, colour) "
                "and takes the plate's fine detail.\n"
                "divide: blur(generated) * plate/(blur(plate)+eps) (lighting-invariant).\n"
                "subtract: blur(generated) + (plate - blur(plate)).\n"
                "An unchanged region comes back unchanged.\n\n"
                "Mask-limited with strength 0–1. Exterior pixels are the generated image "
                "(mask=0 outside region)."
            ),
            inputs=[
                io.Image.Input("plate", tooltip="Original frames whose fine detail (high band: image minus its "
                                                "blur) is transferred. Same size and frame order as generated."),
                io.Image.Input("generated", tooltip="Regenerated frames (e.g. H3 Stitch Back output). Their low "
                                                    "band (blurred lighting/colour) is kept; outside the mask they "
                                                    "pass through. Frames = min(plate, generated)."),
                io.Mask.Input("mask", optional=True,
                              tooltip="Where detail is transferred: 1 = full transfer, 0 = generated unchanged; "
                                      "resized bilinearly to the frame, last frame reused if short. Not wired = "
                                      "the whole frame."),
                io.Combo.Input("mode", options=["divide", "subtract"], default="divide",
                               tooltip="divide: blur(generated) * plate / (blur(plate) + eps), detail as a ratio. "
                                       "subtract: blur(generated) + (plate - blur(plate)), detail as an offset. "
                                       "Identical inputs return unchanged in both."),
                io.Combo.Input("blur_method", options=["Gaussian", "Box"], default="Gaussian",
                               tooltip="Low-pass filter that splits low and high bands: Gaussian or Box (flat "
                                       "average), both with blur_radius.",
                               extra_dict={"c2c_section": "Filter"}),
                io.Int.Input("blur_radius", default=8, min=1, max=64,
                             tooltip="Blur radius in pixels. Detail finer than this comes from the plate; "
                                     "coarser structure stays from generated. Larger = more plate structure "
                                     "transferred."),
                io.Float.Input("strength", default=1.0, min=0.0, max=1.0, step=0.01,
                               tooltip="0-1 mix from generated (0) to the full reinjected result (1) inside the "
                                       "mask."),
                io.Float.Input("eps", default=1e-4, min=1e-6, max=1e-2, step=1e-5,
                               tooltip="divide mode: added to |blur(plate)| before dividing, to keep near-black "
                                       "areas from blowing up. Larger = weaker detail in very dark regions.",
                               extra_dict={"c2c_section": "Filter", "c2c_show_if": {"mode": ["divide"]}}),
            ],
            outputs=[
                io.Image.Output("images", tooltip="IMAGE batch (RGB, not clamped), one frame per input pair: "
                                                  "generated with the plate's fine detail inside the mask."),
                io.String.Output("report", tooltip="Plain-text summary: mode, strength, blur radius, frame count."),
            ],
        )

    @classmethod
    def execute(
        cls,
        plate,
        generated,
        mask=None,
        mode="divide",
        blur_method="Gaussian",
        blur_radius=8,
        strength=1.0,
        eps=1e-4,
    ):
        if plate.ndim == 3:
            plate = plate.unsqueeze(0)
        if generated.ndim == 3:
            generated = generated.unsqueeze(0)
        b = min(plate.shape[0], generated.shape[0])

        dev = plate.device
        if mm is not None:
            try:
                dev = mm.get_torch_device()
            except Exception:
                dev = plate.device

        chunk = 4

        def _work(dev):
            out_frames = []
            for c0 in range(0, b, chunk):
                c1 = min(c0 + chunk, b)
                for i in range(c0, c1):
                    m = None
                    if mask is not None:
                        m = mask[min(i, mask.shape[0] - 1)].to(dev)
                    frame = detail_reinject_frame(
                        plate[i].to(dev),
                        generated[i].to(dev),
                        m,
                        mode=mode,
                        radius=int(blur_radius),
                        method=blur_method,
                        strength=float(strength),
                        eps=float(eps),
                    )
                    out_frames.append(frame.unsqueeze(0))
            return torch.cat(out_frames, dim=0).to(plate.device, plate.dtype)

        out = run_with_cpu_fallback(_work, device=dev, label="MiniMaxH3_DetailReinject")
        report = (
            f"detail_reinject mode={mode} strength={strength} radius={blur_radius}\n"
            f"frames={b} chunk_size={chunk} (per-frame MB budget)"
        )
        return io.NodeOutput(out, report)
