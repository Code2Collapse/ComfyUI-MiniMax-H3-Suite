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
                "divide: lighting-invariant (plate/(blur+eps) * generated).\n"
                "subtract: plate - blur added to generated.\n\n"
                "Mask-limited with strength 0–1. Exterior pixels are the generated image "
                "(mask=0 outside region)."
            ),
            inputs=[
                io.Image.Input("plate"),
                io.Image.Input("generated"),
                io.Mask.Input("mask", optional=True),
                io.Combo.Input("mode", options=["divide", "subtract"], default="divide"),
                io.Combo.Input("blur_method", options=["Gaussian", "Box"], default="Gaussian"),
                io.Int.Input("blur_radius", default=8, min=1, max=64),
                io.Float.Input("strength", default=1.0, min=0.0, max=1.0, step=0.01),
                io.Float.Input("eps", default=1e-4, min=1e-6, max=1e-2, step=1e-5),
            ],
            outputs=[
                io.Image.Output("images"),
                io.String.Output("report"),
            ],
        )

    @classmethod
    def fingerprint_inputs(
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
        base = hashlib.md5(plate.cpu().numpy().tobytes()).hexdigest()
        base += ":" + hashlib.md5(generated.cpu().numpy().tobytes()).hexdigest()
        if mask is not None:
            base += ":" + hashlib.md5(mask.cpu().numpy().tobytes()).hexdigest()
        return f"{base}|{mode}|{blur_method}|{blur_radius}|{strength}|{eps}"

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
