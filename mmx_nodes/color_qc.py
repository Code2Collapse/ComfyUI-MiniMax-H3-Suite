"""H3 Color QC — objective colour/HDR/texture/flicker metrics for Hybrid HDR A/B."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import torch

from comfy_api.latest import io

_PKG = Path(__file__).resolve().parents[1]
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from mmx_utils.color_qc import build_preview_strip, measure_color_qc
from mmx_utils.device import run_with_cpu_fallback

try:
    import comfy.model_management as mm
except Exception:  # noqa: BLE001
    mm = None


class MiniMaxH3_ColorQC(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MiniMaxH3_ColorQC",
            display_name="H3 Color QC",
            category="MiniMax H3/QC",
            description=(
                "Score a decoded clip for Hybrid HDR A/B: Hasler-Süsstrunk colourfulness "
                "(display sRGB 0-255), linear luma DR, clipped highlight fraction, headroom, "
                "texture energy, temporal flicker, and optional Oklab distance to a reference."
            ),
            inputs=[
                io.Image.Input("images"),
                io.Image.Input(
                    "reference",
                    optional=True,
                    tooltip="Optional reference frame for Oklab palette distance.",
                ),
            ],
            outputs=[
                io.String.Output("report"),
                io.Image.Output("preview"),
            ],
        )

    @classmethod
    def fingerprint_inputs(cls, images, reference=None):
        h = hashlib.md5(images.cpu().numpy().tobytes()).hexdigest()
        if reference is not None:
            h += hashlib.md5(reference.cpu().numpy().tobytes()).hexdigest()
        return h

    @classmethod
    def execute(cls, images, reference=None) -> io.NodeOutput:
        dev = images.device
        if mm is not None:
            try:
                dev = mm.get_torch_device()
            except Exception:
                dev = images.device

        def _work(device):
            imgs = images.to(device)
            ref = reference.to(device) if reference is not None else None
            metrics, report = measure_color_qc(imgs, ref)
            preview = build_preview_strip(imgs, metrics)
            return report, preview.to(images.dtype)

        report, preview = run_with_cpu_fallback(_work, device=dev, label="MiniMaxH3_ColorQC")
        return io.NodeOutput(report, preview)
