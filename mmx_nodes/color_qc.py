"""H3 Color QC — objective colour/HDR/texture/flicker metrics for Hybrid HDR A/B."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
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
                "texture energy, temporal flicker, and optional Oklab distance to a reference. Also per frame: "
                "colourfulness trend, frame-to-frame Oklab drift (worst frame named), chroma and luma histograms, "
                "a banding heuristic and deltaE_OK against the reference - all in metrics_json for A/B tables."
            ),
            inputs=[
                io.Image.Input("images", tooltip="Decoded clip as display sRGB frames (VAE Decode output); every "
                                                 "frame is measured, sRGB-decoded to linear for the luma / DR / "
                                                 "headroom metrics."),
                io.Image.Input(
                    "reference",
                    optional=True,
                    tooltip="Optional reference frame for Oklab palette distance.",
                ),
            ],
            outputs=[
                io.String.Output("report", tooltip="Text summary of the clip metrics (colourfulness, luma DR, "
                                                   "clipped fraction, headroom, texture, flicker, Oklab distance)."),
                io.Image.Output("preview", tooltip="One 256x48 IMAGE strip: grey bar = luma p1..p99, red bar = "
                                                   "clipped fraction, green bar = colourfulness (/109), thumbnail "
                                                   "of the first frame at the right."),
                # appended last (L8.06): saved links to report / preview keep their slots
                io.String.Output("metrics_json", tooltip="Every number above plus per-frame lists, as JSON."),
            ],
        )

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
            return report, preview.to(images.dtype), json.dumps(asdict(metrics))

        report, preview, metrics_json = run_with_cpu_fallback(_work, device=dev, label="MiniMaxH3_ColorQC")
        return io.NodeOutput(report, preview, metrics_json)
