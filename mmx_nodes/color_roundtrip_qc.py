"""W4 — Colour round-trip QC (scene-linear ↔ display ↔ scene-linear)."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import torch

from comfy_api.latest import io

_PKG = Path(__file__).resolve().parents[1]
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from mmx_utils.color_roundtrip_qc import run_display_roundtrip_qc
from mmx_utils.ocio_config import config_source_note, list_colorspaces, list_displays, list_views

try:
    import comfy.model_management as mm
except ImportError:
    mm = None


def _schema_lists() -> tuple[list[str], list[str], list[str]]:
    colorspaces = list_colorspaces()
    displays = list_displays()
    views = list_views(displays[0] if displays else "sRGB - Display")
    return colorspaces, displays, views


class MiniMaxH3_ColorRoundTripQC(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        colorspaces, displays, views = _schema_lists()
        default_scene = "ACEScg" if "ACEScg" in colorspaces else colorspaces[0]
        default_display = displays[0] if displays else "sRGB - Display"
        default_view = views[0] if views else "ACES 2.0 - SDR 100 nits (Rec.709)"
        return io.Schema(
            node_id="MiniMaxH3_ColorRoundTripQC",
            display_name="H3 Color Round-Trip QC",
            category="MiniMax H3/Color",
            description=(
                "Encode a probe through scene-linear → display → scene-linear and report error. "
                "Fails loudly in report when tolerance is exceeded or HDR highlights are lost. "
                "Use the HDR probe [0, 0.5, 1, 4, 16] in tests — values above 1.0 must survive."
            ),
            inputs=[
                io.Image.Input("image"),
                io.Combo.Input(
                    "scene_colorspace",
                    options=colorspaces,
                    default=default_scene,
                ),
                io.Combo.Input("display", options=displays, default=default_display),
                io.Combo.Input("view", options=views, default=default_view),
                io.Float.Input("tolerance", default=1e-4, min=0.0, max=1.0, step=1e-6),
            ],
            outputs=[
                io.Boolean.Output("passed"),
                io.Float.Output("max_error"),
                io.String.Output("report"),
                io.Image.Output("error_map"),
            ],
        )

    @classmethod
    def fingerprint_inputs(
        cls,
        image,
        scene_colorspace="ACEScg",
        display="sRGB - Display",
        view="ACES 2.0 - SDR 100 nits (Rec.709)",
        tolerance=1e-4,
    ):
        raw = image.detach().cpu().numpy().tobytes()
        return (
            hashlib.md5(raw).hexdigest()
            + f"|{scene_colorspace}|{display}|{view}|{tolerance}"
        )

    @classmethod
    def execute(
        cls,
        image,
        scene_colorspace="ACEScg",
        display="sRGB - Display",
        view="ACES 2.0 - SDR 100 nits (Rec.709)",
        tolerance=1e-4,
    ) -> io.NodeOutput:
        if image.ndim == 3:
            image = image.unsqueeze(0)

        result = run_display_roundtrip_qc(
            image,
            scene_colorspace=scene_colorspace,
            display=display,
            view=view,
            tolerance=float(tolerance),
        )
        header = config_source_note()
        report = f"{header}\n{result.report}"
        if not result.passed:
            report = "FAILED — " + report

        h, w = int(image.shape[1]), int(image.shape[2])
        err = torch.from_numpy(result.error_map).to(image.device, image.dtype)
        if err.ndim == 3:
            err = err.unsqueeze(0)
        if err.shape[1:3] != (h, w):
            err = torch.nn.functional.interpolate(
                err.permute(0, 3, 1, 2),
                size=(h, w),
                mode="nearest",
            ).permute(0, 2, 3, 1)

        return io.NodeOutput(
            bool(result.passed),
            float(result.max_error),
            report,
            err,
        )
