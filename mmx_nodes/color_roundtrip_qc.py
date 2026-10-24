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
from mmx_utils.ocio_config import all_views, check_view, config_source_note, list_colorspaces, list_displays

try:
    import comfy.model_management as mm
except Exception:  # not just ImportError: the comfy_kitchen skew raises AttributeError
    mm = None


def _schema_lists() -> tuple[list[str], list[str], list[str]]:
    colorspaces = list_colorspaces()
    displays = list_displays()
    views = all_views(displays)
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
                # C2C L9.39: appended. A gate whose bool feeds nothing never ran (not an output node) and could
                # not stop anything; it is an output node now, and this decides what a failure does.
                io.Combo.Input("on_fail", options=["report only", "stop the run"], default="report only",
                               optional=True,
                               tooltip="report only: the verdict shows on the node and in the outputs. stop the "
                                       "run: a failure stops the queue - put this gate BEFORE your Save (wire the "
                                       "passthrough output) and nothing failing is saved."),
            ],
            outputs=[
                io.Boolean.Output("passed"),
                io.Float.Output("max_error"),
                io.String.Output("report"),
                io.Image.Output("error_map"),
                io.Image.Output("image_out", tooltip="The checked image, unchanged: wire it on so a failed gate "
                                                      "(on_fail = stop the run) stops before what comes next."),
            ],
            is_output_node=True,
        )

    @classmethod
    def execute(
        cls,
        image,
        scene_colorspace="ACEScg",
        display="sRGB - Display",
        view="ACES 2.0 - SDR 100 nits (Rec.709)",
        tolerance=1e-4,
        on_fail="report only",
    ) -> io.NodeOutput:
        try:
            check_view(display, view)
        except ValueError as exc:
            raise ValueError(f"{cls.__name__.replace('MiniMaxH3_', 'H3 ')}: {exc}") from exc
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
            if on_fail == "stop the run":
                raise ValueError(f"H3 Color Round-Trip QC stopped the run: {report}")

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
            image,
            ui={"text": [report]},
        )
