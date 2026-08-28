"""I1 — OCIO scene-linear ↔ display bridge (HDR-safe, no clamp)."""

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
from mmx_utils.ocio_bridge import OCIO_FORCES_PER_FRAME_PLANES, transform_image_batch
from mmx_utils.ocio_config import config_source_note, list_colorspaces, list_displays, list_views

try:
    import comfy.model_management as mm
except Exception:  # not just ImportError: the comfy_kitchen skew raises AttributeError
    mm = None


def _schema_lists() -> tuple[list[str], list[str], list[str]]:
    colorspaces = list_colorspaces()
    displays = list_displays()
    views = list_views(displays[0] if displays else "sRGB - Display")
    return colorspaces, displays, views


class MiniMaxH3_OCIOBridge(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        colorspaces, displays, views = _schema_lists()
        default_scene = "ACEScg" if "ACEScg" in colorspaces else colorspaces[0]
        default_display = displays[0] if displays else "sRGB - Display"
        default_view = views[0] if views else "ACES 2.0 - SDR 100 nits (Rec.709)"
        return io.Schema(
            node_id="MiniMaxH3_OCIOBridge",
            display_name="H3 OCIO Bridge",
            category="MiniMax H3/Color",
            description=(
                "Scene-linear ↔ display via OpenColorIO. No torch.clamp on HDR paths — "
                "values above 1.0 survive an invertible round trip. "
                "to_display: source scene → display/view. "
                "to_scene_linear: inverse display/view → target scene. "
                "OCIO applies one CPU plane per frame (known throughput cost)."
            ),
            inputs=[
                io.Image.Input("image"),
                io.Combo.Input(
                    "direction",
                    options=["to_display", "to_scene_linear"],
                    default="to_display",
                ),
                io.Combo.Input(
                    "source_colorspace",
                    options=colorspaces,
                    default=default_scene,
                ),
                io.Combo.Input(
                    "target_colorspace",
                    options=colorspaces,
                    default=default_scene,
                ),
                io.Combo.Input("display", options=displays, default=default_display),
                io.Combo.Input("view", options=views, default=default_view),
            ],
            outputs=[
                io.Image.Output("image"),
                io.String.Output("report"),
            ],
        )

    @classmethod
    def fingerprint_inputs(
        cls,
        image,
        direction="to_display",
        source_colorspace="ACEScg",
        target_colorspace="ACEScg",
        display="sRGB - Display",
        view="ACES 2.0 - SDR 100 nits (Rec.709)",
    ):
        raw = image.detach().cpu().numpy().tobytes()
        return (
            hashlib.md5(raw).hexdigest()
            + f"|{direction}|{source_colorspace}|{target_colorspace}|{display}|{view}"
        )

    @classmethod
    def execute(
        cls,
        image,
        direction="to_display",
        source_colorspace="ACEScg",
        target_colorspace="ACEScg",
        display="sRGB - Display",
        view="ACES 2.0 - SDR 100 nits (Rec.709)",
    ) -> io.NodeOutput:
        if image.ndim == 3:
            image = image.unsqueeze(0)

        dev = image.device
        if mm is not None:
            try:
                dev = mm.get_torch_device()
            except Exception:
                dev = image.device

        chunk = 4
        src_note = config_source_note()

        def _work(device: torch.device) -> torch.Tensor:
            return transform_image_batch(
                image.to(device),
                direction=direction,  # type: ignore[arg-type]
                source_colorspace=source_colorspace,
                target_colorspace=target_colorspace,
                display=display,
                view=view,
                chunk_size=chunk,
            )

        out = run_with_cpu_fallback(_work, device=dev, label="MiniMaxH3_OCIOBridge")
        per_frame = "yes (OCIO PackedImageDesc is 2-D per call)" if OCIO_FORCES_PER_FRAME_PLANES else "no"
        report = (
            f"OCIO bridge direction={direction} "
            f"{source_colorspace!r} -> {target_colorspace!r} "
            f"display={display!r} view={view!r}\n"
            f"{src_note}\n"
            f"frames={int(image.shape[0])} chunk={chunk} per_frame_ocio={per_frame}"
        )
        return io.NodeOutput(out.to(image.dtype), report)
