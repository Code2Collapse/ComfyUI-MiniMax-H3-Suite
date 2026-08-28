"""N16 — HDR round-trip with tag propagation (built on ocio_bridge)."""

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
from mmx_utils.hdr_roundtrip import assert_hdr_not_clamped, tag_out_for_direction, transform_hdr_batch
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


class MiniMaxH3_HDRRoundtrip(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        colorspaces, displays, views = _schema_lists()
        default_scene = "ACEScg" if "ACEScg" in colorspaces else colorspaces[0]
        default_display = displays[0] if displays else "sRGB - Display"
        default_view = views[0] if views else "ACES 2.0 - SDR 100 nits (Rec.709)"
        return io.Schema(
            node_id="MiniMaxH3_HDRRoundtrip",
            display_name="H3 HDR Roundtrip",
            category="MiniMax H3/Color",
            description=(
                "Scene-linear ↔ view via OCIO with HDR tag propagation. "
                "Uses live display/view names (e.g. 'ACES 2.0 - SDR 100 nits (Rec.709)'). "
                "No torch.clamp on HDR paths — values above 1.0 must survive to_scene."
            ),
            inputs=[
                io.Image.Input("image"),
                io.Combo.Input("direction", options=["to_view", "to_scene"], default="to_view"),
                io.Combo.Input("colorspace", options=colorspaces, default=default_scene),
                io.Combo.Input("display", options=displays, default=default_display),
                io.Combo.Input("view", options=views, default=default_view),
                io.Combo.Input("tag", options=["HDR", "Rec709", "ACEScg"], default="HDR"),
            ],
            outputs=[
                io.Image.Output("image"),
                io.String.Output("tag_out"),
                io.String.Output("report"),
            ],
        )

    @classmethod
    def fingerprint_inputs(
        cls,
        image,
        direction="to_view",
        colorspace="ACEScg",
        display="sRGB - Display",
        view="ACES 2.0 - SDR 100 nits (Rec.709)",
        tag="HDR",
    ):
        raw = image.detach().cpu().numpy().tobytes()
        return hashlib.md5(raw).hexdigest() + f"|{direction}|{colorspace}|{display}|{view}|{tag}"

    @classmethod
    def execute(
        cls,
        image,
        direction="to_view",
        colorspace="ACEScg",
        display="sRGB - Display",
        view="ACES 2.0 - SDR 100 nits (Rec.709)",
        tag="HDR",
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
        tag_in = tag or "HDR"

        def _work(device: torch.device) -> torch.Tensor:
            return transform_hdr_batch(
                image.to(device),
                direction=direction,  # type: ignore[arg-type]
                colorspace=colorspace,
                display=display,
                view=view,
                chunk_size=chunk,
            )

        out = run_with_cpu_fallback(_work, device=dev, label="MiniMaxH3_HDRRoundtrip")
        if (direction or "to_view").lower() == "to_scene":
            audit = assert_hdr_not_clamped(image, out, tag=tag_in)
        else:
            audit = (
                f"tag={tag_in}: to_view display transform — HDR peak audit skipped "
                "(view transforms may crush highlights by design)."
            )
        tag_out = tag_out_for_direction(direction, colorspace, tag_in)
        report = (
            f"HDR roundtrip direction={direction} tag_in={tag_in} tag_out={tag_out}\n"
            f"colorspace={colorspace!r} display={display!r} view={view!r}\n"
            f"{src_note}\n{audit}"
        )
        return io.NodeOutput(out.to(image.dtype), tag_out, report)
