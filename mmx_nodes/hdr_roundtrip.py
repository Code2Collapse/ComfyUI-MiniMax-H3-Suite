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

from mmx_utils.device import output_device, run_with_cpu_fallback
from mmx_utils.hdr_roundtrip import assert_hdr_not_clamped, tag_out_for_direction, transform_hdr_batch
from mmx_utils.ocio_config import all_views, check_view, config_source_note, list_colorspaces, list_displays

try:
    import comfy.model_management as mm
except Exception:  # not just ImportError: the comfy_kitchen skew raises AttributeError
    mm = None


CODE_OFF = "off (OCIO view)"
# the log codes an H3 HDR LoRA can generate in (docs/research/l9178_minimax_hdr_rnd.md section 4); curves with no
# footroom (ACEScc, DaVinci Intermediate) are left out: their output could never be checked for fake HDR
CODE_CHOICES = ["ACEScct", "ARRI LogC3", "ARRI LogC4", "Sony S-Log3", "Panasonic V-Log", "RED Log3G10", "Cineon"]
CODE_TIP = ("HDR for H3 (experimental): to_view encodes ACEScg scene-linear into this log code (what an H3 HDR LoRA "
            "sees and generates; values the VAE cannot hold - below 0 / above 1 - are counted in the report); "
            "to_scene decodes generated log frames back to ACEScg scene-linear and checks them for fake HDR "
            "(pixels below the curve's black code = negative light = an SDR picture pushed through a log curve). "
            "off: the OCIO view round trip as before. Save Video (C2C) writes the result as 16-bit LogC3 / EXR.")


def _code_space(image: torch.Tensor, direction: str, colorspace: str, code: str) -> tuple[torch.Tensor, str, str]:
    """(image, tag_out, report) for the log code-space mode. Works in ACEScg only - no silent gamut guess."""
    import numpy as np

    from mmx_utils import hdr_log as H
    if colorspace != "ACEScg":
        raise ValueError(f"H3 HDR Roundtrip: code_space works in ACEScg scene-linear; colorspace is {colorspace!r}. "
                         "Set colorspace to ACEScg (convert the plate to ACEScg first if it is in another space).")
    x = image[..., :3].detach().float().cpu().numpy()
    if (direction or "to_view").lower() == "to_view":
        codes = H.encode(x, code)
        hi, lo = float((codes > 1.0).mean()), float((codes < 0.0).mean())
        report = (f"HDR code space: ACEScg scene-linear -> {code} (gamut {H.code_gamut(code)}).\n"
                  f"input: {H.stops_above_white(x):+.2f} stops above white (99.9th pct); "
                  f"{H.ceiling(code):.1f}x white is the brightest value a [0, 1] VAE input carries.\n"
                  f"outside [0, 1] (the VAE clips these): above {hi:.4%}, below {lo:.4%}.")
        out, tag_out = codes, f"{code} code"
    else:
        lin = H.decode(x, code)
        foot = H.footroom_fraction(x, code)
        verdict = ("FAKE HDR: these frames are display-referred pixels pushed through a log curve - the highlights "
                   "were clipped before decoding" if foot is not None and foot > 0.01 else "footroom check passed")
        report = (f"HDR code space: {code} -> ACEScg scene-linear.\n"
                  f"output: {H.stops_above_white(lin):+.2f} stops above white (99.9th pct); peak {float(lin.max()):.3g}.\n"
                  f"footroom: {'n/a' if foot is None else f'{foot:.3%}'} of values below the black code "
                  f"{H.code_of_black(code):.4f} -> {verdict}.")
        out, tag_out = lin, "ACEScg"
    if image.shape[-1] > 3:                                        # alpha rides along untouched
        out = np.concatenate([out, image[..., 3:].detach().float().cpu().numpy()], axis=-1)
    return torch.from_numpy(np.ascontiguousarray(out, dtype=np.float32)), tag_out, report


def _schema_lists() -> tuple[list[str], list[str], list[str]]:
    colorspaces = list_colorspaces()
    displays = list_displays()
    views = all_views(displays)
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
                # L9.181 (owner 2026-11-10, HDR in MiniMax): appended last, default off = the node as it always was
                io.Combo.Input("code_space", options=[CODE_OFF, *CODE_CHOICES], default=CODE_OFF,
                               optional=True, tooltip=CODE_TIP),
            ],
            outputs=[
                io.Image.Output("image"),
                io.String.Output("tag_out"),
                io.String.Output("report"),
            ],
        )

    @classmethod
    def execute(
        cls,
        image,
        direction="to_view",
        colorspace="ACEScg",
        display="sRGB - Display",
        view="ACES 2.0 - SDR 100 nits (Rec.709)",
        tag="HDR",
        code_space=CODE_OFF,
    ) -> io.NodeOutput:
        if code_space and code_space != CODE_OFF:
            if image.ndim == 3:
                image = image.unsqueeze(0)
            out, tag_out, report = _code_space(image, direction, colorspace, code_space)
            return io.NodeOutput(out.to(device=output_device(), dtype=image.dtype), tag_out, report)
        try:
            check_view(display, view)
        except ValueError as exc:
            raise ValueError(f"{cls.__name__.replace('MiniMaxH3_', 'H3 ')}: {exc}") from exc
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
        return io.NodeOutput(out.to(device=output_device(), dtype=image.dtype), tag_out, report)
