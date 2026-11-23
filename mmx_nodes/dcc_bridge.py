"""N14 — DCC bridge: ingest external EXR sim frames (R21: no solver)."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import torch

from comfy_api.latest import io

_PKG = Path(__file__).resolve().parents[1]
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from mmx_utils.dcc_bridge import (
    INTERACTION_MODES,
    SIM_TYPES,
    DCCSequenceError,
    build_retention_payload,
    load_exr_sequence,
    retention_json_text,
)
from mmx_utils.device import run_with_cpu_fallback

try:
    import comfy.model_management as mm
except Exception:  # not just ImportError: the comfy_kitchen skew raises AttributeError
    mm = None


class MiniMaxH3_DCCBridge(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MiniMaxH3_DCCBridge",
            display_name="H3 DCC Bridge",
            category="MiniMax H3/Bridge",
            description=(
                "Ingest externally-rendered EXR sim frames via OpenImageIO (subimage 0 only). "
                "Flat RGB or single-channel EXR; >3 channels raises — no silent truncation. "
                "HDR values above 1.0 are preserved (16f/32f). No solver — missing EXR raises."
            ),
            inputs=[
                io.String.Input("exr_sequence", default="", multiline=False,
                                tooltip="Rendered EXR frames to load: a folder (every .exr / .sxr, sorted), a glob "
                                        "such as D:/sim/smoke.*.exr, or one file. Read with OpenImageIO; empty or "
                                        "no match is an error."),
                io.Combo.Input("sim_type", options=list(SIM_TYPES), default="fluid",
                               tooltip="Kind of simulation. Only written as text into retention_json and the report "
                                       "(\"<Video 1>: <type> sim reference\"); it does not change the frames."),
                io.Combo.Input("interaction_mode", options=list(INTERACTION_MODES), default="fully_preserved",
                               tooltip="Only written as text into retention_json (the 'camera: <mode>' line) and the "
                                       "report; it does not change the frames."),
                io.Float.Input("fps", default=24.0, min=1.0, max=120.0, step=0.01,
                               tooltip="Rate of the external sim, only recorded in retention_json ('timing' line and "
                                       "fps field); frames are not retimed."),
            ],
            outputs=[
                io.Image.Output("reference_video", tooltip="The EXR frames as an IMAGE batch, one per file in sorted "
                                                           "order, linear float with values above 1.0 kept "
                                                           "(no clamp); 1-channel EXRs become grey RGB."),
                io.String.Output("retention_json", tooltip="JSON: video_label <Video 1>, sim_type, interaction_mode, "
                                                           "fps and retention_analysis_lines (four text lines: "
                                                           "sim, camera, environment, timing)."),
                io.String.Output("report", tooltip="Text: frames loaded, size, peak value, sim type and mode."),
            ],
        )

    @classmethod
    def fingerprint_inputs(cls, exr_sequence="", sim_type="fluid", interaction_mode="fully_preserved", fps=24.0):
        return hashlib.md5(f"{exr_sequence}|{sim_type}|{interaction_mode}|{fps}".encode()).hexdigest()

    @classmethod
    def execute(
        cls,
        exr_sequence="",
        sim_type="fluid",
        interaction_mode="fully_preserved",
        fps=24.0,
    ) -> io.NodeOutput:
        dev = torch.device("cpu")
        if mm is not None:
            try:
                dev = mm.get_torch_device()
            except Exception:
                pass

        def _work(device: torch.device) -> tuple[torch.Tensor, str, str]:
            try:
                video = load_exr_sequence(exr_sequence)
            except DCCSequenceError as exc:
                raise ValueError(str(exc)) from exc
            peak = float(video.max()) if video.numel() else 0.0
            payload = build_retention_payload(
                sim_type=sim_type,
                interaction_mode=interaction_mode,
                video_index=1,
                fps=float(fps),
            )
            ret = retention_json_text(payload)
            report = (
                f"Loaded {int(video.shape[0])} EXR frames {video.shape[2]}x{video.shape[1]} "
                f"peak={peak:.4g} sim={sim_type} mode={interaction_mode} — "
                f"subimage=0 flat-RGB only (>3ch unsupported); "
                f"{payload['video_label']} retention lines emitted; no clamp on HDR path."
            )
            return video.to(device), ret, report

        video, ret, report = run_with_cpu_fallback(_work, device=dev, label="MiniMaxH3_DCCBridge")
        return io.NodeOutput(video, ret, report)
