"""P2 — Sigma inspector: report table + sigma_json UI for the live plot widget."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import torch

from comfy_api.latest import io, ui

_PKG = Path(__file__).resolve().parents[1]
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from mmx_utils.sigma_schedule import (
    TRAINED_SHIFT_AUDIO,
    TRAINED_SHIFT_VIDEO,
    build_sigma_inspector_json,
    format_sigma_inspector_table,
)


class MiniMaxH3_SigmaInspector(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MiniMaxH3_SigmaInspector",
            display_name="H3 Sigma Inspector",
            category="MiniMax H3/Sampling",
            description=(
                "Read-only debug table: per-step sigma_v, sigma_a, dσa/dσv, and effective "
                "video denoise. sigma_json feeds the live plot widget; shift widgets "
                "recompute the curve client-side without re-queueing."
            ),
            inputs=[
                io.Sigmas.Input("sigmas"),
                io.Float.Input("shift_video", default=TRAINED_SHIFT_VIDEO, min=0.01, max=100.0, step=0.01),
                io.Float.Input("shift_audio", default=TRAINED_SHIFT_AUDIO, min=0.01, max=100.0, step=0.01),
            ],
            outputs=[
                io.String.Output("report"),
                io.String.Output("sigma_json"),
            ],
        )

    @classmethod
    def fingerprint_inputs(cls, sigmas, shift_video=TRAINED_SHIFT_VIDEO, shift_audio=TRAINED_SHIFT_AUDIO):
        if isinstance(sigmas, torch.Tensor):
            raw = sigmas.detach().cpu().numpy().tobytes()
        else:
            raw = str(sigmas).encode()
        return hashlib.md5(raw).hexdigest() + f"|{shift_video}|{shift_audio}"

    @classmethod
    def execute(cls, sigmas, shift_video=TRAINED_SHIFT_VIDEO, shift_audio=TRAINED_SHIFT_AUDIO) -> io.NodeOutput:
        t = sigmas if isinstance(sigmas, torch.Tensor) else torch.tensor(sigmas, dtype=torch.float32)
        sv = float(shift_video)
        sa = float(shift_audio)
        report = format_sigma_inspector_table(t, shift_video=sv, shift_audio=sa)
        sigma_json = build_sigma_inspector_json(t, shift_video=sv, shift_audio=sa)
        return io.NodeOutput(report, sigma_json, ui={"mmx_sigma": [sigma_json]})
