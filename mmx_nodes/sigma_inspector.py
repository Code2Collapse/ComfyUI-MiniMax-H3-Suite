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
    synthesize_linear_sigmas,
)


class MiniMaxH3_SigmaInspector(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MiniMaxH3_SigmaInspector",
            display_name="H3 Sigma Inspector",
            category="MiniMax H3/Sampling",
            description=(
                "Read-only debug table: per-step sigma_v, sigma_a, analytic dσa/dσv "
                "(shift-invariant), finite-difference dσa/dσv (what the sampler steps), "
                "and carry factor σv/σa. Wire BasicScheduler sigmas or leave sigmas "
                "unconnected and set steps to synthesise a linear schedule. sigma_json "
                "feeds the live plot widget."
            ),
            inputs=[
                io.Sigmas.Input(
                    "sigmas",
                    optional=True,
                    tooltip="When connected, inspect this schedule. Otherwise synthesised from steps.",
                ),
                io.Int.Input("steps", default=20, min=2, max=200),
                io.Float.Input("shift_video", default=TRAINED_SHIFT_VIDEO, min=0.01, max=100.0, step=0.01),
                io.Float.Input("shift_audio", default=TRAINED_SHIFT_AUDIO, min=0.01, max=100.0, step=0.01),
            ],
            outputs=[
                io.String.Output("report"),
                io.String.Output("sigma_json"),
            ],
        )

    @classmethod
    def fingerprint_inputs(
        cls,
        sigmas=None,
        steps=20,
        shift_video=TRAINED_SHIFT_VIDEO,
        shift_audio=TRAINED_SHIFT_AUDIO,
    ):
        if isinstance(sigmas, torch.Tensor):
            raw = sigmas.detach().cpu().numpy().tobytes()
        elif sigmas is not None:
            raw = str(sigmas).encode()
        else:
            raw = f"steps:{int(steps)}".encode()
        return hashlib.md5(raw).hexdigest() + f"|{shift_video}|{shift_audio}"

    @classmethod
    def execute(
        cls,
        sigmas=None,
        steps=20,
        shift_video=TRAINED_SHIFT_VIDEO,
        shift_audio=TRAINED_SHIFT_AUDIO,
    ) -> io.NodeOutput:
        if sigmas is None:
            t = synthesize_linear_sigmas(int(steps))
        else:
            t = sigmas if isinstance(sigmas, torch.Tensor) else torch.tensor(sigmas, dtype=torch.float32)
        sv = float(shift_video)
        sa = float(shift_audio)
        report = format_sigma_inspector_table(t, shift_video=sv, shift_audio=sa)
        sigma_json = build_sigma_inspector_json(t, shift_video=sv, shift_audio=sa)
        return io.NodeOutput(report, sigma_json, ui={"mmx_sigma": [sigma_json]})
