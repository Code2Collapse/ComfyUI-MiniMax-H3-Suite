"""P3 — Legal H3 scheduler validation (sigmas passthrough)."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import torch

from comfy_api.latest import io

_PKG = Path(__file__).resolve().parents[1]
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from mmx_utils.sigma_schedule import validate_h3_sigmas, validate_turbo_step_contract
from mmx_utils.turbo_lora import read_turbo_step_contract


class MiniMaxH3_LegalScheduler(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MiniMaxH3_LegalScheduler",
            display_name="H3 Legal Scheduler",
            category="MiniMax H3/Sampling",
            description=(
                "Validate a sigma schedule for H3 sampling. Rejects SplitSigmas / bong_tangent. "
                "Warns on non-simple schedulers and partial denoise. When optional model is wired "
                "from MiniMaxH3_TurboLoRA, enforces the turbo step contract (4/8/16). "
                "Passes sigmas through unchanged when legal. "
                "Wire BasicScheduler sigmas → this → SamplerCustomAdvanced."
            ),
            inputs=[
                io.Sigmas.Input("sigmas"),
                io.Model.Input(
                    "model",
                    optional=True,
                    tooltip="Wire from TurboLoRA to enforce minimax_h3_turbo_step_contract.",
                ),
                io.Combo.Input(
                    "scheduler",
                    options=["simple", "sgm_uniform", "karras", "exponential", "beta"],
                    default="simple",
                ),
                io.Float.Input("denoise", default=1.0, min=0.0, max=1.0, step=0.01),
            ],
            outputs=[
                io.Sigmas.Output("sigmas"),
                io.String.Output("report"),
            ],
        )

    @classmethod
    def fingerprint_inputs(cls, sigmas, model=None, scheduler="simple", denoise=1.0):
        if isinstance(sigmas, torch.Tensor):
            raw = sigmas.detach().cpu().numpy().tobytes()
        else:
            raw = str(sigmas).encode()
        return hashlib.md5(raw).hexdigest() + f"|{scheduler}|{denoise}"

    @classmethod
    def execute(cls, sigmas, model=None, scheduler="simple", denoise=1.0) -> io.NodeOutput:
        t = sigmas if isinstance(sigmas, torch.Tensor) else torch.tensor(sigmas, dtype=torch.float32)
        ok, report = validate_h3_sigmas(t, scheduler=scheduler, denoise=denoise)
        extra: list[str] = []
        contract = read_turbo_step_contract(model)
        if contract is not None:
            tok, turbo_notes = validate_turbo_step_contract(t, contract)
            extra.extend(turbo_notes)
            if not tok:
                ok = False
        if extra:
            report = report + "\n" + "\n".join(extra)
        if not ok:
            raise ValueError(report)
        return io.NodeOutput(t, report)
