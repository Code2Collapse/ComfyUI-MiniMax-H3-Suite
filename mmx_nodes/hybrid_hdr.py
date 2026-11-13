"""Apply extracted Hybrid HDR LoRA to a Ref2VA MODEL at constant strength."""

from __future__ import annotations

import sys
from pathlib import Path

from comfy_api.latest import io

_PKG = Path(__file__).resolve().parents[1]
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from mmx_utils.h3_scheduled_adapter import APPLY_TO, CURVES, Schedule
from mmx_utils.hybrid_hdr import apply_hybrid_hdr_lora, apply_hybrid_hdr_scheduled

MODES = ("constant merge", "scheduled (experimental)")

try:
    import folder_paths
except Exception:  # noqa: BLE001
    folder_paths = None  # type: ignore[assignment]


def _lora_names() -> list[str]:
    if folder_paths is None:
        return []
    try:
        return list(folder_paths.get_filename_list("loras"))
    except Exception:
        return []


class MiniMaxH3_HybridHDR(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        names = _lora_names() or ["<no loras folder>"]
        return io.Schema(
            node_id="MiniMaxH3_HybridHDR",
            display_name="H3 Hybrid HDR LoRA",
            category="MiniMax H3/Sampling",
            description=(
                "Apply the extracted Hybrid HDR LoRA (a checkpoint-delta experiment, not proven HDR-specific) "
                "to a Ref2VA MODEL.\n\n"
                "mode 'constant merge' (default): fixed strength, as before. For late-step HDR with this mode, "
                "sample the first steps with plain Ref2VA, then continue with this patched MODEL "
                "(KSamplerAdvanced / SamplerCustomAdvanced start/end steps - not SplitSigmas on H3).\n"
                "mode 'scheduled (experimental)': one sampler run; the strength follows the sampling sigma "
                "(start_sigma .. end_sigma with ramps). Not yet validated on a real H3 run."
            ),
            inputs=[
                io.Model.Input("model"),
                io.Combo.Input("lora_name", options=names, default=names[0]),
                io.Float.Input(
                    "strength",
                    default=1.0,
                    min=0.0,
                    max=2.0,
                    step=0.01,
                    tooltip="LoRA merge strength. Try 0.5 for a gentle HDR lift.",
                ),
                io.Int.Input(
                    "block_start",
                    default=25,
                    min=0,
                    max=49,
                    tooltip="First DiT block to patch (inclusive). Singularity merge used b25-49.",
                ),
                io.Int.Input(
                    "block_end",
                    default=49,
                    min=0,
                    max=49,
                    tooltip="Last DiT block to patch (inclusive).",
                ),
                io.Combo.Input("mode", options=list(MODES), default=MODES[0], optional=True,
                               tooltip="constant merge = fixed strength (unchanged behaviour). scheduled = "
                                       "strength follows the sampling sigma in one run (experimental)."),
                io.Float.Input("start_sigma", default=1.0, min=0.0, max=1.0, step=0.01, optional=True,
                               tooltip="scheduled: the adapter starts acting at this sigma (sampling runs from "
                                       "1 down to 0). 1 = from the first step."),
                io.Float.Input("end_sigma", default=0.0, min=0.0, max=1.0, step=0.01, optional=True,
                               tooltip="scheduled: the adapter stops below this sigma. 0 = to the last step."),
                io.Float.Input("ramp_in", default=0.0, min=0.0, max=1.0, step=0.01, optional=True,
                               tooltip="scheduled: sigma span after start_sigma over which strength rises from 0."),
                io.Float.Input("ramp_out", default=0.0, min=0.0, max=1.0, step=0.01, optional=True,
                               tooltip="scheduled: sigma span before end_sigma over which strength falls to 0."),
                io.Combo.Input("schedule_curve", options=list(CURVES), default=CURVES[0], optional=True,
                               tooltip="scheduled: shape of the ramps."),
                io.Combo.Input("apply_to", options=list(APPLY_TO), default=APPLY_TO[0], optional=True,
                               tooltip="scheduled: 'all rows' = as the adapter was made. 'text + visual rows' = "
                                       "the adapter adds nothing to audio rows (audio can still change slightly "
                                       "through attention to the adapted video)."),
                io.Boolean.Input("include_global_parameters", default=True, optional=True,
                                 tooltip="Also apply LoRA layers that belong to no DiT block (embeddings, refiner, "
                                         "output heads). On = the previous behaviour; the block range never "
                                         "filtered them."),
            ],
            outputs=[
                io.Model.Output("model"),
                io.String.Output("report"),
            ],
        )

    @classmethod
    def fingerprint_inputs(cls, model=None, lora_name=None, strength=1.0, block_start=25, block_end=49, **kwargs):
        extra = "|".join(f"{k}={kwargs[k]}" for k in sorted(kwargs))
        return f"{lora_name}|{strength}|{block_start}|{block_end}|{extra}"

    @classmethod
    def execute(cls, model, lora_name, strength=1.0, block_start=25, block_end=49, mode=MODES[0],
                start_sigma=1.0, end_sigma=0.0, ramp_in=0.0, ramp_out=0.0, schedule_curve=CURVES[0],
                apply_to=APPLY_TO[0], include_global_parameters=True, **_ignored) -> io.NodeOutput:
        if folder_paths is None:
            raise RuntimeError(
                "MiniMaxH3_HybridHDR needs ComfyUI folder_paths to resolve LoRA files."
            )
        if int(block_start) > int(block_end):
            raise ValueError("block_start must be <= block_end")
        path = folder_paths.get_full_path("loras", lora_name)
        if mode == MODES[1]:
            sched = Schedule(float(strength), float(start_sigma), float(end_sigma), float(ramp_in), float(ramp_out),
                             str(schedule_curve)).validate()
            patched, report = apply_hybrid_hdr_scheduled(model, path, sched, int(block_start), int(block_end),
                                                         str(apply_to), bool(include_global_parameters))
        else:
            patched, report = apply_hybrid_hdr_lora(
                model,
                path,
                float(strength),
                int(block_start),
                int(block_end),
                bool(include_global_parameters),
            )
        return io.NodeOutput(patched, report)
