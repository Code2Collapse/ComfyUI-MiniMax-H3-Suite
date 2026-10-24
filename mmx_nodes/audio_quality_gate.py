"""P12 — Post-sampling audio quality gate (mel L1, centroid, RMS).

Mel spectrograms are used here as *measurement* for QC only. Invariant 8
("no mel-spectrogram conditioning") forbids feeding mel into the H3 model —
this node never touches the sampler or latent audio stream.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import torch

from comfy_api.latest import io

_PKG = Path(__file__).resolve().parents[1]
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from mmx_utils.audio_quality_gate import evaluate_audio_quality
from mmx_utils.device import run_with_cpu_fallback

try:
    import comfy.model_management as mm
except Exception:  # not just ImportError — comfy_kitchen skew raises AttributeError
    mm = None


class MiniMaxH3_AudioQualityGate(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MiniMaxH3_AudioQualityGate",
            display_name="H3 Audio Quality Gate",
            category="MiniMax H3/Sampling",
            description=(
                "Post-sampling QC on the decoded audio stream: mel L1 vs reference, "
                "spectral-centroid drift (EasyCache muffled-bass signature), and RMS. "
                "Fails the shot when bass collapses or the centroid drifts. "
                "Mel here is measurement-only — not model conditioning."
            ),
            inputs=[
                io.Audio.Input("reference", tooltip="Reference stem (e.g. isolated vocal)."),
                io.Audio.Input("generated", tooltip="Decoded audio from the sampler/VAE."),
                io.Float.Input(
                    "mel_l1_max",
                    default=0.35,
                    min=0.0,
                    max=2.0,
                    step=0.01,
                    tooltip="Mean |mel_ref - mel_gen| above this fails.",
                ),
                io.Float.Input(
                    "centroid_drift_hz_max",
                    default=800.0,
                    min=0.0,
                    max=8000.0,
                    step=10.0,
                    tooltip="Spectral-centroid drift (Hz) above this fails.",
                ),
                io.Float.Input(
                    "rms_ratio_min",
                    default=0.55,
                    min=0.0,
                    max=2.0,
                    step=0.01,
                    tooltip="generated_rms / reference_rms below this fails.",
                ),
                io.Float.Input(
                    "low_mel_ratio_min",
                    default=0.45,
                    min=0.0,
                    max=2.0,
                    step=0.01,
                    tooltip="Low-mel energy ratio below this fails (bass collapse).",
                ),
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
                io.String.Output("report"),
                io.Audio.Output("generated_out", tooltip="The checked audio, unchanged: wire it on to your Save so "
                                                          "a failed gate (on_fail = stop the run) stops before it."),
            ],
            is_output_node=True,
        )

    @classmethod
    def execute(
        cls,
        reference,
        generated,
        mel_l1_max=0.35,
        centroid_drift_hz_max=800.0,
        rms_ratio_min=0.55,
        low_mel_ratio_min=0.45,
        on_fail="report only",
    ) -> io.NodeOutput:
        ref_wf = reference["waveform"]
        gen_wf = generated["waveform"]
        sr = int(reference.get("sample_rate", generated.get("sample_rate", 32_000)))
        gen_sr = int(generated.get("sample_rate", sr))
        if gen_sr != sr:
            raise ValueError(
                f"Audio Quality Gate: sample_rate mismatch reference={sr} generated={gen_sr}"
            )

        dev = ref_wf.device
        if mm is not None:
            try:
                dev = mm.get_torch_device()
            except Exception:
                dev = ref_wf.device

        def _work(device: torch.device):
            return evaluate_audio_quality(
                ref_wf.to(device),
                gen_wf.to(device),
                sample_rate=sr,
                device=device,
                mel_l1_max=float(mel_l1_max),
                centroid_drift_hz_max=float(centroid_drift_hz_max),
                rms_ratio_min=float(rms_ratio_min),
                low_mel_ratio_min=float(low_mel_ratio_min),
            )

        result = run_with_cpu_fallback(_work, device=dev, label="MiniMaxH3_AudioQualityGate")
        report = result.report
        if not result.passed:
            report = "FAILED — " + report
            if on_fail == "stop the run":
                raise ValueError(f"H3 Audio Quality Gate stopped the run: {report}")
        return io.NodeOutput(bool(result.passed), report, generated, ui={"text": [report]})
