# PORTED FROM: ComfyUI-Spectrum-MiniMax-H3 :: comfyui_spectrum_h3/nodes.py,
#   objective_media_nodes.py — author xmarre, GPL-3.0
"""V3 ComfyNode wrappers for Spectrum H3 forecasting and objective-media benchmarks."""

from __future__ import annotations

import sys
from pathlib import Path

from comfy_api.latest import io

_PKG = Path(__file__).resolve().parents[1]
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from mmx_utils.spectrum_h3.nodes import SpectrumApplyMiniMaxH3
from mmx_utils.spectrum_h3.objective_media_nodes import (
    DEFAULT_PROVENANCE_JSON,
    OBJECTIVE_MEDIA_TYPE,
    SpectrumH3ObjectiveCaptureReset,
    SpectrumH3ObjectiveMediaStage,
    SpectrumH3ObjectiveQualityCompare,
    SpectrumH3ObjectiveSequentialCapture,
    SpectrumH3ObjectiveStagedQualityCompare,
)

_CATEGORY = "MiniMax H3/Spectrum"
_OBJECTIVE_MEDIA = io.Custom(OBJECTIVE_MEDIA_TYPE)
_ROLE_OPTIONS = (
    "R - native reference",
    "A - legacy Spectrum",
    "B - candidate",
)

_apply = SpectrumApplyMiniMaxH3()
_stage = SpectrumH3ObjectiveMediaStage()
_compare = SpectrumH3ObjectiveQualityCompare()
_staged_compare = SpectrumH3ObjectiveStagedQualityCompare()
_sequential = SpectrumH3ObjectiveSequentialCapture()
_reset = SpectrumH3ObjectiveCaptureReset()


class MiniMaxH3_SpectrumApply(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MiniMaxH3_SpectrumApply",
            display_name="Spectrum Apply MiniMax H3",
            category=_CATEGORY,
            description=(
                "Install Spectrum H3 spectral forecasting on a native MiniMax-H3 model. "
                "Clones the model and patches sampler wrappers without mutating the input."
            ),
            inputs=[
                io.Model.Input("model"),
                io.Boolean.Input("enabled", default=True),
                io.Float.Input(
                    "blend_weight",
                    default=0.50,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip=(
                        "Direct video spectral share. Audio uses audio_blend_weight. "
                        "Video forecasts can still affect later audio through joint transformer calls."
                    ),
                ),
                io.Int.Input("degree", default=1, min=1, max=16, step=1),
                io.Float.Input("ridge_lambda", default=0.10, min=0.0, max=10.0, step=0.01),
                io.Float.Input("window_size", default=2.0, min=1.0, max=16.0, step=0.05),
                io.Float.Input("flex_window", default=0.75, min=0.0, max=8.0, step=0.05),
                io.Int.Input("warmup_steps", default=1, min=0, max=64, step=1),
                io.Int.Input("tail_actual_steps", default=1, min=0, max=64, step=1),
                io.Int.Input("max_history", default=8, min=2, max=64, step=1),
                io.Boolean.Input("debug", default=False),
                io.Combo.Input(
                    "history_storage",
                    options=["system_ram", "vram"],
                    default="system_ram",
                    optional=True,
                ),
                io.Boolean.Input("bootstrap_first_forecast", default=True, optional=True),
                io.Boolean.Input("anchor_residual_feedback", default=False, optional=True),
                io.Boolean.Input("selective_rollback_correction", default=False, optional=True),
                io.Boolean.Input("offline_smoothing_replay", default=True, optional=True),
                io.Float.Input(
                    "audio_blend_weight",
                    default=0.0,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    optional=True,
                ),
                io.Combo.Input(
                    "offline_archive_storage",
                    options=["system_ram", "vram"],
                    default="system_ram",
                    optional=True,
                ),
                io.Combo.Input(
                    "model_aware_mode",
                    options=["off", "schedule", "schedule_confidence", "full"],
                    default="off",
                    optional=True,
                ),
                io.Float.Input(
                    "model_aware_risk_threshold",
                    default=0.65,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    optional=True,
                ),
                io.Boolean.Input("model_aware_trust_shrinkage", default=False, optional=True),
                io.Boolean.Input(
                    "model_aware_replay_generic_correction",
                    default=False,
                    optional=True,
                ),
                io.Combo.Input(
                    "generic_correction_mode",
                    options=[
                        "legacy",
                        "coordinate_rls",
                        "coordinate_rls_reliability",
                        "regional",
                    ],
                    default="coordinate_rls",
                    optional=True,
                ),
                io.Combo.Input(
                    "generic_correction_limiter",
                    options=["rational", "hard_clip", "tanh"],
                    default="hard_clip",
                    optional=True,
                ),
                io.Float.Input(
                    "generic_correction_limit",
                    default=0.40,
                    min=0.01,
                    max=1.0,
                    step=0.01,
                    optional=True,
                ),
                io.Combo.Input(
                    "generic_correction_attenuation",
                    options=[
                        "mode_default",
                        "no_attenuation",
                        "general_confidence",
                        "correction_reliability",
                        "combined_conservative",
                    ],
                    default="no_attenuation",
                    optional=True,
                ),
            ],
            outputs=[io.Model.Output("model")],
        )

    @classmethod
    def execute(
        cls,
        model,
        enabled=True,
        blend_weight=0.50,
        degree=1,
        ridge_lambda=0.10,
        window_size=2.0,
        flex_window=0.75,
        warmup_steps=1,
        tail_actual_steps=1,
        max_history=8,
        debug=False,
        history_storage="system_ram",
        bootstrap_first_forecast=True,
        anchor_residual_feedback=False,
        selective_rollback_correction=False,
        offline_smoothing_replay=True,
        audio_blend_weight=0.0,
        offline_archive_storage="system_ram",
        model_aware_mode="off",
        model_aware_risk_threshold=0.65,
        model_aware_trust_shrinkage=False,
        model_aware_replay_generic_correction=False,
        generic_correction_mode="coordinate_rls",
        generic_correction_limiter="hard_clip",
        generic_correction_limit=0.40,
        generic_correction_attenuation="no_attenuation",
    ) -> io.NodeOutput:
        return io.NodeOutput(
            *_apply.apply(
                model,
                enabled,
                blend_weight,
                degree,
                ridge_lambda,
                window_size,
                flex_window,
                warmup_steps,
                tail_actual_steps,
                max_history,
                debug,
                history_storage=history_storage,
                bootstrap_first_forecast=bootstrap_first_forecast,
                anchor_residual_feedback=anchor_residual_feedback,
                selective_rollback_correction=selective_rollback_correction,
                offline_smoothing_replay=offline_smoothing_replay,
                audio_blend_weight=audio_blend_weight,
                offline_archive_storage=offline_archive_storage,
                model_aware_mode=model_aware_mode,
                model_aware_risk_threshold=model_aware_risk_threshold,
                model_aware_trust_shrinkage=model_aware_trust_shrinkage,
                model_aware_replay_generic_correction=model_aware_replay_generic_correction,
                generic_correction_mode=generic_correction_mode,
                generic_correction_limiter=generic_correction_limiter,
                generic_correction_limit=generic_correction_limit,
                generic_correction_attenuation=generic_correction_attenuation,
            )
        )


class MiniMaxH3_ObjectiveMediaStage(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MiniMaxH3_ObjectiveMediaStage",
            display_name="Spectrum H3 Objective Media Stage (One-Shot / Full Media)",
            category=_CATEGORY,
            description=SpectrumH3ObjectiveMediaStage.DESCRIPTION,
            inputs=[
                io.Image.Input("video"),
                io.Audio.Input("audio", optional=True),
            ],
            outputs=[_OBJECTIVE_MEDIA.Output("staged_media")],
        )

    @classmethod
    def execute(cls, video, audio=None) -> io.NodeOutput:
        return io.NodeOutput(_stage.stage(video, audio)[0])


class MiniMaxH3_ObjectiveQualityCompare(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MiniMaxH3_ObjectiveQualityCompare",
            display_name="Spectrum H3 Objective Quality Compare (One-Shot / Full Media)",
            category=_CATEGORY,
            description=SpectrumH3ObjectiveQualityCompare.DESCRIPTION,
            is_output_node=True,
            inputs=[
                io.Image.Input("reference_video"),
                io.Image.Input("legacy_video"),
                io.Image.Input("candidate_video"),
                io.Float.Input("fps", default=24.0, min=0.01, max=240.0, step=0.01),
                io.String.Input("benchmark_id", default="h3-objective-seed-1"),
                io.Int.Input("seed", default=0, min=0, max=0xFFFFFFFFFFFFFFFF),
                io.String.Input(
                    "provenance_json",
                    default=DEFAULT_PROVENANCE_JSON,
                    multiline=True,
                ),
                io.Int.Input("frame_chunk_size", default=4, min=1, max=32, step=1),
                io.Audio.Input("reference_audio", optional=True),
                io.Audio.Input("legacy_audio", optional=True),
                io.Audio.Input("candidate_audio", optional=True),
            ],
            outputs=[
                io.String.Output("summary"),
                io.String.Output("report_json_path"),
                io.String.Output("report_markdown_path"),
                io.String.Output("aggregate_json_path"),
                io.String.Output("aggregate_markdown_path"),
            ],
        )

    @classmethod
    def execute(
        cls,
        reference_video,
        legacy_video,
        candidate_video,
        fps,
        benchmark_id,
        seed,
        provenance_json,
        frame_chunk_size,
        reference_audio=None,
        legacy_audio=None,
        candidate_audio=None,
    ) -> io.NodeOutput:
        return io.NodeOutput(
            *_compare.compare(
                reference_video,
                legacy_video,
                candidate_video,
                fps,
                benchmark_id,
                seed,
                provenance_json,
                frame_chunk_size,
                reference_audio=reference_audio,
                legacy_audio=legacy_audio,
                candidate_audio=candidate_audio,
            )
        )


class MiniMaxH3_ObjectiveStagedQualityCompare(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MiniMaxH3_ObjectiveStagedQualityCompare",
            display_name="Spectrum H3 Objective Quality Compare (Staged One-Shot / Full Media)",
            category=_CATEGORY,
            description=SpectrumH3ObjectiveStagedQualityCompare.DESCRIPTION,
            is_output_node=True,
            inputs=[
                _OBJECTIVE_MEDIA.Input("reference_media"),
                _OBJECTIVE_MEDIA.Input("legacy_media"),
                _OBJECTIVE_MEDIA.Input("candidate_media"),
                io.Float.Input("fps", default=24.0, min=0.01, max=240.0, step=0.01),
                io.String.Input("benchmark_id", default="h3-objective-seed-1"),
                io.Int.Input("seed", default=0, min=0, max=0xFFFFFFFFFFFFFFFF),
                io.String.Input(
                    "provenance_json",
                    default=DEFAULT_PROVENANCE_JSON,
                    multiline=True,
                ),
                io.Int.Input("frame_chunk_size", default=4, min=1, max=32, step=1),
            ],
            outputs=MiniMaxH3_ObjectiveQualityCompare.define_schema().outputs,
        )

    @classmethod
    def execute(
        cls,
        reference_media,
        legacy_media,
        candidate_media,
        fps,
        benchmark_id,
        seed,
        provenance_json,
        frame_chunk_size,
    ) -> io.NodeOutput:
        return io.NodeOutput(
            *_staged_compare.compare(
                reference_media,
                legacy_media,
                candidate_media,
                fps,
                benchmark_id,
                seed,
                provenance_json,
                frame_chunk_size,
            )
        )


class MiniMaxH3_ObjectiveSequentialCapture(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MiniMaxH3_ObjectiveSequentialCapture",
            display_name="Spectrum H3 Objective Media Capture (Sequential - Bounded)",
            category=_CATEGORY,
            description=SpectrumH3ObjectiveSequentialCapture.DESCRIPTION,
            is_output_node=True,
            not_idempotent=True,
            inputs=[
                io.Image.Input("video"),
                io.Combo.Input("role", options=list(_ROLE_OPTIONS), default=_ROLE_OPTIONS[0]),
                io.Float.Input("fps", default=24.0, min=0.01, max=240.0, step=0.01),
                io.String.Input("benchmark_id", default="h3-objective-seed-1"),
                io.Int.Input("generation_seed", force_input=True, min=0, max=0xFFFFFFFFFFFFFFFF),
                io.Int.Input("steps", default=20, min=1, max=1000, step=1),
                io.String.Input(
                    "compatibility_tag",
                    default="minimax-h3-er-sde-current-workflow",
                ),
                io.Int.Input("frame_chunk_size", default=4, min=1, max=32, step=1),
                io.Boolean.Input("reset_before_capture", default=False),
                io.Audio.Input("audio", optional=True),
            ],
            outputs=MiniMaxH3_ObjectiveQualityCompare.define_schema().outputs,
        )

    @classmethod
    def execute(
        cls,
        video,
        role,
        fps,
        benchmark_id,
        generation_seed,
        steps,
        compatibility_tag,
        frame_chunk_size,
        reset_before_capture=False,
        audio=None,
    ) -> io.NodeOutput:
        return io.NodeOutput(
            *_sequential.capture(
                video,
                role,
                fps,
                benchmark_id,
                generation_seed,
                steps,
                compatibility_tag,
                frame_chunk_size,
                reset_before_capture=reset_before_capture,
                audio=audio,
            )
        )


class MiniMaxH3_ObjectiveCaptureReset(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MiniMaxH3_ObjectiveCaptureReset",
            display_name="Spectrum H3 Objective Media Capture Reset",
            category=_CATEGORY,
            description=SpectrumH3ObjectiveCaptureReset.DESCRIPTION,
            is_output_node=True,
            not_idempotent=True,
            inputs=[
                io.String.Input("benchmark_id", default="h3-objective-seed-1"),
                io.Combo.Input("scope", options=["benchmark", "all"], default="benchmark"),
            ],
            outputs=[io.String.Output("summary")],
        )

    @classmethod
    def execute(cls, benchmark_id, scope) -> io.NodeOutput:
        return io.NodeOutput(_reset.clear(benchmark_id, scope)[0])


__all__ = [
    "MiniMaxH3_ObjectiveCaptureReset",
    "MiniMaxH3_ObjectiveMediaStage",
    "MiniMaxH3_ObjectiveQualityCompare",
    "MiniMaxH3_ObjectiveSequentialCapture",
    "MiniMaxH3_ObjectiveStagedQualityCompare",
    "MiniMaxH3_SpectrumApply",
]
