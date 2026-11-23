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
                io.Model.Input(
                    "model",
                    tooltip=(
                        "Native MiniMax-H3 MODEL. When enabled, it is cloned and the clone gets the Spectrum "
                        "sampler and transformer wrappers; the input model is not modified. "
                        "A model that is not native H3 raises an error."
                    ),
                ),
                io.Boolean.Input(
                    "enabled",
                    default=True,
                    tooltip=(
                        "Off returns the input model unchanged (no clone, no patch). On: on some sampler steps "
                        "the H3 transformer is skipped and its final hidden feature is forecast from earlier actual steps."
                    ),
                ),
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
                io.Int.Input("degree", default=1, min=1, max=16, step=1, extra_dict={'c2c_section': 'Forecast', 'c2c_show_if': {'enabled': [True]}},
                             tooltip=(
                                 "Degree of the Chebyshev ridge polynomial fitted over past actual steps to forecast the "
                                 "transformer feature. Needs at least degree+1 stored steps (max_history >= degree+1). "
                                 "Values other than 1 disable bootstrap_first_forecast."
                             )),
                io.Float.Input("ridge_lambda", default=0.10, min=0.0, max=10.0, step=0.01, extra_dict={'c2c_section': 'Forecast', 'c2c_show_if': {'enabled': [True]}},
                               tooltip=(
                                   "Ridge (L2) regularisation added to the polynomial fit. 0 = plain least squares; larger "
                                   "values pull the fitted coefficients toward zero. model_aware_mode schedule_confidence "
                                   "and full may adapt it per step."
                               )),
                io.Float.Input("window_size", default=2.0, min=1.0, max=16.0, step=0.05, extra_dict={'c2c_section': 'Forecast', 'c2c_show_if': {'enabled': [True]}},
                               tooltip=(
                                   "Starting forecast interval in steps: once history is ready, every floor(window)-th step "
                                   "is an actual transformer call and the steps between are forecast. 1 = every step actual "
                                   "(nothing skipped)."
                               )),
                io.Float.Input("flex_window", default=0.75, min=0.0, max=8.0, step=0.05, extra_dict={'c2c_section': 'Forecast', 'c2c_show_if': {'enabled': [True]}},
                               tooltip=(
                                   "Added to the current window after each scheduled actual step past warmup, capped at "
                                   "max(window_size, max_history). 0 keeps the interval fixed at window_size; larger values "
                                   "skip more steps later in sampling."
                               )),
                io.Int.Input("warmup_steps", default=1, min=0, max=64, step=1, extra_dict={'c2c_section': 'Forecast', 'c2c_show_if': {'enabled': [True]}},
                             tooltip=(
                                 "Number of first sampler steps that always run the real transformer before forecasting "
                                 "may start. Values above 1 disable bootstrap_first_forecast."
                             )),
                io.Int.Input("tail_actual_steps", default=1, min=0, max=64, step=1, extra_dict={'c2c_section': 'Forecast', 'c2c_show_if': {'enabled': [True]}},
                             tooltip=(
                                 "Number of final sampler steps forced to run the real transformer. Some samplers enforce "
                                 "their own minimum tail (RES uses three steps); ER-SDE offline replay may promote one "
                                 "extra penultimate step."
                             )),
                io.Int.Input("max_history", default=8, min=2, max=64, step=1, extra_dict={'c2c_section': 'Forecast', 'c2c_show_if': {'enabled': [True]}},
                             tooltip=(
                                 "How many past actual transformer features are kept for the fit (oldest dropped first). "
                                 "Must be >= degree+1. Also caps how far flex_window can grow the window. Memory use grows "
                                 "with it (see history_storage)."
                             )),
                io.Boolean.Input("debug", default=False, extra_dict={'c2c_section': 'Diagnostics', 'c2c_show_if': {'enabled': [True]}},
                                 tooltip=(
                                     "Logs per-run Spectrum summaries, replay transitions and skipped-probe reasons to the "
                                     "console. Does not change the result."
                                 )),
                io.Combo.Input(
                    "history_storage",
                    options=["system_ram", "vram"],
                    default="system_ram",
                    optional=True,
                    extra_dict={'c2c_section': 'Memory', 'c2c_show_if': {'enabled': [True]}},
                    tooltip=(
                        "Where the bounded feature history (max_history entries) is kept: system_ram = CPU memory "
                        "(copied on each actual step), vram = the model's GPU device. The offline replay archive uses "
                        "offline_archive_storage instead."
                    ),
                ),
                io.Boolean.Input("bootstrap_first_forecast", default=True, optional=True, extra_dict={'c2c_section': 'Correction', 'c2c_show_if': {'enabled': [True]}},
                                 tooltip=(
                                     "Lets sampler step 1 be forecast from the single actual step-0 feature. Applies only "
                                     "with degree 1 and warmup_steps <= 1; otherwise it is turned off with a console warning."
                                 )),
                io.Boolean.Input("anchor_residual_feedback", default=False, optional=True, extra_dict={'c2c_section': 'Correction', 'c2c_show_if': {'enabled': [True]}},
                                 tooltip=(
                                     "Experimental: on actual steps, measures the forecast error and may request extra "
                                     "actual refreshes (budgeted); never adds a hidden residual. Only one of this, "
                                     "selective_rollback_correction and offline_smoothing_replay may be on, else an error."
                                 )),
                io.Boolean.Input("selective_rollback_correction", default=False, optional=True, extra_dict={'c2c_section': 'Correction', 'c2c_show_if': {'enabled': [True]}},
                                 tooltip=(
                                     "Experimental: thresholded, budgeted rollback that re-runs steps whose forecast error "
                                     "was too large; reviewed for the deterministic Euler sampler only. Turn "
                                     "offline_smoothing_replay off first (the trajectory modes are mutually exclusive)."
                                 )),
                io.Boolean.Input("offline_smoothing_replay", default=True, optional=True, extra_dict={'c2c_section': 'Replay', 'c2c_show_if': {'enabled': [True]}},
                                 tooltip=(
                                     "Two-pass mode: the first sampler pass records every actual transformer feature, then "
                                     "a replay pass forecasts using past and future anchors, without video-to-audio "
                                     "feedback. Costs a second sampler pass plus the archive memory."
                                 )),
                io.Float.Input(
                    "audio_blend_weight",
                    default=0.0,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    optional=True,
                    extra_dict={'c2c_section': 'Forecast', 'c2c_show_if': {'enabled': [True]}},
                    tooltip=(
                        "Audio rows' forecast mix (0-1): share of the polynomial (spectral) forecast, the rest is "
                        "linear extrapolation from the last two actual steps. 0 = no spectral mixing of audio. "
                        "Video uses blend_weight."
                    ),
                ),
                io.Combo.Input(
                    "offline_archive_storage",
                    options=["system_ram", "vram"],
                    default="system_ram",
                    optional=True,
                    extra_dict={'c2c_section': 'Memory', 'c2c_show_if': {'enabled': [True], 'offline_smoothing_replay': [True]}},
                    tooltip=(
                        "Where offline_smoothing_replay keeps every actual feature until replay ends. Not capped by "
                        "max_history. system_ram = CPU memory; vram = GPU memory (faster, can run out on small GPUs)."
                    ),
                ),
                io.Combo.Input(
                    "model_aware_mode",
                    options=["off", "schedule", "schedule_confidence", "full"],
                    default="off",
                    optional=True,
                    extra_dict={'c2c_section': 'Model-aware', 'c2c_show_if': {'enabled': [True]}},
                    tooltip=(
                        "Experimental. off = fixed window schedule. schedule = risky forecasts become real transformer "
                        "calls. schedule_confidence = also adapts ridge, usable degree and blend share. full = also "
                        "applies the bounded generic correction (generic_correction_*), which adds no transformer call."
                    ),
                ),
                io.Float.Input(
                    "model_aware_risk_threshold",
                    default=0.65,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    optional=True,
                    extra_dict={'c2c_section': 'Model-aware', 'c2c_show_if': {'enabled': [True], 'model_aware_mode': ['schedule', 'schedule_confidence', 'full']}},
                    tooltip=(
                        "Risk score (0-1) at or above which a planned forecast is replaced by a real transformer call. "
                        "Lower = more real calls (slower, fewer forecasts). Ignored when model_aware_mode is off."
                    ),
                ),
                io.Boolean.Input("model_aware_trust_shrinkage", default=False, optional=True, extra_dict={'c2c_section': 'Model-aware', 'c2c_show_if': {'enabled': [True], 'model_aware_mode': ['full']}},
                                 tooltip=(
                                     "Experimental research switch: shrinks forecasts by a measured forecast-trust probe. "
                                     "Requires model_aware_mode='full' (other modes raise an error); in offline replay it "
                                     "only records shadow diagnostics. Adds no transformer call."
                                 )),
                io.Boolean.Input(
                    "model_aware_replay_generic_correction",
                    default=False,
                    optional=True,
                    extra_dict={'c2c_section': 'Model-aware', 'c2c_show_if': {'enabled': [True], 'model_aware_mode': ['full'], 'offline_smoothing_replay': [True]}},
                    tooltip=(
                        "Offline replay with model_aware_mode='full' only. Off = the replay pass does not reuse the causal "
                        "latest-delta correction. On = restores the older behaviour that applied it during replay "
                        "(kept for reproduction)."
                    ),
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
                    extra_dict={'c2c_section': 'Generic correction', 'c2c_show_if': {'enabled': [True], 'model_aware_mode': ['schedule', 'schedule_confidence', 'full']}},
                    tooltip=(
                        "Applied only with model_aware_mode='full': how the gain on the latest-delta correction is "
                        "estimated. legacy = previous EWMA path; coordinate_rls = recursive least squares scaled by step "
                        "spacing; coordinate_rls_reliability = adds a reliability factor; regional = experimental "
                        "per-region gains."
                    ),
                ),
                io.Combo.Input(
                    "generic_correction_limiter",
                    options=["rational", "hard_clip", "tanh"],
                    default="hard_clip",
                    optional=True,
                    extra_dict={'c2c_section': 'Generic correction', 'c2c_show_if': {'enabled': [True], 'model_aware_mode': ['schedule', 'schedule_confidence', 'full']}},
                    tooltip=(
                        "How the correction gain is bounded to +/- generic_correction_limit (model_aware_mode='full'). "
                        "hard_clip = clamp; tanh = smooth saturation; rational = g/(1+|g|/limit). rational with limit "
                        "0.25 and mode legacy reproduces the older exact path."
                    ),
                ),
                io.Float.Input(
                    "generic_correction_limit",
                    default=0.40,
                    min=0.01,
                    max=1.0,
                    step=0.01,
                    optional=True,
                    extra_dict={'c2c_section': 'Generic correction', 'c2c_show_if': {'enabled': [True], 'model_aware_mode': ['schedule', 'schedule_confidence', 'full']}},
                    tooltip=(
                        "Symmetric bound on the correction gain used by generic_correction_limiter (model_aware_mode='full'). "
                        "Smaller = smaller possible correction. Default 0.40; 0.25 with legacy/rational/mode_default gives "
                        "the older exact baseline."
                    ),
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
                    extra_dict={'c2c_section': 'Generic correction', 'c2c_show_if': {'enabled': [True], 'model_aware_mode': ['schedule', 'schedule_confidence', 'full']}},
                    tooltip=(
                        "Scales the correction gain down by confidence (model_aware_mode='full'). no_attenuation = none; "
                        "general_confidence = forecast confidence; correction_reliability = measured correction "
                        "reliability; combined_conservative = both; mode_default = per-mode choice (legacy keeps its own path)."
                    ),
                ),
            ],
            outputs=[
                io.Model.Output(
                    "model",
                    tooltip=(
                        "Patched clone of the H3 model with Spectrum forecasting installed (or the input model unchanged "
                        "when enabled is off). Wire it to the sampler."
                    ),
                )
            ],
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
                io.Image.Input(
                    "video",
                    tooltip=(
                        "Decoded IMAGE batch of one result (R, A or B). Copied to CPU at full resolution and held "
                        "for the Staged Quality Compare node."
                    ),
                ),
                io.Audio.Input(
                    "audio",
                    optional=True,
                    tooltip=(
                        "Optional decoded AUDIO of the same result. Must be a waveform [batch, channels, samples] "
                        "with finite values; copied to CPU as float32."
                    ),
                ),
            ],
            outputs=[
                _OBJECTIVE_MEDIA.Output(
                    "staged_media",
                    tooltip=(
                        "CPU copy of this result's video (and audio). Wire three of these into Objective Quality "
                        "Compare (Staged) as reference, legacy and candidate. Holds full decoded media in RAM."
                    ),
                )
            ],
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
                io.Image.Input(
                    "reference_video",
                    tooltip=(
                        "R: decoded IMAGE batch from native H3 with Spectrum bypassed (the reference). Must have the "
                        "same frame count and size as legacy_video and candidate_video."
                    ),
                ),
                io.Image.Input(
                    "legacy_video",
                    tooltip=(
                        "A: decoded IMAGE batch from the same workflow with the legacy Spectrum settings. Compared "
                        "against the reference; same shape as reference_video."
                    ),
                ),
                io.Image.Input(
                    "candidate_video",
                    tooltip=(
                        "B: decoded IMAGE batch from the same workflow with the candidate settings. The report gives "
                        "B's error relative to A's, both measured against R; same shape as reference_video."
                    ),
                ),
                io.Float.Input(
                    "fps", default=24.0, min=0.01, max=240.0, step=0.01,
                    tooltip=(
                        "Frame rate of the three videos (frames per second). Recorded in the report and used for the "
                        "duration in seconds; it does not resample anything."
                    ),
                ),
                io.String.Input(
                    "benchmark_id", default="h3-objective-seed-1",
                    tooltip=(
                        "Name of this R/A/B comparison; must not be empty. Reports with the same benchmark_id and seed "
                        "count once in the aggregate; use a new id per prompt/seed."
                    ),
                ),
                io.Int.Input(
                    "seed", default=0, min=0, max=0xFFFFFFFFFFFFFFFF,
                    tooltip=(
                        "Generation seed used for all three results. Only recorded as a label in the report; "
                        "(benchmark_id, seed) identifies one entry in the cross-seed aggregate."
                    ),
                ),
                io.String.Input(
                    "provenance_json",
                    default=DEFAULT_PROVENANCE_JSON,
                    multiline=True,
                    extra_dict={'c2c_section': 'Provenance'},
                    tooltip=(
                        "JSON object describing how R, A and B were generated (model, sampler, steps, VAE...). It is "
                        "validated and hashed: only reports with identical compatibility data share one aggregate. "
                        "The default is marked user-unverified."
                    ),
                ),
                io.Int.Input(
                    "frame_chunk_size", default=4, min=1, max=32, step=1, extra_dict={'c2c_section': 'Performance'},
                    tooltip=(
                        "Frames processed per chunk when computing the video metrics (clamped to 1-32). Larger = fewer "
                        "passes but more memory at once."
                    ),
                ),
                io.Audio.Input(
                    "reference_audio", optional=True,
                    tooltip=(
                        "Optional R audio. Audio metrics (MR-STFT, correlation, SI-SDR, lag) are computed only when "
                        "all three audio inputs are connected; one or two alone raise an error."
                    ),
                ),
                io.Audio.Input(
                    "legacy_audio", optional=True,
                    tooltip="Optional A audio; connect together with reference_audio and candidate_audio.",
                ),
                io.Audio.Input(
                    "candidate_audio", optional=True,
                    tooltip="Optional B audio; connect together with reference_audio and legacy_audio.",
                ),
            ],
            outputs=[
                io.String.Output(
                    "summary",
                    tooltip=(
                        "Text summary (also printed to the console): verdict, benchmark/seed/group, triad count and "
                        "B-vs-A video (and audio) metric advantages. Sequential capture returns a status line until "
                        "the third role arrives."
                    ),
                ),
                io.String.Output(
                    "report_json_path",
                    tooltip=(
                        "File path of this comparison's JSON report, written under the ComfyUI user cache "
                        "(spectrum_h3/objective_media/v1/runs/<group>/). Empty string when no report was written."
                    ),
                ),
                io.String.Output(
                    "report_markdown_path",
                    tooltip=(
                        "File path of the same report as Markdown, next to the JSON. Empty string when no report "
                        "was written."
                    ),
                ),
                io.String.Output(
                    "aggregate_json_path",
                    tooltip=(
                        "File path of the JSON aggregate over all compatible reports in this group (one per "
                        "benchmark_id/seed; newest 24 kept). Empty string when no report was written."
                    ),
                ),
                io.String.Output(
                    "aggregate_markdown_path",
                    tooltip=(
                        "File path of the aggregate as Markdown (under .../aggregates/). Empty string when no report "
                        "was written."
                    ),
                ),
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
                _OBJECTIVE_MEDIA.Input(
                    "reference_media",
                    tooltip=(
                        "R: staged_media from a Media Stage node fed by native H3 with Spectrum bypassed. Audio is "
                        "compared when all three media carry audio; some but not all raises an error."
                    ),
                ),
                _OBJECTIVE_MEDIA.Input(
                    "legacy_media",
                    tooltip="A: staged_media of the legacy Spectrum result; same video shape as reference_media.",
                ),
                _OBJECTIVE_MEDIA.Input(
                    "candidate_media",
                    tooltip="B: staged_media of the candidate result; same video shape as reference_media.",
                ),
                io.Float.Input(
                    "fps", default=24.0, min=0.01, max=240.0, step=0.01,
                    tooltip=(
                        "Frame rate of the staged videos (frames per second). Recorded in the report and used for "
                        "the duration in seconds; it does not resample anything."
                    ),
                ),
                io.String.Input(
                    "benchmark_id", default="h3-objective-seed-1",
                    tooltip=(
                        "Name of this R/A/B comparison; must not be empty. Reports with the same benchmark_id and seed "
                        "count once in the aggregate; use a new id per prompt/seed."
                    ),
                ),
                io.Int.Input(
                    "seed", default=0, min=0, max=0xFFFFFFFFFFFFFFFF,
                    tooltip=(
                        "Generation seed used for all three results. Only recorded as a label in the report; "
                        "(benchmark_id, seed) identifies one entry in the cross-seed aggregate."
                    ),
                ),
                io.String.Input(
                    "provenance_json",
                    default=DEFAULT_PROVENANCE_JSON,
                    multiline=True,
                    extra_dict={'c2c_section': 'Provenance'},
                    tooltip=(
                        "JSON object describing how R, A and B were generated (model, sampler, steps, VAE...). It is "
                        "validated and hashed: only reports with identical compatibility data share one aggregate. "
                        "The default is marked user-unverified."
                    ),
                ),
                io.Int.Input(
                    "frame_chunk_size", default=4, min=1, max=32, step=1, extra_dict={'c2c_section': 'Performance'},
                    tooltip=(
                        "Frames processed per chunk when computing the video metrics (clamped to 1-32). Larger = fewer "
                        "passes but more memory at once."
                    ),
                ),
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
                io.Image.Input(
                    "video",
                    tooltip=(
                        "Decoded IMAGE batch of this run. Reduced to a float16 CPU copy of at most 393,216 pixels per "
                        "frame and held in RAM until all three roles of the benchmark are captured."
                    ),
                ),
                io.Combo.Input(
                    "role", options=list(_ROLE_OPTIONS), default=_ROLE_OPTIONS[0],
                    tooltip=(
                        "Which result this run is: R = native H3 with Spectrum bypassed, A = legacy Spectrum, "
                        "B = candidate. Run the workflow three times with one benchmark_id, once per role, any order."
                    ),
                ),
                io.Float.Input(
                    "fps", default=24.0, min=0.01, max=240.0, step=0.01,
                    tooltip=(
                        "Frame rate of the video (frames per second), recorded in the report. Must be identical for "
                        "R, A and B of one benchmark."
                    ),
                ),
                io.String.Input(
                    "benchmark_id", default="h3-objective-seed-1",
                    tooltip=(
                        "Groups the three role captures into one R/A/B triad; must not be empty. Use a new id for a "
                        "new prompt or seed. At most 2 incomplete benchmarks are held; older ones are evicted."
                    ),
                ),
                io.Int.Input(
                    "generation_seed", force_input=True, min=0, max=0xFFFFFFFFFFFFFFFF,
                    tooltip=(
                        "Connect the same fixed INT seed that drives the generation. Must be identical for R, A and B; "
                        "recorded as the report seed."
                    ),
                ),
                io.Int.Input(
                    "steps", default=20, min=1, max=1000, step=1,
                    tooltip=(
                        "Sampler step count of the generation, recorded in the provenance. Must be identical for "
                        "R, A and B. It does not run any sampling itself."
                    ),
                ),
                io.String.Input(
                    "compatibility_tag",
                    default="minimax-h3-er-sde-current-workflow",
                    extra_dict={'c2c_section': 'Provenance'},
                    tooltip=(
                        "Short label you assert for the unchanged model/precision/scheduler/conditioning/decoder "
                        "setup. Must be non-empty and identical for R, A and B; reports only aggregate with the same tag."
                    ),
                ),
                io.Int.Input(
                    "frame_chunk_size", default=4, min=1, max=32, step=1, extra_dict={'c2c_section': 'Performance'},
                    tooltip=(
                        "Frames processed per chunk when the video metrics are computed after the third role "
                        "(clamped to 1-32). Larger = fewer passes, more memory at once. Must be identical for R, A and B."
                    ),
                ),
                io.Boolean.Input(
                    "reset_before_capture", default=False,
                    tooltip=(
                        "On: discards any incomplete capture stored under this benchmark_id before storing this role. "
                        "Needed to re-capture a role that is already stored."
                    ),
                ),
                io.Audio.Input(
                    "audio", optional=True,
                    tooltip=(
                        "Optional decoded AUDIO of this run, kept on CPU for the audio metrics. Connect it for all "
                        "three roles or none (the source layout must match)."
                    ),
                ),
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
                io.String.Input(
                    "benchmark_id", default="h3-objective-seed-1",
                    tooltip=(
                        "Benchmark whose incomplete sequential captures are released from CPU RAM when "
                        "scope=benchmark. Ignored when scope=all."
                    ),
                ),
                io.Combo.Input(
                    "scope", options=["benchmark", "all"], default="benchmark",
                    tooltip=(
                        "benchmark = release only benchmark_id's pending captures; all = release every pending "
                        "sequential capture. Written reports on disk are not touched."
                    ),
                ),
            ],
            outputs=[
                io.String.Output(
                    "summary",
                    tooltip="One line naming the benchmarks that were released, or 'nothing'. Also printed to the console.",
                )
            ],
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
