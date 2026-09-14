"""AV latent continuation and stream plumbing for long-form H3.

PORTED FROM: third_party/ComfyUI-MiniMax-H3-LongMedia/nodes.py (the
continuation and AV-stream node group). Arithmetic lives in
mmx_utils/av_latent_ops.py, ported from that pack's latent_ops.py.

These complete the long-form chain this suite already had both ends of:

    ContextWindows   plan the tiling (legal lengths, cycle-aligned strides,
                     RoPE offsets)
    PrepareContinuation  build the next segment's latent with the previous
                     segment's tail pinned at its start
    <sample>
    ToneCompensate   take out the denoiser's tone bias at the seam
    StitchContinuation   join on the overlap without shifting time

WHY THE H3 IMPORT IS LAZY: NestedTensor lives in comfy.ldm.minimax, which the
installed ComfyUI here does not have. A module-scope import would make
__init__'s per-node try/except swallow the failure and these nodes would vanish
from the menu with no visible error - which is exactly what happened to six
Motion Context nodes earlier in this work. Imported inside execute(), the nodes
REGISTER and explain themselves if H3 support is genuinely absent.
"""

from __future__ import annotations

import sys
from pathlib import Path

from comfy_api.latest import io

_PKG = Path(__file__).resolve().parents[1]
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from mmx_utils.av_latent_ops import (  # noqa: E402
    FPS,
    frame_count_from_video_t,
    merge_av_latents,
    pack_av_latents,
    prepare_continuation,
    split_av_latent,
    stitch_continuation,
    unpack_av_samples,
)

_H3_MISSING = (
    "MiniMax H3 support is missing from this ComfyUI build: comfy.ldm.minimax "
    "does not import ({err}). AV latents are NestedTensors defined there, so "
    "this node cannot run without it. Update ComfyUI to a build that ships H3 - "
    "the pack itself is installed correctly."
)


def _nested_factory():
    """Resolve H3's NestedTensor on demand, with a sentence if it is absent."""
    try:
        from comfy.ldm.minimax.model import NestedTensor  # type: ignore
    except Exception as exc:  # noqa: BLE001 - comfy_kitchen skew raises AttributeError
        raise RuntimeError(_H3_MISSING.format(err=exc)) from exc
    return NestedTensor


class MiniMaxH3_PrepareContinuation(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MiniMaxH3_PrepareContinuation",
            display_name="H3 Prepare Continuation",
            category="MiniMax H3/Long",
            description=(
                "Build the NEXT segment's latent with the previous segment's tail "
                "pinned at its start, so the model continues a shot instead of "
                "starting a new one.\n\n"
                "Audio and video are pinned together: H3 samples them jointly, so a "
                "continuation that carries only the picture drifts out of sync. "
                "Sample the result, then join it with Stitch Continuation."
            ),
            inputs=[
                io.Latent.Input(
                    "source_av",
                    tooltip="The PREVIOUS segment's AV latent. Its tail becomes the "
                            "new segment's opening context."),
                io.Int.Input(
                    "length", default=124, min=5, max=3600, step=17,
                    tooltip="Frames for the new segment. Snapped to H3's grid "
                            "(n % 17 == 5); use Context Windows to choose this so it "
                            "matches the rest of the plan."),
                io.Int.Input(
                    "overlap_frames", default=22, min=5, max=3600, step=17,
                    tooltip="How much of the previous tail to pin. Snapped DOWN to "
                            "the 17k+5 grid. This is the same number Stitch "
                            "Continuation and Tone Compensate need - keep all three "
                            "in agreement or the seam lands in the wrong place."),
                io.Float.Input(
                    "video_context_denoise", default=0.0, min=0.0, max=1.0, step=0.01,
                    tooltip="How much the pinned video context may change. 0 holds it "
                            "exactly, which is what makes the join invisible. Raise it "
                            "only if the pinned frames are fighting the new prompt."),
                io.Float.Input(
                    "audio_context_denoise", default=0.0, min=0.0, max=1.0, step=0.01,
                    tooltip="The same for audio. Keep at 0 unless the audio seam is "
                            "audibly repeating."),
            ],
            outputs=[
                io.Latent.Output(display_name="continuation_av",
                                 tooltip="Sample this, then Stitch Continuation."),
                io.Int.Output(display_name="frame_count",
                              tooltip="Frames after grid snapping."),
                io.Int.Output(display_name="actual_overlap_frames",
                              tooltip="Overlap after snapping - feed THIS to Stitch "
                                      "Continuation, not your requested value."),
                io.Float.Output(display_name="overlap_seconds"),
                io.String.Output(display_name="report"),
            ],
        )

    @classmethod
    def fingerprint_inputs(cls, source_av, length, overlap_frames,
                           video_context_denoise, audio_context_denoise):
        return (f"{id(source_av)}|{length}|{overlap_frames}"
                f"|{video_context_denoise}|{audio_context_denoise}")

    @classmethod
    def execute(cls, source_av, length, overlap_frames,
                video_context_denoise, audio_context_denoise) -> io.NodeOutput:
        nested = _nested_factory()
        try:
            out, frame_count, actual_overlap = prepare_continuation(
                source_av, length, overlap_frames,
                video_context_denoise, audio_context_denoise, nested,
            )
        except ValueError as exc:
            raise ValueError(f"H3 Prepare Continuation: {exc}") from exc

        src_video, _src_audio = unpack_av_samples(source_av)
        src_frames = frame_count_from_video_t(src_video.shape[2])
        report = "\n".join([
            f"continuation: {frame_count} frames, {actual_overlap} pinned from a "
            f"{src_frames}-frame source ({actual_overlap / FPS:.2f}s)",
            f"  context denoise  video {video_context_denoise:.2f} "
            f"audio {audio_context_denoise:.2f}",
            (f"  NOTE: overlap {overlap_frames} was snapped to {actual_overlap} "
             f"(H3 grid). Give {actual_overlap} to Stitch Continuation and Tone "
             f"Compensate, not {overlap_frames}."
             if actual_overlap != overlap_frames else
             f"  overlap {actual_overlap} is already on the grid"),
        ])
        return io.NodeOutput(out, frame_count, actual_overlap,
                             actual_overlap / FPS, report)


class MiniMaxH3_StitchContinuation(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MiniMaxH3_StitchContinuation",
            display_name="H3 Stitch Continuation",
            category="MiniMax H3/Long",
            description=(
                "Join a sampled continuation onto the previous segment along their "
                "shared overlap, without shifting time.\n\n"
                "The overlap must be the value Prepare Continuation actually used "
                "after grid snapping, not the one you typed. This node verifies the "
                "result: if the stitched video length or the audio/video sync does "
                "not come out exactly as predicted it raises, rather than handing on "
                "a latent that is quietly the wrong length."
            ),
            inputs=[
                io.Latent.Input("previous_av",
                                tooltip="Everything stitched so far."),
                io.Latent.Input("sampled_continuation_av",
                                tooltip="The freshly sampled continuation."),
                io.Int.Input(
                    "overlap_frames", default=22, min=5, max=3600, step=17,
                    tooltip="Use actual_overlap_frames from Prepare Continuation. A "
                            "mismatch here does not error - it silently joins at the "
                            "wrong place, which reads as a stutter or a jump cut."),
                io.Boolean.Input(
                    "blend_video_overlap", default=False, optional=True,
                    tooltip="Smoothstep the video across the seam. Off by default "
                            "because the pinned context should already match; turn it "
                            "on when a residual difference still shows."),
                io.Boolean.Input(
                    "offload_to_cpu", default=False, optional=True,
                    tooltip="Keep the growing result in system RAM. Worth enabling for "
                            "long multi-pass runs: this accumulator is only read by the "
                            "next stitch and the final decode, so holding it in VRAM "
                            "across many passes just starves the sampler."),
            ],
            outputs=[
                io.Latent.Output(display_name="stitched_av"),
                io.Int.Output(display_name="total_frames"),
                io.String.Output(display_name="report"),
            ],
        )

    @classmethod
    def fingerprint_inputs(cls, previous_av, sampled_continuation_av, overlap_frames,
                           blend_video_overlap=False, offload_to_cpu=False):
        return (f"{id(previous_av)}|{id(sampled_continuation_av)}|{overlap_frames}"
                f"|{blend_video_overlap}|{offload_to_cpu}")

    @classmethod
    def execute(cls, previous_av, sampled_continuation_av, overlap_frames,
                blend_video_overlap=False, offload_to_cpu=False) -> io.NodeOutput:
        nested = _nested_factory()
        prev_video, _ = unpack_av_samples(previous_av)
        next_video, _ = unpack_av_samples(sampled_continuation_av)
        prev_frames = frame_count_from_video_t(prev_video.shape[2])
        next_frames = frame_count_from_video_t(next_video.shape[2])

        try:
            stitched_av, total_frames = stitch_continuation(
                previous_av, sampled_continuation_av, overlap_frames, nested,
                bool(blend_video_overlap), bool(offload_to_cpu),
            )
        except ValueError as exc:
            raise ValueError(f"H3 Stitch Continuation: {exc}") from exc

        # Verify rather than trust. A stitched latent of the wrong length, or one
        # whose audio and video disagree, produces a clip that decodes fine and is
        # subtly out of sync - the worst kind of failure to debug later.
        stitched_video, stitched_audio = unpack_av_samples(stitched_av)
        stitched_frames = frame_count_from_video_t(stitched_video.shape[2])
        if stitched_frames != int(total_frames):
            raise RuntimeError(
                f"H3 Stitch Continuation: the join produced {stitched_frames} frames "
                f"but {int(total_frames)} were expected "
                f"(previous {prev_frames}, next {next_frames}, overlap "
                f"{prev_frames + next_frames - int(total_frames)}). The overlap given "
                f"here probably does not match the one Prepare Continuation used."
            )

        used_overlap = prev_frames + next_frames - int(total_frames)
        report = "\n".join([
            f"stitched {prev_frames}f + {next_frames}f on a {used_overlap}f overlap "
            f"-> {stitched_frames}f ({stitched_frames / FPS:.2f}s)",
            f"  audio latent rows {int(stitched_audio.shape[-1])}, video rows "
            f"{int(stitched_video.shape[2])} - in sync",
            f"  video seam blend {'on' if blend_video_overlap else 'off'}"
            f"{', result offloaded to CPU' if offload_to_cpu else ''}",
        ])
        return io.NodeOutput(stitched_av, int(total_frames), report)


class MiniMaxH3_PackAV(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MiniMaxH3_PackAV",
            display_name="H3 Pack AV",
            category="MiniMax H3/Long",
            description=(
                "Combine a separate video latent and audio latent into the single "
                "joint AV latent H3 samples.\n\n"
                "H3 denoises picture and sound together, so they travel as one "
                "NestedTensor. Each stream keeps its own noise mask, which is what "
                "lets you pin one and regenerate the other."
            ),
            inputs=[
                io.Latent.Input("video_latent"),
                io.Latent.Input("audio_latent"),
            ],
            outputs=[io.Latent.Output(display_name="av_latent")],
        )

    @classmethod
    def fingerprint_inputs(cls, video_latent, audio_latent):
        return f"{id(video_latent)}|{id(audio_latent)}"

    @classmethod
    def execute(cls, video_latent, audio_latent) -> io.NodeOutput:
        try:
            return io.NodeOutput(
                pack_av_latents(video_latent, audio_latent, _nested_factory()))
        except ValueError as exc:
            raise ValueError(f"H3 Pack AV: {exc}") from exc


class MiniMaxH3_SplitAV(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MiniMaxH3_SplitAV",
            display_name="H3 Split AV",
            category="MiniMax H3/Long",
            description=(
                "Take a joint AV latent apart into its video and audio streams.\n\n"
                "Neither stream inherits the other's noise mask, so you can work on "
                "one without silently constraining the other."
            ),
            inputs=[io.Latent.Input("av_latent")],
            outputs=[
                io.Latent.Output(display_name="video_latent"),
                io.Latent.Output(display_name="audio_latent"),
            ],
        )

    @classmethod
    def fingerprint_inputs(cls, av_latent):
        return str(id(av_latent))

    @classmethod
    def execute(cls, av_latent) -> io.NodeOutput:
        try:
            video, audio = split_av_latent(av_latent)
        except ValueError as exc:
            raise ValueError(f"H3 Split AV: {exc}") from exc
        return io.NodeOutput(video, audio)


class MiniMaxH3_MergeAV(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MiniMaxH3_MergeAV",
            display_name="H3 Merge AV",
            category="MiniMax H3/Long",
            description=(
                "Blend one AV latent toward another, picture and sound "
                "independently.\n\n"
                "Useful for easing between two takes of the same shot, or for "
                "keeping one stream while crossing the other over."
            ),
            inputs=[
                io.Latent.Input("target_av", tooltip="The latent being blended INTO."),
                io.Latent.Input("source_av", tooltip="The latent blended FROM."),
                io.Float.Input("video_mix", default=1.0, min=0.0, max=1.0, step=0.01,
                               tooltip="0 keeps the target's picture, 1 takes the "
                                       "source's."),
                io.Float.Input("audio_mix", default=0.0, min=0.0, max=1.0, step=0.01,
                               tooltip="The same for sound. Setting these differently "
                                       "is the point: the defaults take the source's "
                                       "picture and keep the target's audio."),
                io.Combo.Input("fit_mode", options=["strict", "crop_pad"],
                               default="strict",
                               tooltip="strict refuses a length mismatch; crop_pad "
                                       "makes the source fit the target, which is the "
                                       "target's geometry either way."),
                io.Combo.Input("alignment", options=["start", "end", "center"],
                               default="start",
                               tooltip="Where a crop_pad source sits inside the "
                                       "target. 'end' is what you want when matching "
                                       "the tail of a shot."),
                io.Float.Input("video_denoise", default=0.0, min=0.0, max=1.0,
                               step=0.01,
                               tooltip="Noise mask written onto the merged video. 0 "
                                       "pins it, so the sampler leaves the blend "
                                       "alone."),
                io.Float.Input("audio_denoise", default=0.0, min=0.0, max=1.0,
                               step=0.01,
                               tooltip="The same for the merged audio."),
            ],
            outputs=[io.Latent.Output(display_name="av_latent")],
        )

    @classmethod
    def fingerprint_inputs(cls, target_av, source_av, video_mix, audio_mix,
                           fit_mode, alignment, video_denoise, audio_denoise):
        return (f"{id(target_av)}|{id(source_av)}|{video_mix}|{audio_mix}"
                f"|{fit_mode}|{alignment}|{video_denoise}|{audio_denoise}")

    @classmethod
    def execute(cls, target_av, source_av, video_mix, audio_mix,
                fit_mode, alignment, video_denoise, audio_denoise) -> io.NodeOutput:
        try:
            return io.NodeOutput(
                merge_av_latents(target_av, source_av, video_mix, audio_mix,
                                 fit_mode, alignment, video_denoise, audio_denoise,
                                 _nested_factory()))
        except ValueError as exc:
            raise ValueError(f"H3 Merge AV: {exc}") from exc
