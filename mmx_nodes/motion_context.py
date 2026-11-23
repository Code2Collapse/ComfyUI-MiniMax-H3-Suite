# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) NikoDemon80 — ComfyUI-H3-Motion-Context
# PORTED FROM: ComfyUI-H3-Motion-Context :: nodes.py, probe_node.py @ third_party

"""H3 motion-context chaining nodes (V3 API).

Ported from ComfyUI-H3-Motion-Context by NikoDemon80 (GPL-3.0).
"""

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

from comfy_api.latest import io

_PKG = Path(__file__).resolve().parents[1]
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from mmx_utils.motion_context import (
    apply_motion_context,
    load_context_latent,
    register_chain_routes,
    resolve_latent_path,
    run_seam_probe,
    save_context_latent,
    trim_clip,
)

# Chain UI routes are intentionally not registered (no JavaScript in this pack).
register_chain_routes()

_CATEGORY = "MiniMax H3/Motion"
_ATTRIBUTION = (
    "Ported from ComfyUI-H3-Motion-Context (NikoDemon80, GPL-3.0)."
)


class MiniMaxH3_MotionContext(io.ComfyNode):
    """Pin previous-clip motion at the head of an H3 clip."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MiniMaxH3_MotionContext",
            display_name="H3 Motion Context",
            category=_CATEGORY,
            description=(
                "Pin a run of consecutive frames from a previous clip as "
                "never-denoised conditioning rows. With context_latent wired, "
                "both picture and sound are sliced from the previous clip's "
                "latent. " + _ATTRIBUTION
            ),
            inputs=[
                io.Conditioning.Input(
                    "conditioning",
                    tooltip="Positive H3 conditioning of this clip. The pinned frames are added as keyframes at "
                            "frames 0..n-1; existing keyframes inside that head are dropped.",
                ),
                io.Vae.Input(
                    "vae",
                    tooltip="H3 video VAE: encodes context_frames (used only when no context_latent is wired).",
                ),
                io.Latent.Input(
                    "latent",
                    tooltip="This clip's (empty) H3 latent. Only its shape is read: width / height and the "
                            "pixel frame count; it is not changed or output.",
                ),
                io.Combo.Input(
                    "context_length",
                    options=["22", "5", "39", "56"],
                    default="22",
                    tooltip="Frames of the previous clip's picture to carry over.",
                ),
                io.Int.Input(
                    "audio_context_length",
                    default=24,
                    min=0,
                    max=240,
                    tooltip="Frames of tail audio to pin (0 follows video window).",
                    extra_dict={"c2c_section": "Audio"},
                ),
                io.Image.Input(
                    "context_frames",
                    optional=True,
                    tooltip="Decoded frames of the previous clip when no context_latent.",
                ),
                io.Latent.Input(
                    "context_latent",
                    optional=True,
                    tooltip="Previous clip's sampler output latent.",
                ),
                io.Vae.Input(
                    "audio_vae",
                    optional=True,
                    tooltip="H3 audio VAE for context_audio.",
                ),
                io.Audio.Input(
                    "context_audio",
                    optional=True,
                    tooltip="Audio of the previous clip.",
                ),
            ],
            outputs=[
                io.Conditioning.Output(
                    "conditioning",
                    tooltip="The conditioning with the previous clip's tail pinned as never-denoised keyframe "
                            "rows at the head (plus the tail audio on the timeline). Unchanged when no context "
                            "is wired.",
                ),
                io.Int.Output(
                    "trim_frames",
                    tooltip="Pixel frames pinned at the head (0 when no context). Wire to Motion Context Trim "
                            "to drop them after decoding.",
                ),
                io.String.Output(
                    "report",
                    tooltip="Text: video source, frames pinned, keyframe indices, clip size, trim and audio "
                            "placement.",
                ),
            ],
        )

    @classmethod
    def fingerprint_inputs(
        cls,
        conditioning,
        vae,
        latent,
        context_length="22",
        audio_context_length=24,
        context_frames=None,
        context_latent=None,
        audio_vae=None,
        context_audio=None,
    ):
        parts = [
            str(context_length),
            str(audio_context_length),
            str(context_frames is not None),
            str(context_latent is not None),
            str(context_audio is not None),
        ]
        if context_frames is not None:
            parts.append(hashlib.md5(context_frames.cpu().numpy().tobytes()).hexdigest())
        if context_latent is not None:
            samples = context_latent.get("samples")
            if hasattr(samples, "tensors"):
                parts.append(str(tuple(t.shape for t in samples.tensors)))
            elif isinstance(samples, (list, tuple)):
                parts.append(str(tuple(getattr(s, "shape", ()) for s in samples)))
        return ":".join(parts)

    @classmethod
    def execute(
        cls,
        conditioning,
        vae,
        latent,
        context_length="22",
        audio_context_length=24,
        context_frames=None,
        context_latent=None,
        audio_vae=None,
        context_audio=None,
    ) -> io.NodeOutput:
        out, trim, report = apply_motion_context(
            conditioning,
            vae,
            latent,
            int(context_length),
            audio_context_length=audio_context_length,
            context_frames=context_frames,
            context_latent=context_latent,
            audio_vae=audio_vae,
            context_audio=context_audio,
        )
        return io.NodeOutput(out, trim, report)


class MiniMaxH3_MotionContextTrim(io.ComfyNode):
    """Drop the pinned head off a decoded clip, picture and sound together."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MiniMaxH3_MotionContextTrim",
            display_name="H3 Motion Context Trim",
            category=_CATEGORY,
            description=(
                "Remove the leading pinned frames from a decoded H3 clip, "
                "trimming picture and sound by the same duration. "
                + _ATTRIBUTION
            ),
            inputs=[
                io.Image.Input(
                    "images",
                    tooltip="Decoded frames of the clip, pinned head included.",
                ),
                io.Int.Input(
                    "trim_frames",
                    default=0,
                    min=0,
                    max=4096,
                    tooltip="Leading frames to remove (Motion Context trim_frames). Audio loses trim_frames / "
                            "fps seconds. 0 = nothing removed; must be less than the frame count.",
                ),
                io.Audio.Input(
                    "audio",
                    optional=True,
                    tooltip="Decoded audio for the same clip.",
                ),
                io.Float.Input(
                    "fps",
                    default=24.0,
                    min=1.0,
                    max=240.0,
                    step=0.001,
                    tooltip="Frame rate used to convert trim into audio duration.", extra_dict={"c2c_section": "Audio"},
                ),
                io.Boolean.Input(
                    "match_tail",
                    default=True,
                    tooltip="Trim or pad audio tail to match frames/fps exactly.", extra_dict={"c2c_section": "Audio"},
                ),
            ],
            outputs=[
                io.Image.Output(
                    "images",
                    tooltip="The frames after the first trim_frames.",
                ),
                io.Audio.Output(
                    "audio",
                    tooltip="Audio with the same duration cut from the start (and, with match_tail, cut / "
                            "zero-padded to frames / fps). None when no audio is wired.",
                ),
                io.String.Output(
                    "report",
                    tooltip="Text: frames trimmed, tail adjustment and picture / sound drift in ms.",
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        images,
        trim_frames=0,
        audio=None,
        fps=24.0,
        match_tail=True,
    ) -> io.NodeOutput:
        out_images, out_audio, report = trim_clip(
            images, trim_frames, audio=audio, fps=fps, match_tail=match_tail)
        return io.NodeOutput(out_images, out_audio, report)


class MiniMaxH3_MotionContextSaveLatent(io.ComfyNode):
    """Save an H3 AV latent to disk for the next run."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MiniMaxH3_MotionContextSaveLatent",
            display_name="H3 Motion Context Save Latent",
            category=_CATEGORY,
            description=(
                "Save the sampler's AV latent so the next run's Motion Context "
                "node can pin audio from it via the matching Load node. "
                + _ATTRIBUTION
            ),
            inputs=[
                io.Latent.Input(
                    "latent",
                    tooltip="The sampler's output latent.",
                ),
                io.String.Input(
                    "filename_prefix",
                    default="h3_context/clip",
                    tooltip="Saved under the ComfyUI output folder.",
                ),
                io.Int.Input(
                    "clip_index",
                    default=1,
                    min=0,
                    max=9999,
                    step=1,
                    tooltip="Fixed slot for this clip (0 = auto-numbered runs).",
                ),
            ],
            outputs=[
                io.String.Output(
                    "latent_path",
                    tooltip="Full path of the written .safetensors (video + audio latents): "
                            "<prefix>_<clip_index:05d>.safetensors, or an auto-numbered name when clip_index "
                            "is 0.",
                ),
                io.String.Output(
                    "report",
                    tooltip="Text: saved path and the video / audio latent shapes.",
                ),
            ],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, latent, filename_prefix, clip_index=1) -> io.NodeOutput:
        path, report = save_context_latent(latent, filename_prefix, clip_index)
        return io.NodeOutput(path, report)


class MiniMaxH3_MotionContextLoadLatent(io.ComfyNode):
    """Load a saved H3 AV latent for the context_latent input."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MiniMaxH3_MotionContextLoadLatent",
            display_name="H3 Motion Context Load Latent",
            category=_CATEGORY,
            description=(
                "Load a latent saved by H3 Motion Context Save Latent, "
                "for the context_latent input only. Not decodable. "
                + _ATTRIBUTION
            ),
            inputs=[
                io.String.Input(
                    "latent_path",
                    default="h3_context",
                    tooltip="Saved latent file or folder.",
                ),
                io.Int.Input(
                    "clip_index",
                    default=0,
                    min=0,
                    max=9999,
                    step=1,
                    tooltip="Clip to continue FROM (0 = first clip, nothing loaded).",
                ),
            ],
            outputs=[
                io.Latent.Output(
                    "latent",
                    tooltip="The saved AV latent ({'samples': [video, audio]}) for Motion Context "
                            "context_latent only - not decodable. None when clip_index is 0.",
                ),
                io.String.Output(
                    "report",
                    tooltip="Text: loaded path and latent shapes, or that clip_index 0 loads nothing.",
                ),
            ],
            not_idempotent=True,
        )

    @classmethod
    def fingerprint_inputs(cls, latent_path, clip_index=0):
        if int(clip_index) <= 0:
            return "disabled"
        try:
            p = resolve_latent_path(latent_path, clip_index)
            return "%s:%d:%d" % (p, int(clip_index), os.stat(p).st_mtime_ns)
        except Exception:
            return "unresolved:%s:%d" % (latent_path, int(clip_index))

    @classmethod
    def execute(cls, latent_path, clip_index=0) -> io.NodeOutput:
        latent, report = load_context_latent(latent_path, clip_index)
        return io.NodeOutput(latent, report)


class MiniMaxH3_MotionContextChain(io.ComfyNode):
    """DEPRECATED (A16): replaced by H3 Continuation Next + Commit.

    It never chained anything (the upstream pack's JavaScript buttons were never shipped). Kept registered, hidden from
    the node search, so saved workflows still load; running it says what to use instead.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MiniMaxH3_MotionContextChain",
            display_name="H3 Motion Context Chain",
            category=_CATEGORY,
            description=(
                "Deprecated: replaced by H3 Continuation Next + Commit (long video in segments, resumable, state "
                "on disk). This node does nothing; it is kept so saved workflows load. " + _ATTRIBUTION
            ),
            is_deprecated=True,
            inputs=[
                io.Int.Input(
                    "segments",
                    default=0,
                    min=0,
                    max=9999,
                    step=1,
                    tooltip="Unused: kept so saved workflows load (this node is deprecated).",
                    extra_dict={"c2c_hidden": True},
                ),
            ],
            outputs=[
                io.String.Output("report", tooltip="Plain-text deprecation note pointing to H3 Continuation Next + "
                                                   "Commit; the node does no other work."),
            ],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, segments=0) -> io.NodeOutput:
        report = (
            "H3 Motion Context Chain is deprecated and does nothing (segments=%d). For long video use H3 "
            "Continuation Next (before the sampler) and H3 Continuation Commit (after the decoders): each queue "
            "generates the next segment, keeps the session on disk and resumes after a crash." % int(segments))
        return io.NodeOutput(report)


class MiniMaxH3_MotionContextSeamProbe(io.ComfyNode):
    """Measure a chain join: continuation quality and level step."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MiniMaxH3_MotionContextSeamProbe",
            display_name="H3 Motion Context Seam Probe",
            category=_CATEGORY,
            description=(
                "Measure a chain join in the graph: does clip B continue clip A's "
                "audio, and does the level step at the cut. " + _ATTRIBUTION
            ),
            inputs=[
                io.Audio.Input(
                    "clip_b_untrimmed",
                    tooltip="This clip's audio before the trim node.",
                ),
                io.Int.Input(
                    "trim_frames",
                    default=0,
                    min=0,
                    max=4096,
                    tooltip="From Motion Context trim_frames output.",
                ),
                io.Latent.Input(
                    "clip_a_latent",
                    optional=True,
                    tooltip="Previous clip's AV latent (context_latent).",
                ),
                io.Vae.Input(
                    "audio_vae",
                    optional=True,
                    tooltip="H3 audio VAE to decode clip A.",
                ),
                io.Float.Input(
                    "fps",
                    default=24.0,
                    min=1.0,
                    max=240.0,
                    step=0.001,
                    tooltip="Frame rate that converts trim_frames (and clip A's frame count) into seconds / "
                            "samples. H3 clips are 24 fps.",
                ),
                io.Float.Input(
                    "window_ms",
                    default=50.0,
                    min=5.0,
                    max=500.0,
                    step=1.0, extra_dict={"c2c_section": "Probe"},
                    tooltip="Length in ms of each clip B window cross-correlated against clip A's tail "
                            "(windows step every 25 ms over the pinned span).",
                ),
                io.Float.Input(
                    "search_ms",
                    default=40.0,
                    min=5.0,
                    max=500.0,
                    step=1.0, extra_dict={"c2c_section": "Probe"},
                    tooltip="How far in ms either side of the expected position the lag is searched; a lag "
                            "pegged at this limit is flagged in the report.",
                ),
            ],
            outputs=[
                io.Audio.Output(
                    "audio",
                    tooltip="clip_b_untrimmed passed through unchanged.",
                ),
                io.String.Output(
                    "report",
                    tooltip="Text: continuation over the pinned span (normalized cross-correlation, lag in ms) "
                            "and the level / noise-floor step across the cut, with a reading. Measurement "
                            "only.",
                ),
            ],
            is_output_node=True,
        )

    @classmethod
    def execute(
        cls,
        clip_b_untrimmed,
        trim_frames=0,
        clip_a_latent=None,
        audio_vae=None,
        fps=24.0,
        window_ms=50.0,
        search_ms=40.0,
    ) -> io.NodeOutput:
        passthrough, report = run_seam_probe(
            clip_b_untrimmed,
            trim_frames,
            clip_a_latent=clip_a_latent,
            audio_vae=audio_vae,
            fps=fps,
            window_ms=window_ms,
            search_ms=search_ms,
        )
        return io.NodeOutput(
            passthrough,
            report,
            ui={"text": [report]},
        )
