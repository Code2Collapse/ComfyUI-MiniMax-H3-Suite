"""H3 Tail From Latent — the end of one clip, ready to start the next.

Ported from ComfyUI_MiniMax_H3_Extender (Apache-2.0), which uses it inside its
own multi-clip director. Standalone it is the manual way to chain clips: take
the last half-second of picture and the matching half-second of sound, feed
both back as references, and the next generation continues rather than
restarting.

The fix over a hand-built version is that the audio length is derived from the
FINAL frame count, after grid alignment - not from what was asked for. Round
the two independently and picture drifts against sound at every join.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from comfy_api.latest import io

_PKG = Path(__file__).resolve().parents[1]
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from mmx_utils.tail_extract import (
    FPS,
    TailExtractError,
    describe,
    normalise_audio,
    slice_audio_tail,
    tail_frame_count,
)


def _video_stream(samples):
    """The video half of an H3 AV latent."""
    latent = samples["samples"] if isinstance(samples, dict) else samples
    if getattr(latent, "is_nested", False):
        parts = latent.unbind()
        if not parts:
            raise TailExtractError(
                "This latent is a nested tensor with no streams in it. It did "
                "not come from an H3 sampler.")
        latent = parts[0]
    return latent


def _audio_stream(samples):
    """The audio half, which is the LAST stream in the nested latent."""
    latent = samples["samples"] if isinstance(samples, dict) else samples
    if not getattr(latent, "is_nested", False):
        return None
    parts = latent.unbind()
    return parts[-1] if len(parts) > 1 else None


class MiniMaxH3_TailFromLatent(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MiniMaxH3_TailFromLatent",
            display_name="H3 Tail From Latent",
            category="MiniMax H3/Continuation",
            description=(
                "Decode an H3 AV latent and return its TAIL - the last N "
                "seconds of picture with the matching audio - so the next clip "
                "can continue from it instead of restarting. The audio length "
                "follows the final frame count, so picture and sound cannot "
                "drift apart at the join."
            ),
            inputs=[
                io.Latent.Input("samples", tooltip="The sampler's AV latent."),
                io.Vae.Input("vae"),
                io.Vae.Input(
                    "audio_vae", optional=True,
                    tooltip="Wire this to get the tail's audio as well. "
                            "Without it the audio output is silent, and a "
                            "reference video with no soundtrack loses the "
                            "voice continuity across the join."),
                io.Float.Input(
                    "tail_seconds", default=0.5, min=0.1, max=15.0, step=0.05,
                    tooltip="How much of the end to take. Around half a second "
                            "is enough motion to continue from; much longer "
                            "and the reference starts dictating the next "
                            "clip's whole shot."),
                io.Boolean.Input(
                    "align_to_h3_grid", default=True,
                    tooltip="Round the frame count up to H3's 17k+5 grid. Off "
                            "the grid a reference video is reinterpreted, not "
                            "refused, and its motion arrives at the wrong "
                            "speed. Only turn this off if something downstream "
                            "wants an exact length."),
            ],
            outputs=[
                io.Image.Output(
                    display_name="ref_video",
                    tooltip="The tail frames. Wire to a reference video input."),
                io.Audio.Output(
                    display_name="ref_video_audio",
                    tooltip="The SAME span of audio. Wire to the matching "
                            "reference-video audio input, not a standalone "
                            "audio reference."),
                io.Image.Output(
                    display_name="last_frame",
                    tooltip="The final frame alone, for use as a first-frame "
                            "keyframe on the next clip."),
                io.Int.Output(display_name="frame_count"),
                io.Float.Output(display_name="duration_seconds"),
                io.String.Output(display_name="report"),
            ],
        )

    @classmethod
    def execute(cls, samples, vae, audio_vae=None, tail_seconds=0.5,
                align_to_h3_grid=True) -> io.NodeOutput:
        frames = vae.decode(_video_stream(samples))
        if frames.ndim == 5:
            frames = frames.reshape(-1, *frames.shape[-3:])

        total = int(frames.shape[0])
        count = tail_frame_count(total, tail_seconds, align=bool(align_to_h3_grid))
        duration = count / FPS

        if audio_vae is not None:
            audio_latent = _audio_stream(samples)
            if audio_latent is None:
                raise TailExtractError(
                    "An audio VAE is connected but this latent has no audio "
                    "stream. Either it came from a video-only path, or the AV "
                    "latent was split upstream - reconnect the sampler's own "
                    "output.")
            waveform = normalise_audio(audio_vae.decode(audio_latent).movedim(-1, 1))
            sample_rate = int(
                samples.get("sample_rate")
                if isinstance(samples, dict) and "sample_rate" in samples
                else getattr(audio_vae, "audio_sample_rate_output",
                             getattr(audio_vae, "audio_sample_rate", 32000)))
            audio = slice_audio_tail(
                {"waveform": waveform, "sample_rate": sample_rate}, duration)
            audio_note = ""
        else:
            sample_rate = 32000
            audio = {"waveform": torch.zeros(
                1, 2, max(1, int(round(duration * sample_rate)))),
                "sample_rate": sample_rate}
            audio_note = ("\nNo audio VAE is connected, so the audio output is "
                          "SILENCE of the right length. Using it as a reference "
                          "soundtrack would tell the next clip the join is "
                          "silent.")

        report = describe(total, count, tail_seconds,
                          aligned=bool(align_to_h3_grid)) + audio_note
        return io.NodeOutput(frames[-count:], audio, frames[-1:].clone(),
                             int(count), float(duration), report)
