"""H3 Text Encoder Unload — free Qwen3-VL-32B before sampling.

PORTED FROM: comfyui-deno-custom-nodes (GPL-3.0). See CREDITS.md and
mmx_utils/encoder_unload.py for why this is worth a node of its own on H3.
"""

from __future__ import annotations

import sys
from pathlib import Path

from comfy_api.latest import io

_PKG = Path(__file__).resolve().parents[1]
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from mmx_utils.encoder_unload import (
    check_can_leave_accelerator,
    describe,
    unload_clip,
)

try:
    from comfy_execution.graph_utils import ExecutionBlocker
except Exception:  # noqa: BLE001 - moved between modules across versions
    try:
        from comfy_execution.graph import ExecutionBlocker
    except Exception:  # noqa: BLE001
        ExecutionBlocker = None


class MiniMaxH3_TextEncoderUnload(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MiniMaxH3_TextEncoderUnload",
            display_name="H3 Text Encoder Unload",
            category="MiniMax H3/Sampling",
            description=(
                "Free the text encoder from VRAM after conditioning and before "
                "sampling. H3's encoder is Qwen3-VL-32B and is used once, at "
                "the start - then it sits in memory for the whole sample. On a "
                "card that is tight this is the difference between a clip "
                "rendering and an out-of-memory. Only the encoder wired in "
                "here is freed: the diffusion model, VAEs and ControlNets are "
                "untouched, which a general 'free memory' cannot promise."
            ),
            inputs=[
                io.Conditioning.Input(
                    "positive",
                    tooltip="Positive conditioning, passed through unchanged. "
                            "This node sits ON the conditioning wire because "
                            "that is the only way to say 'after the encode, "
                            "before the sampler' in a dataflow graph."),
                io.Conditioning.Input(
                    "negative", optional=True,
                    tooltip="Negative conditioning, if you have one. Connect "
                            "an encoded negative or a Conditioning Zero Out. "
                            "Whatever is connected must finish before the "
                            "encoder is freed, which is what wiring it here "
                            "guarantees. NOTE: H3's own reference pipeline "
                            "uses BasicGuider and has no negative at all."),
                io.Clip.Input(
                    "clip",
                    tooltip="The exact encoder to free. Connect the SAME CLIP "
                            "output the text-encode nodes used - a different "
                            "one frees a model nothing was waiting on."),
            ],
            outputs=[
                io.Conditioning.Output(
                    display_name="positive",
                    tooltip="The positive conditioning, unchanged, emitted "
                            "after the encoder has been freed."),
                io.Conditioning.Output(
                    display_name="negative",
                    tooltip="The negative conditioning, unchanged. Leave this "
                            "output unconnected if you did not connect a "
                            "negative input."),
                io.String.Output(display_name="report"),
            ],
        )

    @classmethod
    def fingerprint_inputs(cls, **_kwargs):
        # NaN on purpose, and the one place this pack allows it. The unload is
        # a SIDE EFFECT, not a value: a cache hit would skip it, so a second
        # run - or a run after another branch reloaded the encoder - would
        # free nothing while still reporting success.
        return float("nan")

    @classmethod
    def IS_CHANGED(cls, **_kwargs):
        return float("nan")

    @classmethod
    def execute(cls, positive, clip, negative=None) -> io.NodeOutput:
        check_can_leave_accelerator(clip)
        unload_clip(clip, label="H3 text encoder")

        if negative is None:
            # Not None and not an empty conditioning: either would be accepted
            # by a sampler and silently generate against nothing. A blocker
            # stops the branch with a message instead.
            if ExecutionBlocker is not None:
                negative_out = ExecutionBlocker(
                    "The negative output of H3 Text Encoder Unload is "
                    "connected, but nothing is connected to its negative "
                    "INPUT. Wire a negative conditioning in, or disconnect "
                    "this output - H3's own reference pipeline uses "
                    "BasicGuider and has no negative conditioning at all.")
            else:
                raise RuntimeError(
                    "The negative output is connected but the negative input "
                    "is not. This ComfyUI build is too old to block just that "
                    "branch, so the whole run is stopped instead.")
        else:
            negative_out = negative

        return io.NodeOutput(positive, negative_out, describe(clip))
