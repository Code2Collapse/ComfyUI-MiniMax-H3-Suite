"""N7 — H3 Frame Handles (editorial padding + grid alignment)."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import torch

from comfy_api.latest import io

_PKG = Path(__file__).resolve().parents[1]
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from mmx_utils.frame_handles import (
    PADDING_MODES,
    apply_frame_padding,
    format_handle_report,
    plan_frame_handles,
    sync_audio_to_frame_change,
)
from mmx_utils.h3_constants import FPS


class MiniMaxH3_FrameHandles(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MiniMaxH3_FrameHandles",
            display_name="H3 Frame Handles",
            category="MiniMax H3/Spine",
            description=(
                "Add or trim editorial handles and land frame count on a model grid.\n"
                "padding_mode: disabled | H3 (17n+5) | WAN (4n+1) | LTX2 (8n+1).\n"
                "Audio is re-synced when frame count changes (silence padded at head)."
            ),
            inputs=[
                io.Image.Input("images"),
                io.Int.Input(
                    "handle_frames",
                    default=0,
                    min=-400,
                    max=400,
                    tooltip="Frames to add (positive) or trim (negative). 0 + grid mode = auto-pad.",
                ),
                io.Combo.Input("padding_mode", options=list(PADDING_MODES), default="H3 (17n+5)"),
                io.Combo.Input("handle_side", options=["head", "tail"], default="head"),
                io.Float.Input("fps", default=float(FPS), min=1.0, max=120.0, step=0.01),
                io.Audio.Input("audio", optional=True),
            ],
            outputs=[
                io.Image.Output("images"),
                io.Audio.Output("audio"),
                io.Int.Output("frame_count"),
                io.String.Output("report"),
            ],
        )

    @classmethod
    def fingerprint_inputs(
        cls,
        images,
        handle_frames=0,
        padding_mode="H3 (17n+5)",
        handle_side="head",
        fps=float(FPS),
        audio=None,
    ):
        base = hashlib.md5(images.cpu().numpy().tobytes()).hexdigest()
        if audio is not None:
            base += ":" + hashlib.md5(audio["waveform"].cpu().numpy().tobytes()).hexdigest()
        return f"{base}|{handle_frames}|{padding_mode}|{handle_side}|{fps}"

    @classmethod
    def execute(
        cls,
        images,
        handle_frames=0,
        padding_mode="H3 (17n+5)",
        handle_side="head",
        fps=float(FPS),
        audio=None,
    ):
        if images.ndim == 3:
            images = images.unsqueeze(0)
        original = int(images.shape[0])
        target, head_pad, tail_pad, policy = plan_frame_handles(
            original,
            handle_frames=int(handle_frames),
            padding_mode=padding_mode,
            handle_side=handle_side,
        )
        out_images = apply_frame_padding(images, head_pad=head_pad, tail_pad=tail_pad)
        out_audio = sync_audio_to_frame_change(
            audio,
            head_pad=head_pad,
            tail_pad=tail_pad,
            fps=float(fps),
        )
        report = format_handle_report(
            original_frames=original,
            final_frames=int(out_images.shape[0]),
            head_pad=head_pad,
            tail_pad=tail_pad,
            padding_mode=padding_mode,
            fps=float(fps),
            policy_note=policy,
        )
        return io.NodeOutput(out_images, out_audio, int(out_images.shape[0]), report)
