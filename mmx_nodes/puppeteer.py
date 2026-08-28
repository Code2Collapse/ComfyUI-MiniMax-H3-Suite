"""N15 — Pose puppeteer: driving performance as conditioning hints (catalog #61)."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import torch

from comfy_api.latest import io

_PKG = Path(__file__).resolve().parents[1]
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from mmx_utils.device import run_with_cpu_fallback
from mmx_utils.puppeteer import RIG_TYPES, render_motion_hints

try:
    import comfy.model_management as mm
except ImportError:
    mm = None


class MiniMaxH3_PosePuppeteer(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MiniMaxH3_PosePuppeteer",
            display_name="H3 Pose Puppeteer",
            category="MiniMax H3/Control",
            description=(
                "Package N8 driving pose as conditioning-ready motion_hints + motion_text (#61). "
                "Union is blocked (D1) — hints only; no Union apply path exists today. "
                "Does not drive generation by itself."
            ),
            inputs=[
                io.Image.Input("driving_pose", tooltip="Pose hints IMAGE from N8 StrongestPose."),
                io.String.Input("keypoints_json", optional=True, multiline=True),
                io.Image.Input("character_ref"),
                io.Combo.Input("rig_type", options=list(RIG_TYPES), default="human_to_creature"),
            ],
            outputs=[
                io.Image.Output("motion_hints"),
                io.String.Output("motion_text"),
                io.String.Output("report"),
            ],
        )

    @classmethod
    def fingerprint_inputs(cls, driving_pose, keypoints_json=None, character_ref=None, rig_type="human_to_creature"):
        h = hashlib.md5(driving_pose.cpu().numpy().tobytes()).hexdigest()
        if character_ref is not None:
            h += ":" + hashlib.md5(character_ref.cpu().numpy().tobytes()).hexdigest()
        if keypoints_json:
            h += ":" + hashlib.md5(keypoints_json.encode()).hexdigest()
        return f"{h}|{rig_type}"

    @classmethod
    def execute(
        cls,
        driving_pose,
        keypoints_json=None,
        character_ref=None,
        rig_type="human_to_creature",
    ) -> io.NodeOutput:
        if character_ref is None:
            raise ValueError("character_ref is required — connect a reference plate for the creature/character.")
        dev = driving_pose.device
        if mm is not None:
            try:
                dev = mm.get_torch_device()
            except Exception:
                dev = driving_pose.device

        def _work(device: torch.device) -> tuple[torch.Tensor, str, str]:
            hints, text = render_motion_hints(
                driving_pose.to(device),
                character_ref.to(device),
                rig_type=rig_type,
                keypoints_json=keypoints_json,
            )
            report = (
                f"rig={rig_type} frames={int(hints.shape[0])} "
                f"size={hints.shape[2]}x{hints.shape[1]} — conditioning hints only (Union blocked D1)"
            )
            return hints, text, report

        hints, text, report = run_with_cpu_fallback(_work, device=dev, label="MiniMaxH3_PosePuppeteer")
        return io.NodeOutput(hints.to(driving_pose.dtype), text, report)
