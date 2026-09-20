"""MiniMaxH3_SwapControl — drive a swap from a dupe's performance.

H3 has no face conditioning path. There is no face_images input, no expression
coefficients, nothing face-specific anywhere in `comfy/ldm/minimax/model.py`.
Expression and lip movement can therefore only reach the model through the
CONTROL VIDEO, which the Fun ControlNet-Union accepts as Canny, Depth, HED,
MLSD or Pose.

So the control has to carry the dupe's performance - and the trap is that it
then also carries the dupe's ANATOMY. Draw the dupe's face contour into the
control and the swap comes back with the reference actor's texture on the
dupe's skull.

This node renders only the landmark groups that carry PERFORMANCE - brows,
nose, eyes, mouth - and never the jaw. See mmx_utils/swap_regions.py; there is
no jaw edge table in the codebase to switch on.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from comfy_api.latest import io

_PKG = Path(__file__).resolve().parents[1]
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from mmx_utils.pose_interop import (  # noqa: E402
    describe_source,
    has_face,
    pose_keypoint_to_wholebody,
)
from mmx_utils.swap_control import render_swap_control, scope_edges  # noqa: E402
from mmx_utils.swap_regions import SWAP_SCOPES, describe  # noqa: E402

SCOPES = ["face", "lips", "head", "body", "person"]


class MiniMaxH3_SwapControl(io.ComfyNode):

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3_SwapControl",
            display_name="H3 Swap Control (face / head / body / lips)",
            category="MiniMax H3/Control",
            description=(
                "Turn a dupe's detected pose into the control video that drives "
                "a swap on H3. H3 has NO face conditioning path - no "
                "face_images, no expression coefficients - so expression and "
                "lip movement can only arrive through the control video.\n\n"
                "The point of this node is what it leaves OUT. It renders the "
                "landmark groups that carry performance (brows, nose, eyes, "
                "mouth) and never the jaw contour, so the dupe drives the "
                "expression and the feature positions while the head SHAPE "
                "stays free to follow your reference actor. A square-headed "
                "dupe does not give a square-headed swap.\n\n"
                "Feed the output to the control_video input of H3 Mask-Aware "
                "ControlNet. Any whole-body detector works - SDPose is the "
                "strongest (72.8 AP on COCO-WholeBody against DWPose's ~66, "
                "and better out of domain) - but a BODY-ONLY detector has no "
                "face landmarks at all and cannot drive expression."
            ),
            inputs=[
                io.Custom("POSE_KEYPOINT").Input(
                    "pose_keypoint",
                    tooltip="Whole-body pose from the DUPE - the performance "
                            "you want recreated. Must be a whole-body detector "
                            "(SDPose, DWPose, ViTPose-wholebody): a 17-point "
                            "body model carries no face landmarks, so there is "
                            "nothing to drive the lips with."),
                io.Combo.Input(
                    "swap_scope", options=SCOPES, default="face",
                    tooltip="Which VFX operation this is. face: brows, nose, "
                            "eyes and mouth drive it, jaw excluded, head shape "
                            "from your reference. lips: mouth only, everything "
                            "else is the untouched plate - the lipsync-only "
                            "case. head: as face but hair and skull are "
                            "regenerated too. body: body, feet and hands, face "
                            "left alone. person: all of it."),
                io.Int.Input(
                    "width", default=768, min=64, max=8192, step=8,
                    tooltip="Control frame width. Match the generation, or the "
                            "control lands at the wrong scale."),
                io.Int.Input(
                    "height", default=432, min=64, max=8192, step=8,
                    tooltip="Control frame height. Match the generation."),
                io.Float.Input(
                    "confidence_gate", default=0.3, min=0.0, max=1.0, step=0.01,
                    tooltip="Landmarks below this are not drawn. Raise it if a "
                            "flickery detection is making lines twitch between "
                            "frames; lower it if features drop out during fast "
                            "motion."),
                io.Int.Input(
                    "line_width", default=4, min=1, max=32,
                    tooltip="Body line thickness in pixels."),
                io.Int.Input(
                    "face_line_width", default=0, min=0, max=32,
                    tooltip="Face line thickness. 0 means half the body width, "
                            "which is the useful default: a thick line closes "
                            "the gap between the lips, and the model then "
                            "cannot tell an open mouth from a shut one."),
                io.Int.Input(
                    "person_index", default=0, min=0, max=32, optional=True,
                    tooltip="Which detected person is the dupe, when the frame "
                            "holds more than one."),
            ],
            outputs=[
                io.Image.Output(display_name="control_video",
                                tooltip="Feed to H3 Mask-Aware ControlNet's "
                                        "control_video."),
                io.String.Output(display_name="report"),
            ],
        )

    @classmethod
    def execute(cls, pose_keypoint, swap_scope, width, height, confidence_gate,
                line_width, face_line_width, person_index=0):
        if swap_scope not in SWAP_SCOPES:
            raise ValueError(
                f"Unknown swap scope {swap_scope!r}. Choose one of: "
                + ", ".join(SCOPES) + ".")

        kps, canvas = pose_keypoint_to_wholebody(
            pose_keypoint, person_index=person_index, canvas=None)

        # The detector's canvas is rarely the generation size; rescale rather
        # than render a skeleton that sits in a corner of the frame.
        sx, sy = width / float(canvas[0]), height / float(canvas[1])
        if abs(sx - 1.0) > 1e-6 or abs(sy - 1.0) > 1e-6:
            kps = kps.copy()
            kps[..., 0] *= sx
            kps[..., 1] *= sy

        frames = render_swap_control(
            kps, swap_scope, (width, height),
            confidence_gate=confidence_gate,
            line_width=int(line_width),
            face_line_width=int(face_line_width) or None,
        )

        possible = len(scope_edges(swap_scope)) * max(1, frames.shape[0])
        drawn = int(sum(
            1 for t in range(frames.shape[0])
            for (a, b), _ in scope_edges(swap_scope)
            if kps[t, a, 2] >= confidence_gate and kps[t, b, 2] >= confidence_gate
        ))

        lines = [
            describe(swap_scope),
            "",
            describe_source(kps, (width, height), confidence_gate),
            f"Drew {drawn} of {possible} possible edges.",
        ]
        if swap_scope in ("face", "head", "lips", "person") and not has_face(
                kps, confidence_gate):
            lines.append(
                "WARNING: this scope needs face landmarks and none passed the "
                "gate, so nothing will drive the expression. Check the "
                "detector is whole-body, or lower confidence_gate.")

        return io.NodeOutput(
            torch.from_numpy(np.ascontiguousarray(frames)), "\n".join(lines))


NODE_LIST = [MiniMaxH3_SwapControl]
