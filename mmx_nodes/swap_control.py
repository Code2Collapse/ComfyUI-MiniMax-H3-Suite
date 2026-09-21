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
    has_pupils,
    pose_keypoint_to_wholebody,
)
from mmx_utils.swap_control import render_swap_control, scope_edges  # noqa: E402
from mmx_utils.swap_mask import build_swap_mask, describe_mask  # noqa: E402
from mmx_utils.swap_regions import MASK_PAD, SWAP_SCOPES, describe  # noqa: E402

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
                "The whole face drives it: jaw for head pose and chin drop, "
                "brows and eyes for expression, PUPILS for gaze direction, "
                "mouth for lip movement. The same region is masked, so the "
                "skull shape can still move toward your reference actor while "
                "the dupe's performance comes through.\n\n"
                "That is a real tension, not a solved problem: a strong jaw "
                "control pulls face width toward the dupe. Lower the "
                "ControlNet strength if the head starts taking the dupe's "
                "shape, or turn drive_jaw off to remove the contour from the "
                "control entirely - at the cost of head pose and chin drop.\n\n"
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
                    tooltip="Which VFX operation this is. face: the whole "
                            "face drives it and the face region is masked. "
                            "lips: mouth and jaw drive it - the chin has to "
                            "drop for the mouth to open - but only the mouth "
                            "is regenerated, so a lip-sync never reshapes the "
                            "chin. head: as face, with hair and skull in the "
                            "mask too. body: body, feet and hands, face left "
                            "alone. person: all of it."),
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
                io.Boolean.Input(
                    "drive_jaw", default=True, optional=True,
                    tooltip="Draw the dupe's jaw contour into the control. ON "
                            "is usually right: those 17 points carry head POSE "
                            "(which way the head is turned) and the CHIN DROP "
                            "that lets the mouth open, not just skull shape. "
                            "They also carry the dupe's face WIDTH, so if the "
                            "swapped head starts looking like the dupe, lower "
                            "the ControlNet strength before reaching for this. "
                            "OFF removes the contour entirely - the reference "
                            "actor's skull is then unopposed, but a profile "
                            "will read as a front-on face and the mouth cannot "
                            "open as far."),
                io.Boolean.Input(
                    "drive_mouth", default=True, optional=True,
                    tooltip="Draw the dupe's lips into the control. Leave it ON "
                            "when the audio is that dupe's own take, or when "
                            "there is no audio at all - then the lips follow "
                            "the performance. Turn it OFF for a DUB: the mouth "
                            "is still masked, so it is free to change, but "
                            "nothing is telling it the old lip shape and a "
                            "locked audio track decides instead. With it off "
                            "and no audio locked, the lips have nothing driving "
                            "them at all."),
                io.Float.Input(
                    "mask_pad", default=-1.0, min=-1.0, max=2.0, step=0.01,
                    optional=True,
                    tooltip="How far the region mask reaches past the landmark "
                            "hull, as a fraction of the hull's own size. -1 "
                            "uses the per-scope default (head pads furthest, "
                            "because hair has no landmarks to find). Padding "
                            "scales about the hull's centroid, so growing it "
                            "does not move the boundary relative to the face."),
                io.Int.Input(
                    "mask_feather", default=0, min=0, max=64, optional=True,
                    tooltip="Soften the mask edge, in pixels. Feather in PIXEL "
                            "space here; the latent reduction downstream is "
                            "where the 32px token grid takes over."),
                io.Int.Input(
                    "person_index", default=0, min=0, max=32, optional=True,
                    tooltip="Which detected person is the dupe, when the frame "
                            "holds more than one."),
            ],
            outputs=[
                io.Image.Output(display_name="control_video",
                                tooltip="Feed to H3 Mask-Aware ControlNet's "
                                        "control_video."),
                io.Mask.Output(display_name="region_mask",
                               tooltip="The region allowed to change, built "
                                       "from the SAME landmarks as the control "
                                       "so the two cannot drift apart. Send it "
                                       "to H3 Mask To Latent Space (with "
                                       "spatial_method=coverage, and grow it "
                                       "with grow_tokens) and to the "
                                       "ControlNet's pixel-space mask. Masked "
                                       "and driven are NOT the same set: the "
                                       "jaw is both, so head pose comes through "
                                       "while the skull can still move toward "
                                       "your reference, and 'lips' drives the "
                                       "jaw for the chin drop but masks only "
                                       "the mouth, so a lip-sync never reshapes "
                                       "the chin."),
                io.String.Output(display_name="report"),
            ],
        )

    @classmethod
    def execute(cls, pose_keypoint, swap_scope, width, height, confidence_gate,
                line_width, face_line_width, drive_jaw=True, drive_mouth=True,
                mask_pad=-1.0, mask_feather=0, person_index=0):
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
            drive_jaw=bool(drive_jaw),
            drive_mouth=bool(drive_mouth),
        )

        edges = scope_edges(swap_scope, drive_jaw=bool(drive_jaw),
                            drive_mouth=bool(drive_mouth))
        possible = len(edges) * max(1, frames.shape[0])
        drawn = int(sum(
            1 for t in range(frames.shape[0])
            for (a, b), _ in edges
            if kps[t, a, 2] >= confidence_gate and kps[t, b, 2] >= confidence_gate
        ))

        lines = [
            describe(swap_scope, drive_jaw=bool(drive_jaw),
                     drive_mouth=bool(drive_mouth)),
            "",
            describe_source(kps, (width, height), confidence_gate),
            f"Drew {drawn} of {possible} possible edges.",
        ]
        if swap_scope in ("face", "head", "person") and not has_pupils(
                kps, confidence_gate):
            lines.append(
                "No pupils in this pose: eye DIRECTION will not be driven. The "
                "eyes still blink and squint from the lid contours, but they "
                "will not look where the dupe looks. Pupils need an "
                "OpenPose-family face detection (70 face points, not 68).")
        if swap_scope in ("face", "head", "lips", "person") and not has_face(
                kps, confidence_gate):
            lines.append(
                "WARNING: this scope needs face landmarks and none passed the "
                "gate, so nothing will drive the expression. Check the "
                "detector is whole-body, or lower confidence_gate.")

        masks = build_swap_mask(
            kps, swap_scope, (width, height),
            confidence_gate=confidence_gate,
            pad=None if mask_pad < 0 else float(mask_pad),
            feather=int(mask_feather),
        )
        lines.append("")
        lines.append(describe_mask(masks, swap_scope))
        used_pad = MASK_PAD[swap_scope] if mask_pad < 0 else mask_pad
        lines.append(
            f"Mask pad {used_pad:.2f} of the hull, scaled about its "
            "centroid so the boundary does not move relative to the face.")

        return io.NodeOutput(
            torch.from_numpy(np.ascontiguousarray(frames)),
            torch.from_numpy(np.ascontiguousarray(masks)),
            "\n".join(lines))


NODE_LIST = [MiniMaxH3_SwapControl]
