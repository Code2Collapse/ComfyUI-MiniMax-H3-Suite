"""POSE_KEYPOINT (OpenPose JSON) -> COCO-WholeBody 133.

Every pose node in the ecosystem - DWPose, SDPose's ComfyUI workflow,
controlnet_aux - hands over ComfyUI's POSE_KEYPOINT, which is OpenPose JSON.
That is NOT the COCO-WholeBody 133 array the swap tables index, and the
differences are the kind that produce a silently mirrored or shifted skeleton
rather than an error:

  body   OpenPose COCO-18 has a NECK at index 1 and orders the head points
         nose, neck, Rshoulder... COCO-17 has no neck and is
         nose, Leye, Reye, Lear, Rear, Lshoulder... So the mapping is a
         permutation AND a drop, not a slice. Getting it wrong swaps left and
         right, which on a face swap mirrors the expression.

  face   OpenPose emits 70 points: the dlib-68 followed by two PUPILS. The
         first 68 line up exactly; the pupils have no COCO-WholeBody slot and
         are dropped.

  feet   OpenPose COCO-18 has none. Those six slots stay at score 0, so the
         confidence gate drops them and no foot line is invented.

  hands  21 each, same order. Direct copy.

Coordinates: ComfyUI's POSE_KEYPOINT is NORMALISED to 0..1 against
canvas_width/canvas_height. The renderer wants pixels, so they are scaled here
and the canvas is returned with them - a caller that assumes pixels gets a
skeleton in the top-left 1x1 pixel, which is exactly the bug this docstring
exists to prevent.
"""

from __future__ import annotations

import numpy as np

from .swap_regions import FACE, GROUPS, N_WHOLEBODY, SwapRegionError

# COCO-17 slot -> OpenPose COCO-18 slot.
_OP18_TO_COCO17 = (0, 15, 14, 17, 16, 5, 2, 6, 3, 7, 4, 11, 8, 12, 9, 13, 10)

_N_OP_BODY = 18
_N_OP_FACE = 70          # dlib-68 + 2 pupils
_N_HAND = 21


class PoseInteropError(SwapRegionError):
    """Raised with a sentence naming the format that was actually received."""


def _triples(flat, expected, label):
    arr = np.asarray(flat, dtype=np.float32).reshape(-1, 3)
    if arr.shape[0] != expected:
        raise PoseInteropError(
            f"{label} has {arr.shape[0]} points, expected {expected}. This "
            "does not look like OpenPose JSON from a whole-body detector.")
    return arr


def person_to_wholebody(person: dict, canvas: tuple[int, int]) -> np.ndarray:
    """One POSE_KEYPOINT person -> [133, 3] in PIXELS."""
    w, h = float(canvas[0]), float(canvas[1])
    out = np.zeros((N_WHOLEBODY, 3), dtype=np.float32)

    body = person.get("pose_keypoints_2d")
    if body:
        b = _triples(body, _N_OP_BODY, "pose_keypoints_2d")
        for coco, op in enumerate(_OP18_TO_COCO17):
            out[coco] = b[op]

    face = person.get("face_keypoints_2d")
    if face:
        f = _triples(face, _N_OP_FACE, "face_keypoints_2d")
        out[FACE[0]:FACE[1]] = f[:68]        # pupils 68,69 have no slot

    for key, group in (("hand_left_keypoints_2d", "left_hand"),
                       ("hand_right_keypoints_2d", "right_hand")):
        raw = person.get(key)
        if raw:
            lo, hi = GROUPS[group]
            out[lo:hi] = _triples(raw, _N_HAND, key)

    # Normalised 0..1 -> pixels. A detector that already emits pixels would
    # have values far above 1, so only scale when it plainly has not.
    finite = out[out[:, 2] > 0][:, :2]
    if finite.size and float(np.nanmax(finite)) <= 1.5:
        out[:, 0] *= w
        out[:, 1] *= h
    return out


def pose_keypoint_to_wholebody(
    pose_keypoint, *, person_index: int = 0, canvas: tuple[int, int] | None = None
) -> tuple[np.ndarray, tuple[int, int]]:
    """POSE_KEYPOINT (list of frames) -> ([T,133,3] pixels, (w, h))."""
    if isinstance(pose_keypoint, dict):
        pose_keypoint = [pose_keypoint]
    if not isinstance(pose_keypoint, (list, tuple)) or not pose_keypoint:
        raise PoseInteropError(
            "POSE_KEYPOINT is empty. Connect a whole-body pose detector - a "
            "body-only one cannot carry expression, it has no face points.")

    first = pose_keypoint[0]
    if canvas is None:
        w = int(first.get("canvas_width") or 0)
        h = int(first.get("canvas_height") or 0)
        if w <= 0 or h <= 0:
            raise PoseInteropError(
                "POSE_KEYPOINT carries no canvas_width/canvas_height, so "
                "normalised coordinates cannot be turned into pixels. Pass the "
                "width and height explicitly.")
        canvas = (w, h)

    frames = []
    for i, frame in enumerate(pose_keypoint):
        people = frame.get("people") or []
        if not people:
            frames.append(np.zeros((N_WHOLEBODY, 3), dtype=np.float32))
            continue
        if person_index >= len(people):
            raise PoseInteropError(
                f"Frame {i} has {len(people)} person(s); person_index "
                f"{person_index} is out of range.")
        frames.append(person_to_wholebody(people[person_index], canvas))
    return np.stack(frames), canvas


def has_face(keypoints: np.ndarray, gate: float = 0.3) -> bool:
    """Whether any face landmark survived. A body-only pose silently produces
    a swap with no expression at all, which is worth saying out loud."""
    arr = np.asarray(keypoints)
    return bool((arr[..., FACE[0]:FACE[1], 2] >= gate).any())


def describe_source(keypoints: np.ndarray, canvas: tuple[int, int], gate: float = 0.3) -> str:
    arr = np.asarray(keypoints)
    face = int((arr[..., FACE[0]:FACE[1], 2] >= gate).sum())
    body = int((arr[..., :17, 2] >= gate).sum())
    n = max(1, arr.shape[0])
    lines = [
        f"{arr.shape[0]} frame(s) on a {canvas[0]}x{canvas[1]} canvas; "
        f"{body / n:.0f} body and {face / n:.0f} face landmarks per frame "
        "above the confidence gate."
    ]
    if face == 0:
        lines.append(
            "NO face landmarks. The detector is body-only, so expression and "
            "lip movement cannot be driven - use a whole-body detector "
            "(SDPose or DWPose), not a 17-point body model.")
    return "\n".join(lines)
