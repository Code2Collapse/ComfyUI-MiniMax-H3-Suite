"""Which landmarks drive a swap, and which must not.

THE PROBLEM THIS SOLVES
-----------------------
On H3 there is no face conditioning path - no face_images, no expression
coefficients, nothing (grep the model: `comfy/ldm/minimax/model.py` has no
face-specific segment). Expression and lip movement can only reach the model
through the CONTROL VIDEO, which the Fun ControlNet-Union accepts as Canny,
Depth, HED, MLSD or Pose.

So the control has to carry the dupe's performance, and the whole face is
driven. The face block is the dlib-68 layout:

    face[ 0..16]  jaw / face contour   <- head POSE, chin drop, AND skull shape
    face[17..26]  eyebrows             <- expression
    face[27..35]  nose                 <- position anchor
    face[36..47]  eyes                 <- lid opening, blinks
    face[48..67]  mouth                <- lip movement
    pupils (x2)   appended at 133,134  <- eye DIRECTION

THE JAW IS THE AWKWARD ONE and it is a genuine trade, not a bug to design
away. Those 17 points carry three separate things:

  * head POSE - the contour says which way the head is turned. Drop it and a
    profile reads as a front-on face with strange features.
  * jaw ARTICULATION - the chin drops when the mouth opens. Drop it and you
    cap how far the mouth can open, which is the lip movement itself.
  * skull SHAPE - and this one you may not want, because it is the dupe's.

An earlier version of this module excluded the jaw to protect the third. That
bought face shape at the cost of the first two. The jaw is now DRIVEN by
default and also MASKED, so the pose and chin drop come through while the
region stays free to move toward the reference actor - with the tension
managed by ControlNet strength rather than by throwing the landmarks away.
`drive_jaw=False` remains for the cases where identity beats pose.

COCO-WholeBody 133 layout (what DWPose and SDPose emit):
    0..16    body (COCO-17)
    17..22   feet
    23..90   face (the dlib-68 block above, offset by 23)
    91..111  left hand
    112..132 right hand
    [133..134  pupils - our extension, see N_EXTENDED]
"""

from __future__ import annotations

from typing import Iterable

N_WHOLEBODY = 133
# Our internal array is 135: COCO-WholeBody's 133 plus the two PUPIL centres.
# OpenPose emits them (face_keypoints_2d is 70 = dlib-68 + 2 pupils) and
# COCO-WholeBody has no slot, so they used to be dropped on the floor. They
# are the only eye-DIRECTION signal in the data: the 68-point eye contours
# describe the lid opening and never where the eye is looking.
N_EXTENDED = 135

BODY = (0, 17)
FEET = (17, 23)
FACE = (23, 91)
LEFT_HAND = (91, 112)
RIGHT_HAND = (112, 133)
PUPILS = (133, 135)

_F = FACE[0]

# Named groups, as half-open [start, stop) ranges into the 133-point array.
GROUPS: dict[str, tuple[int, int]] = {
    "body": BODY,
    "feet": FEET,
    "left_hand": LEFT_HAND,
    "right_hand": RIGHT_HAND,
    "jaw": (_F + 0, _F + 17),
    "brows": (_F + 17, _F + 27),
    "nose": (_F + 27, _F + 36),
    "eyes": (_F + 36, _F + 48),
    "mouth": (_F + 48, _F + 68),
    "pupils": PUPILS,
}

# Everything the face can do. The jaw is in here deliberately: it carries head
# POSE and jaw ARTICULATION as well as skull shape, and dropping it to protect
# the shape also drops those. `drive_jaw=False` is the lever for the cases
# where identity matters more than pose.
EXPRESSION = ("jaw", "brows", "nose", "eyes", "pupils", "mouth")

# Kept so `drive_jaw=False` has a name rather than a bare string literal.
SHAPE_BEARING = ("jaw",)

# The mouth is the one group another signal can drive. With the native audio
# lock in play H3 shapes the lips from the phonemes, so a control that also
# draws the dupe's mouth overrides it - fine when the audio is that dupe's own
# take, wrong for a dub. drive_mouth=False masks the mouth without driving it,
# which is the only way the audio gets to decide.
AUDIO_DRIVABLE = ("mouth",)

# scope -> (groups rendered into the control, what the silhouette follows)
#
# "plate" means the surrounding image is preserved and the swap is confined to
# the region; "ref" means the reference actor's anatomy decides the outline,
# which is only possible when the jaw is kept OUT of the control.
SWAP_SCOPES: dict[str, dict] = {
    "lips": {
        "groups": ("mouth", "jaw"),
        "silhouette": "plate",
        "why": "Mouth and jaw are driven - the chin has to drop for the mouth "
               "to open - and only the mouth region is regenerated. The rest "
               "of the face is the original plate untouched.",
    },
    "face": {
        "groups": EXPRESSION,
        "silhouette": "ref",
        "why": "The whole face drives it - jaw for head pose and chin drop, "
               "brows, eyes and pupils for expression and gaze direction, "
               "mouth for lip movement. The region is masked too, so the skull "
               "shape can still change toward the reference.",
    },
    "head": {
        "groups": EXPRESSION,
        "silhouette": "ref",
        "why": "As 'face', but the mask covers hair and the whole head, so the "
               "skull outline is regenerated too.",
    },
    "body": {
        "groups": ("body", "feet", "left_hand", "right_hand"),
        "silhouette": "ref",
        "why": "Body pose drives the performance; the face is left to the "
               "plate. Use when the head is already correct.",
    },
    "person": {
        "groups": ("body", "feet", "left_hand", "right_hand") + EXPRESSION,
        "silhouette": "ref",
        "why": "Everything the dupe does, driven - body, hands, face, gaze.",
    },
}

# Nothing is excluded unconditionally any more. `drive_jaw=False` drops the
# shape-bearing group at call time; see scope_groups().
NEVER_RENDERED: tuple[str, ...] = ()

# CONTROL groups (above) say what the dupe DRIVES.
# MASK groups (below) say what is allowed to CHANGE.
#
# They are still not the same set - `lips` drives the jaw so the chin can drop
# but masks only the mouth, because a lip-sync must not reshape the chin - but
# for face/head/person the jaw is now in BOTH:
#
#   jaw DRIVEN -> head pose and chin drop transfer from the dupe
#   jaw MASKED -> the region is regenerated, so it can still move toward the
#                 reference actor's skull
#
# Those two pull against each other, and that is the point: the control says
# where the contour is, the reference says what shape it should be, and
# ControlNet strength decides who wins. Turn the strength up and the head
# takes the dupe's width; turn it down and it drifts toward the reference.
MASK_GROUPS: dict[str, tuple[str, ...]] = {
    "lips": ("mouth",),
    "face": EXPRESSION,
    "head": EXPRESSION,
    "body": ("body", "feet", "left_hand", "right_hand"),
    "person": EXPRESSION + ("body", "feet", "left_hand", "right_hand"),
}

# How far past the landmark hull each scope's mask reaches, as a fraction of
# the hull's own size. A face hull stops at the skin; a head has to take in
# hair, which no landmark marks at all.
MASK_PAD: dict[str, float] = {
    "lips": 0.25,
    "face": 0.12,
    "head": 0.55,
    "body": 0.10,
    "person": 0.15,
}


def mask_groups(scope: str) -> tuple[str, ...]:
    if scope not in MASK_GROUPS:
        raise SwapRegionError(
            f"Unknown swap scope {scope!r}. Choose one of: "
            + ", ".join(sorted(SWAP_SCOPES)) + "."
        )
    return MASK_GROUPS[scope]


def mask_indices(scope: str) -> list[int]:
    return group_indices(mask_groups(scope))


class SwapRegionError(ValueError):
    """Raised with a sentence naming what to do instead."""


def scope_groups(scope: str, *, drive_jaw: bool = True,
                 drive_mouth: bool = True) -> tuple[str, ...]:
    """Groups the control renders for this scope.

    drive_jaw=False drops the face contour, trading head pose and chin drop
    for a stronger guarantee that the dupe's skull width does not transfer.

    drive_mouth=False drops the lips, so a locked audio track decides the
    mouth instead of the dupe's performance. That is the setting for a dub;
    leave it on when the audio is the dupe's own take.

    Both are trades. Neither is free, and the mouth is still MASKED either
    way - what changes is only who tells it what to do.
    """
    if scope not in SWAP_SCOPES:
        raise SwapRegionError(
            f"Unknown swap scope {scope!r}. Choose one of: "
            + ", ".join(sorted(SWAP_SCOPES)) + "."
        )
    groups = tuple(SWAP_SCOPES[scope]["groups"])
    if not drive_jaw:
        groups = tuple(g for g in groups if g not in SHAPE_BEARING)
    if not drive_mouth:
        groups = tuple(g for g in groups if g not in AUDIO_DRIVABLE)
    return groups


def group_indices(groups: Iterable[str]) -> list[int]:
    """Flatten named groups to sorted 133-point indices."""
    out: set[int] = set()
    for g in groups:
        if g not in GROUPS:
            raise SwapRegionError(
                f"Unknown landmark group {g!r}. Known groups: "
                + ", ".join(sorted(GROUPS)) + "."
            )
        lo, hi = GROUPS[g]
        out.update(range(lo, hi))
    return sorted(out)


def scope_indices(scope: str, *, drive_jaw: bool = True,
                  drive_mouth: bool = True) -> list[int]:
    return group_indices(scope_groups(scope, drive_jaw=drive_jaw,
                                      drive_mouth=drive_mouth))


def select(keypoints, scope: str, *, drive_jaw: bool = True,
           drive_mouth: bool = True):
    """Keep only the scope's landmarks; zero the confidence of the rest.

    Takes and returns an array shaped [..., 133, C] with C >= 3 (x, y, score).
    Points outside the scope have their SCORE set to 0 rather than being
    removed, so the array keeps the 133-point layout every downstream renderer
    and detector expects. A renderer that honours score will then skip them.
    """
    import numpy as np

    arr = np.asarray(keypoints)
    if arr.shape[-2] not in (N_WHOLEBODY, N_EXTENDED):
        raise SwapRegionError(
            f"Expected {N_WHOLEBODY} COCO-WholeBody keypoints (what DWPose and "
            f"SDPose emit), or {N_EXTENDED} with the two pupils appended, got "
            f"{arr.shape[-2]}. A 17-point body-only pose cannot carry "
            "expression - it has no face landmarks at all."
        )
    if arr.shape[-1] < 3:
        raise SwapRegionError(
            "Keypoints need at least (x, y, score); scores are how the "
            "excluded landmarks are suppressed.")
    n = arr.shape[-2]
    keep = [i for i in scope_indices(scope, drive_jaw=drive_jaw,
                                     drive_mouth=drive_mouth) if i < n]
    out = arr.copy()
    mask = np.zeros(n, dtype=bool)
    mask[keep] = True
    out[..., ~mask, 2] = 0.0
    return out


def describe(scope: str, *, drive_jaw: bool = True,
             drive_mouth: bool = True) -> str:
    spec = SWAP_SCOPES.get(scope)
    if spec is None:
        return f"Unknown scope {scope!r}."
    groups = scope_groups(scope, drive_jaw=drive_jaw, drive_mouth=drive_mouth)
    n = len(scope_indices(scope, drive_jaw=drive_jaw, drive_mouth=drive_mouth))
    lines = [
        f"Swap scope '{scope}': {n} of {N_EXTENDED} landmarks drive the control.",
        "Groups: " + ", ".join(groups) + ".",
        spec["why"],
    ]
    if "pupils" in groups:
        lines.append(
            "Pupils are driven, so eye DIRECTION transfers. The 68-point eye "
            "contours only describe the lid; gaze lives in the pupil centres.")
    if "mouth" in MASK_GROUPS.get(scope, ()) and "mouth" not in groups:
        lines.append(
            "The mouth is masked but NOT driven, so a locked audio track "
            "decides the lip shape. That is the dub setting. With no audio "
            "locked the lips have nothing telling them what to do.")
    jaw_driven = "jaw" in groups
    jaw_masked = "jaw" in MASK_GROUPS.get(scope, ())
    if jaw_driven and jaw_masked:
        lines.append(
            "The jaw is BOTH driven and masked: the dupe's head pose and chin "
            "drop come through, and the region is still free to move toward "
            "the reference actor's skull. The tension is real - a strong jaw "
            "control pulls the face width toward the dupe - so lower the "
            "ControlNet strength if the head starts taking the dupe's shape.")
    elif jaw_masked:
        lines.append(
            "The jaw is masked but NOT driven: the skull can change toward the "
            "reference, at the cost of the dupe's head pose and chin drop.")
    return "\n".join(lines)
