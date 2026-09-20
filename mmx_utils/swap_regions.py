"""Which landmarks drive a swap, and which must not.

THE PROBLEM THIS SOLVES
-----------------------
On H3 there is no face conditioning path - no face_images, no expression
coefficients, nothing (grep the model: `comfy/ldm/minimax/model.py` has no
face-specific segment). Expression and lip movement can only reach the model
through the CONTROL VIDEO, which the Fun ControlNet-Union accepts as Canny,
Depth, HED, MLSD or Pose.

So the control has to carry the dupe's performance. The trap is that it then
also carries the dupe's ANATOMY: draw the dupe's face contour into the control
and the swapped face inherits the dupe's jaw and skull, giving you the actor's
texture on the dupe's head.

DWPose (COCO-WholeBody, 133 points) makes the fix exact, because the face
block is the dlib-68 layout and the jaw is a contiguous run at the front of it:

    face[ 0..16]  jaw / face contour   <- the DUPE's skull. EXCLUDE.
    face[17..26]  eyebrows             <- expression
    face[27..35]  nose                 <- position anchor
    face[36..47]  eyes                 <- expression, blinks
    face[48..67]  mouth                <- lip movement

Render 17..67 and drop 0..16 and the dupe's performance drives the swap while
the dupe's head shape never enters the control at all. The silhouette is then
free to follow the reference actor.

COCO-WholeBody 133 layout (what DWPose emits):
    0..16    body (COCO-17)
    17..22   feet
    23..90   face (the dlib-68 block above, offset by 23)
    91..111  left hand
    112..132 right hand
"""

from __future__ import annotations

from typing import Iterable

N_WHOLEBODY = 133

BODY = (0, 17)
FEET = (17, 23)
FACE = (23, 91)
LEFT_HAND = (91, 112)
RIGHT_HAND = (112, 133)

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
}

# The features that carry performance without carrying skull shape.
EXPRESSION = ("brows", "nose", "eyes", "mouth")

# scope -> (groups rendered into the control, what the silhouette follows)
#
# "plate" means the surrounding image is preserved and the swap is confined to
# the region; "ref" means the reference actor's anatomy decides the outline,
# which is only possible when the jaw is kept OUT of the control.
SWAP_SCOPES: dict[str, dict] = {
    "lips": {
        "groups": ("mouth",),
        "silhouette": "plate",
        "why": "Only the mouth is driven and only the mouth is regenerated. "
               "The rest of the face, including the jaw, is the original plate "
               "untouched - this is the lipsync-only case.",
    },
    "face": {
        "groups": EXPRESSION,
        "silhouette": "ref",
        "why": "Brows, nose, eyes and mouth drive expression and lip movement. "
               "The jaw is deliberately absent, so the swapped face takes the "
               "reference actor's face shape rather than the dupe's.",
    },
    "head": {
        "groups": EXPRESSION,
        "silhouette": "ref",
        "why": "As 'face', but the mask covers hair and the whole head, so the "
               "skull outline is regenerated too - again from the reference, "
               "because the dupe's contour is not in the control.",
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
        "why": "Everything the dupe does, driven. The jaw is still excluded so "
               "the head keeps the reference actor's shape.",
    },
}

# The jaw is never rendered by any scope. Stated as data so a test can assert
# it rather than a reader having to check five tuples by eye.
NEVER_RENDERED = ("jaw",)


class SwapRegionError(ValueError):
    """Raised with a sentence naming what to do instead."""


def scope_groups(scope: str) -> tuple[str, ...]:
    if scope not in SWAP_SCOPES:
        raise SwapRegionError(
            f"Unknown swap scope {scope!r}. Choose one of: "
            + ", ".join(sorted(SWAP_SCOPES)) + "."
        )
    return tuple(SWAP_SCOPES[scope]["groups"])


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


def scope_indices(scope: str) -> list[int]:
    return group_indices(scope_groups(scope))


def select(keypoints, scope: str):
    """Keep only the scope's landmarks; zero the confidence of the rest.

    Takes and returns an array shaped [..., 133, C] with C >= 3 (x, y, score).
    Points outside the scope have their SCORE set to 0 rather than being
    removed, so the array keeps the 133-point layout every downstream renderer
    and detector expects. A renderer that honours score will then skip them.
    """
    import numpy as np

    arr = np.asarray(keypoints)
    if arr.shape[-2] != N_WHOLEBODY:
        raise SwapRegionError(
            f"Expected {N_WHOLEBODY} COCO-WholeBody keypoints (what DWPose "
            f"emits), got {arr.shape[-2]}. A 17-point body-only pose cannot "
            "carry expression - it has no face landmarks at all."
        )
    if arr.shape[-1] < 3:
        raise SwapRegionError(
            "Keypoints need at least (x, y, score); scores are how the "
            "excluded landmarks are suppressed.")
    keep = scope_indices(scope)
    out = arr.copy()
    mask = np.zeros(N_WHOLEBODY, dtype=bool)
    mask[keep] = True
    out[..., ~mask, 2] = 0.0
    return out


def describe(scope: str) -> str:
    spec = SWAP_SCOPES.get(scope)
    if spec is None:
        return f"Unknown scope {scope!r}."
    n = len(scope_indices(scope))
    lines = [
        f"Swap scope '{scope}': {n} of {N_WHOLEBODY} landmarks drive the control.",
        "Groups: " + ", ".join(spec["groups"]) + ".",
        spec["why"],
    ]
    if "jaw" not in spec["groups"]:
        lines.append(
            "The jaw contour (face points 0-16) is NOT in the control, so the "
            "dupe's skull shape cannot transfer.")
    return "\n".join(lines)
