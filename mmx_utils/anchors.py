# SPDX-License-Identifier: GPL-3.0-or-later
# PORTED FROM: open-source-higgsfield (lucataco, Apache-2.0) :: bodydouble/runner.py refs() / check_props() @ 864c8de
#   (third_party/open-source-higgsfield; licence + NOTICE in THIRD_PARTY_LICENSES/open-source-higgsfield-*)
"""Anchor candidates and prop consistency for the character swap (owner A21 Rev 3 Part 10; ledger L9.78 slice 4).

Ported from the reference:
- an anchor's person is the detected person that overlaps most where the performer stands in the source frame
  (``refs``);
- the prop rule (``check_props``). A prop counts on a view when its mask touches the person's surroundings (about
  8 px at 704 rows), is smaller than a tenth of the person, and is mostly new pixels (SAM answers several names for
  one small object: the first name keeps it). A prop the performer holds in the source frame is the source's: it is
  allowed in the anchor (the anchor IS the first frame repainted). One that only the references (or the prompt)
  have stops the take.

Added here: user-locked props (always allowed, reported as such) and a warn-only policy.
"""
from __future__ import annotations

import re

import numpy as np

from . import body_double as BD


def person_overlapping(masks: list, where, min_share: float = 0.0) -> int | None:
    """Index of the mask that overlaps ``where`` most (no ``where``: the largest); None when there is no mask, or
    when the best covers no more than ``min_share`` of ``where`` (someone merely touching the spot)."""
    k, share = person_overlap(masks, where)
    if k is None or (where is not None and share <= float(min_share)):
        return None
    return k


def person_overlap(masks: list, where) -> tuple[int | None, float]:
    """(index of the mask overlapping ``where`` most, the share of ``where`` it covers); no ``where``: the largest,
    share 1.0."""
    best, best_v = None, -1.0
    w = BD._np(where) > 0.5 if where is not None else None
    for k, m in enumerate(masks):
        a = BD._np(m) > 0.5
        v = float((a & w).sum()) if w is not None and w.shape == a.shape else float(a.sum())
        if v > best_v:
            best, best_v = k, v
    if best is None:
        return None, 0.0
    if w is None or not w.any():
        return best, 1.0
    return best, best_v / float(w.sum())


def near_zone(person: np.ndarray) -> np.ndarray:
    """The person's surroundings: quarter size, a 5 x 5 max filter, back to full size (about 8 px at 704 rows)."""
    h, w = person.shape
    q = BD.resize2d(person.astype(np.float32), max(1, h // 4), max(1, w // 4)) > 0
    return BD.resize2d(BD.dilate(q.astype(np.float32), 5), h, w, mode="nearest") > 0


def check_props(views: list[dict], names: list[str], prompt: str = "", locked: list[str] | None = None,
                anchor_view: int | None = None) -> dict:
    """``views``: [{"name", "person": HxW bool, "props": {name: HxW mask}}], view 0 = the source frame.
    Returns held (in the source), found {name: [view names]}, allowed_locked, ok."""
    locked_set = {n.lower() for n in (locked or [])}
    held: set[str] = set()
    found: dict[str, list[str]] = {}
    detail: list[dict] = []
    for i, v in enumerate(views):
        person = BD._np(v["person"]) > 0.5
        near = near_zone(person)
        seen = np.zeros(person.shape, bool)
        for n in names:
            m = v["props"].get(n)
            if m is None:
                continue
            m = BD._np(m) > 0.5
            if m.shape != person.shape:
                m = BD.resize2d(m.astype(np.float32), *person.shape) > 0.5
            counts = bool((m & near).any() and m.sum() < person.sum() / 10 and (m & ~seen).sum() > m.sum() / 2)
            detail.append({"view": v["name"], "prop": n, "pixels": int(m.sum()), "counts": counts})
            if counts:
                if i == 0:
                    held.add(n)
                elif not (anchor_view is not None and i == anchor_view and n in held):
                    found.setdefault(n, []).append(v["name"])
            seen |= m
    for n in names:
        if prompt and re.search(rf"\b{re.escape(n)}s?\b", prompt, re.I):
            found.setdefault(n, []).append("prompt")
    allowed = {n: w for n, w in found.items() if n.lower() in locked_set}
    found = {n: w for n, w in found.items() if n.lower() not in locked_set}
    return {"held_in_source": sorted(held), "found": found, "allowed_locked": allowed, "ok": not found,
            "detail": detail}


def explain(found: dict[str, list[str]]) -> str:
    what = "; ".join(f"{n} in {', '.join(w)}" for n, w in found.items())
    first = next(iter(found))
    return (f"the new character holds a prop the performer does not ({what}). The video model takes props from the "
            f"source performance, so a prop that only the references or the prompt have pops in and out where the "
            f"performer holds nothing. Remove it from the prompt and the references (e.g. edit the reference: "
            f"\"Remove the {first}. Keep everything else the same.\"), or lock it in locked_props if it is wanted.")
