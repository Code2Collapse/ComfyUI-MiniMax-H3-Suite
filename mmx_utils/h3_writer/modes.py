# PORTED FROM: third_party/ComfyUI-MiniMaxH3-Prompt-Writer/backend/assembly.py (MIT, H3 Prompt Writer contributors)
#   (_final_contract, ASPECT_RATIOS) and backend/media.py (MIT, H3 Prompt Writer contributors) (REFERENCE_LIMITS)
"""The node's modes: which guide, which references, which final grounding contract.

The four base modes and Reference are the Writer's own. V2V edit, Replacement and Continuation are this node's
additions: they use the reference guide with the task classification fixed instead of inferred from the brief.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


class WriterError(Exception):
    """A failure the node reports in one plain sentence (no codes, no traceback text)."""

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


@dataclass(frozen=True)
class Mode:
    label: str
    key: str
    guide: str                      # "base" | "reference"
    pictures: int | None = None     # exact picture count for a base mode; None for reference modes
    needs_video: bool = False
    needs_picture: bool = False


MODE_TABLE: tuple[Mode, ...] = (
    Mode("T2VA - text to video", "T2VA", "base", pictures=0),
    Mode("I2VA - first frame", "I2VA", "base", pictures=1),
    Mode("FL2VA - first and last frame", "FL2VA", "base", pictures=2),
    Mode("L2VA - last frame", "L2VA", "base", pictures=1),
    Mode("Reference - reference to video", "Reference", "reference"),
    Mode("V2V edit - edit the source video", "V2VEdit", "reference", needs_video=True),
    Mode("Replacement - replace a subject in the plate", "Replacement", "reference", needs_video=True,
         needs_picture=True),
    Mode("Continuation - extend the source video", "Continuation", "reference", needs_video=True),
)
MODES: tuple[str, ...] = tuple(m.label for m in MODE_TABLE)
_BY_LABEL = {m.label: m for m in MODE_TABLE}
_BY_KEY = {m.key: m for m in MODE_TABLE}
REFERENCE_MODE_KEYS = frozenset(m.key for m in MODE_TABLE if m.guide == "reference")

ASPECT_RATIOS: tuple[str, ...] = ("16:9", "9:16", "1:1", "4:3", "3:4", "3:2", "2:3", "21:9")
REFERENCE_LIMITS = {"image": 9, "video": 3, "audio": 3, "total": 12}


def mode_for(label_or_key: str) -> Mode:
    mode = _BY_LABEL.get(label_or_key) or _BY_KEY.get(label_or_key)
    if mode is None:
        raise WriterError(f"Unknown mode {label_or_key!r}. Pick one of: {', '.join(MODES)}.")
    return mode


def is_reference_mode(label_or_key: str) -> bool:
    key = _BY_LABEL[label_or_key].key if label_or_key in _BY_LABEL else label_or_key
    return key in REFERENCE_MODE_KEYS


def task_label(mode_key: str, plate_has_audio: bool, has_audio_ref: bool = False) -> str:
    """The summary's square-bracketed task type for the fixed-classification modes (ours)."""
    parts = {
        "Reference": ["reference generation"],
        "V2VEdit": ["video editing"],
        "Replacement": ["video editing", "reference generation"],
        "Continuation": ["video continuation"],
    }.get(mode_key, [])
    if mode_key in ("V2VEdit", "Replacement") and plate_has_audio:
        parts.append("audio reuse")
    elif mode_key == "Continuation" and plate_has_audio:
        parts.append("audio reference")
    if has_audio_ref and mode_key == "Reference":
        parts.append("audio reference")
    return "[" + " + ".join(parts) + "]"


_BASE_RULES = {
    "T2VA": "Preserve any explicit continuous-camera or no-cut instruction instead of introducing an unsupported cut.",
    "I2VA": "Separate facts visible in the first frame from newly requested space or action revealed after it.",
    "FL2VA": "Prioritize exact endpoint geometry and a continuous state/camera path between the first and last frames.",
    "L2VA": "Invent only the minimum compatible preceding state needed to reach the final frame; do not infer a named location or period without evidence.",
}

_REFERENCE_BODY = (
    "Treat every explicitly assigned reference role as exclusive unless the user asks that reference to contribute "
    "additional traits; 'only' and 'solely' emphasize this rule but are not required. Unspecified target environment, lighting, "
    "composition, camera treatment, and atmosphere may be designed as new target content, but never described as "
    "facts derived from a reference. Do not add unsupported subject actions, dialogue, props, visible text, or an "
    "invented ending. Music requested without an uploaded audio asset belongs only in non_diegetic_music and must "
    "not create audio-reference or audio-reuse semantics. Prefer one continuous shot unless cuts are requested; "
    "purposeful camera movement within that shot is allowed. Because H3 receives each source video itself, bind the "
    "complete choreography, temporal order, pacing, and rhythmic character of a motion-only video without "
    "reconstructing individual sampled gestures, named steps, poses, expressions, transitions, or a concluding move. "
    "When a concrete visible object, character, scene, or effect from <Video N> is reused in the target, describe that "
    "reused visual element through an appropriate <Subject N> while keeping <Video N> as its source provenance; do not "
    "automatically create a separate subject for ordinary motion transfer. "
    "If the brief does not explicitly request music, non_diegetic_music must be N/A. "
    "Use the official detail budget for grounded target composition, placement, lighting, atmosphere, camera treatment, "
    "supported action progression, and reference application; never pad solely to reach a word count. Return only the complete "
    "prompt with all six required sections in the official order and no commentary outside the prompt."
)


def final_contract(mode_key: str, task_text: str, plate_has_audio: bool = False) -> str:
    """The closing grounding check appended to the user message (ported; three classifications added)."""
    if mode_key in _BASE_RULES:
        return (
            f"Final grounding check: {_BASE_RULES[mode_key]} "
            "If the brief does not explicitly request non-diegetic music, return N/A for non_diegetic_music. "
            "Return only the complete final MiniMax H3 prompt."
        )
    if mode_key == "Reference":
        explicit_edit = bool(re.search(
            r"\b(?:edit(?:ing)?|continue|continuation|extend|remix|re-cut)\b.{0,40}\bvideo\b|\bvideo\s+editing\b",
            task_text,
            re.IGNORECASE | re.DOTALL,
        ))
        classification = (
            "source-video editing or continuation; scale detailed_description with source complexity"
            if explicit_edit
            else "reference generation, not keyframe completion or source-video editing"
        )
    elif mode_key == "V2VEdit":
        classification = (
            "source-video editing of <Video 1>; begin the summary after the task label with 'The target video is an "
            "edited version of <Video 1>.'; scale detailed_description with source complexity. Task label: "
            + task_label(mode_key, plate_has_audio)
        )
    elif mode_key == "Replacement":
        classification = (
            "video editing + reference generation: <Video 1> is the plate and keeps its composition, camera, motion, "
            "timing, setting, lighting and sound; each <Picture N> supplies only the new subject's identity and "
            "appearance; begin the summary after the task label with 'The target video is an edited version of "
            "<Video 1>.'. Task label: " + task_label(mode_key, plate_has_audio)
        )
    elif mode_key == "Continuation":
        classification = (
            "video continuation of <Video 1>; new content continues from its final state. Task label: "
            + task_label(mode_key, plate_has_audio)
        )
    else:
        raise WriterError(f"Unknown mode {mode_key!r}.")
    return f"Final request classification: {classification}. " + _REFERENCE_BODY
