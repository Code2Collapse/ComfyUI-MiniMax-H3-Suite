"""Offline H3 prompt in the guide structure, without any model (ours).

It lays the brief out in the mode's official structure and assigns every connected reference its role. It never adds
facts the brief does not state: it is a correct skeleton to edit or to hand to a model, not a rich description.
"""

from __future__ import annotations

import re

from .guides import FL2VA_INSTRUCTION, I2VA_INSTRUCTION, L2VA_INSTRUCTION
from .media import PLATE_AUDIO_SOURCE, Reference
from .modes import Mode, task_label
from .references import canonical_reference_tags

# overall_soundscape takes ambience and physical / non-verbal sounds; speech stays in the description (guide 4.6)
_SOUND_WORDS = re.compile(r"(?i)\b(?:sounds?|noises?|hums?|humming|roars?|footsteps|ambience|ambient|silence|silent|"
                          r"audible|hears?|echo(?:es)?|thunder|crackl\w*|rustl\w*|clatter\w*|bangs?|thuds?)\b")
_MUSIC_WORDS = re.compile(r"(?i)\b(?:music|score|song|soundtrack|melody|orchestra\w*|piano|guitar|beat|drums?)\b")


def _sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?;])\s+|\n+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def _sentence(text: str) -> str:
    text = re.sub(r"\s+", " ", text.strip())
    if not text:
        return text
    text = text[0].upper() + text[1:]
    text = text[:-1] + "." if text[-1] == ";" else text
    return text if text[-1] in ".!?\"'>" else text + "."


def _speaker_ids(text: str) -> str:
    """Guide 4.4: a speaker ID before every <d>...</d>. The template has one speaker unless the brief already
    numbers them."""
    if re.search(r"\(S\d+", text):
        return text
    return re.sub(r"(?i)(?<!\) )<d>", "(S1) <d>", text)


def _soundscape(brief: str) -> str:
    picked = [s for s in _sentences(brief)
              if _SOUND_WORDS.search(s) and not _MUSIC_WORDS.search(s) and "<d>" not in s.lower()]
    if picked:
        return " ".join(_sentence(s) for s in picked)
    return "Ambient sound consistent with the described scene, with no sound events beyond those in the brief."


def _music(brief: str) -> str:
    picked = [s for s in _sentences(brief) if _MUSIC_WORDS.search(s)]
    return " ".join(_sentence(s) for s in picked) if picked else "N/A"


def _role(mode: Mode, ref: Reference) -> str:
    if ref.kind == "audio":
        if ref.source == PLATE_AUDIO_SOURCE:
            return "the synchronized audio track of <Video 1>, reused in the target video"
        return "an audio reference; its role comes only from the brief"
    if ref.kind == "video" and ref.ref == "<Video 1>":
        if mode.key in ("V2VEdit", "Replacement"):
            return "the source plate that the target video edits"
        if mode.key == "Continuation":
            return "the source video that the target video continues"
    if ref.kind == "image" and mode.key == "Replacement":
        return "the new subject's identity and appearance"
    return "a reference for the elements the brief assigns to it"


def _retention(mode: Mode, ref: Reference) -> str:
    if ref.kind == "audio":
        if ref.source == PLATE_AUDIO_SOURCE:
            return f"{ref.ref} is reused as the target video's sound."
        return f"{ref.ref} is referenced in the role the brief gives it."
    if ref.kind == "video" and ref.ref == "<Video 1>" and mode.key in ("V2VEdit", "Replacement"):
        return (f"{ref.ref} keeps its composition, camera, motion, timing, setting and lighting; only what the brief "
                f"changes is edited.")
    if ref.kind == "video" and ref.ref == "<Video 1>" and mode.key == "Continuation":
        return f"{ref.ref} provides the starting state; the target continues from its final moment."
    if ref.kind == "image" and mode.key == "Replacement":
        return f"{ref.ref} is transferred as the new subject's identity and appearance, replacing the original subject."
    return f"{ref.ref} is applied in the role the brief assigns to it."


def write(mode: Mode, brief: str, duration: float, refs: list[Reference]) -> str:
    body = _speaker_ids(_sentence(brief))
    sound = _soundscape(brief)
    music = _music(brief)
    if mode.guide == "base":
        dur = f"{float(duration):.2f}"
        head = {
            "I2VA": I2VA_INSTRUCTION,
            "FL2VA": FL2VA_INSTRUCTION.replace("Shot N", "Shot 1").replace("S.SS", dur),
            "L2VA": L2VA_INSTRUCTION.replace("[Shot N]", "[Shot 1]").replace("S.SS", dur),
        }.get(mode.key)
        lines = [head, ""] if head else []
        lines += [
            f"integrated_multimodal_description: [Shot 1] {body}",
            "",
            f"overall_soundscape: {sound}",
            "",
            f"non_diegetic_music: {music}",
        ]
        return "\n".join(lines)

    plate_audio = any(r.source == PLATE_AUDIO_SOURCE for r in refs)
    brief_audio = bool(canonical_reference_tags(brief, "Audio") & {r.ref for r in refs if r.kind == "audio"})
    label = task_label(mode.key, plate_audio, has_audio_ref=brief_audio)
    edited = " The target video is an edited version of <Video 1>." if mode.key in ("V2VEdit", "Replacement") else ""
    definitions = "\n".join(f"{r.ref} is {_role(mode, r)}." for r in refs)
    retention = "\n".join(_retention(mode, r) for r in refs)
    mentions = ", ".join(r.ref for r in refs if r.kind != "audio")
    n_vis = sum(1 for r in refs if r.kind != "audio")
    role = "its assigned role" if n_vis == 1 else "their assigned roles"
    style = (f"The target video follows the brief and uses {mentions} in {role}." if mentions
             else "The target video follows the brief.")
    return "\n".join([
        f"subject_definitions:\n{definitions}",
        "",
        f"summary: {label}{edited} {body}",
        "",
        f"retention_analysis:\n{retention}",
        "",
        f"detailed_description:\n{style}\n[Shot 1] {body}",
        "",
        f"overall_soundscape: {sound}",
        "",
        f"non_diegetic_music: {music}",
    ])
