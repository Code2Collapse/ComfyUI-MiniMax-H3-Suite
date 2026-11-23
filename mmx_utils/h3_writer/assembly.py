# PORTED FROM: third_party/ComfyUI-MiniMaxH3-Prompt-Writer/backend/assembly.py (MIT, H3 Prompt Writer contributors)
# PORTED FROM: third_party/ComfyUI-MiniMaxH3-Prompt-Writer/backend/h3_pipeline.py (MIT, H3 Prompt Writer contributors)
#   (_messages: the per-reference binding texts)
"""Validate the request and build the model conversation in the node's neutral message format.

Neutral parts: {"type": "text", "text"} | {"type": "image", "mime", "data", "ref"} | {"type": "audio", "mime",
"data", "ref"}. Each provider translates them (providers.py, hf_local.py, gguf_local.py).
"""

from __future__ import annotations

import re
from typing import Any

from .guides import reference_base_excerpt
from .media import PLATE_AUDIO_SOURCE, Reference
from .modes import ASPECT_RATIOS, Mode, WriterError, final_contract
from .references import reference_tags
from .system_prompts import resolve_system_prompt
from .text import normalize_unicode_text


def validate(mode: Mode, brief: str, duration: float, aspect_ratio: str, refs: list[Reference],
             notes: list[str]) -> list[Reference]:
    """Words-only checks. Returns the references the mode uses (base modes drop audio with a note)."""
    if not isinstance(brief, str) or not brief.strip():
        raise WriterError("Write a brief: what happens, who is in it, the look and the sound.")
    if isinstance(duration, bool) or not isinstance(duration, (int, float)) or duration <= 0 or duration > 20:
        raise WriterError("duration_seconds must be between 1 and 20 seconds.")
    if aspect_ratio not in ASPECT_RATIOS:
        raise WriterError(f"aspect_ratio must be one of {', '.join(ASPECT_RATIOS)}.")
    pictures = [r for r in refs if r.kind == "image"]
    videos = [r for r in refs if r.kind == "video"]
    audios = [r for r in refs if r.kind == "audio"]
    if mode.guide == "base":
        if videos:
            raise WriterError(f"{mode.label} takes no video. Disconnect video / video_frames, or pick Reference, "
                              f"V2V edit, Replacement or Continuation.")
        if len(pictures) != mode.pictures:
            need = {0: "no picture", 1: "exactly one picture", 2: "exactly two pictures (first, then last frame)"}
            raise WriterError(f"{mode.label} needs {need[mode.pictures]}; {len(pictures)} connected "
                              f"(every batch item of images, images_2, images_3 counts).")
        if audios:
            notes.append("audio: used only by the reference modes; ignored in " + mode.key)
        return [r for r in refs if r.kind != "audio"]
    if not refs:
        raise WriterError(f"{mode.label} needs at least one reference: connect images, video, video_frames or audio.")
    if mode.needs_video and not videos:
        raise WriterError(f"{mode.label} needs the source video as <Video 1>: connect video or video_frames.")
    if mode.needs_picture and not pictures:
        raise WriterError(f"{mode.label} needs at least one picture of the new subject: connect images.")
    available = {r.ref for r in refs}
    missing_tag = re.search(r"<Missing (?:Picture|Video|Audio) [1-9]\d*>", brief)
    if missing_tag:
        raise WriterError(f"{missing_tag[0]} refers to removed or unavailable media. Replace or remove this tag in "
                          f"the brief.")
    unknown = sorted(reference_tags(brief) - available)
    if unknown:
        raise WriterError(f"{unknown[0]} doesn't exist. Connect that reference or remove the tag from the brief.")
    return refs


def _binding(ref: Reference) -> str:
    if ref.kind == "image":
        return f"{ref.ref}: image reference."
    if ref.kind == "video":
        return (
            f"{ref.ref}: one ordered contact sheet sampled from this same video. "
            "Read frames left-to-right, then top-to-bottom, using the displayed order and the accompanying manifest timestamps to infer motion. "
            f"This sheet is only the internal visual representation of {ref.ref}; it is not a <Picture N> "
            "and must never change or renumber the external reference labels."
        )
    return f"{ref.ref}: audio reference clip."


def build(mode: Mode, brief: str, duration: float, aspect_ratio: str, refs: list[Reference],
          guides: dict[str, str], system_override: str = "", send_visual: bool = True,
          send_audio: bool = False) -> dict[str, Any]:
    """The conversation for one generation. ``guides`` = {"base": text, "reference": text, "*_official": bool}."""
    brief = normalize_unicode_text(brief).strip()
    system_prompt, custom = resolve_system_prompt(mode.key, system_override)
    system_parts = [system_prompt] if system_prompt else []
    if mode.guide == "reference":
        system_parts.append(guides["reference"])
        if guides.get("base_official"):
            excerpt = reference_base_excerpt(guides["base"])
            if excerpt:
                system_parts.append(excerpt)
    else:
        system_parts.append(guides["base"])
    plate_audio = any(r.source == PLATE_AUDIO_SOURCE for r in refs)
    for r in refs:
        r.sent = bool(r.data) and ((r.kind in ("image", "video") and send_visual) or (r.kind == "audio" and send_audio))
    if any(r.sent and r.kind == "audio" for r in refs):
        # ours: the ported wrapper assumes audio is never heard; here the clips are attached, so say so
        system_parts.append("For this request the audio references marked 'attached as audio' ARE attached and "
                            "audible to you: use what is audible in them only in the roles the brief assigns, and "
                            "never invent audio content that is not there.")
    references = "\n".join(r.manifest_line() for r in refs) or "None"
    user_text = (
        f"Mode: {mode.key}\n"
        f"Duration: {duration:g} seconds\n"
        f"Aspect ratio: {aspect_ratio}\n\n"
        "Reference manifest (audio that is not attached is not analyzed by the model; derive its copy/reference role only from the user's words and do not invent its content):\n"
        f"{references}\n\n"
        f"Creative brief:\n{brief}\n\n"
        f"{final_contract(mode.key, brief, plate_audio)}"
    )
    parts: list[dict[str, Any]] = []
    for r in sorted(refs, key=lambda x: {"image": 0, "video": 1, "audio": 2}[x.kind]):
        if not r.sent:
            continue
        parts.append({"type": "text", "text": _binding(r)})
        parts.append({"type": "audio" if r.kind == "audio" else "image", "mime": r.mime, "data": r.data,
                      "ref": r.ref})
    parts.append({"type": "text", "text": user_text})
    return {
        "messages": [
            {"role": "system", "parts": [{"type": "text", "text": "\n\n".join(system_parts)}]},
            {"role": "user", "parts": parts},
        ],
        "user_text": user_text,
        "system_custom": custom,
        "plate_has_audio": plate_audio,
        "input": {
            "mode": mode.label,
            "mode_key": mode.key,
            "duration_seconds": duration,
            "aspect_ratio": aspect_ratio,
            "creative_brief": brief,
            "media_manifest": {"assets": [{"reference": r.ref, "type": r.kind} for r in refs]},
        },
    }
