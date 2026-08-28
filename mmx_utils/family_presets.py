"""H3 family prompt catalog — official six-section shape (#14) + style DNA (#16)."""

from __future__ import annotations

import json
import re
from typing import Any

SECTION_ORDER: tuple[str, ...] = (
    "subject_definitions",
    "summary",
    "retention_analysis",
    "detailed_description",
    "overall_soundscape",
    "non_diegetic_music",
)

LEGAL_TASK_TYPES: frozenset[str] = frozenset(
    {
        "keyframe completion",
        "reference generation",
        "video editing",
        "video continuation",
        "audio reuse",
        "audio reference",
    }
)

FAMILY_IDS: tuple[str, ...] = (
    "lipsync",
    "face_swap",
    "identity_refine",
    "upscale_incontext",
    "restyle",
    "product_swap",
    "wardrobe",
    "environment_weather",
    "element_swap",
    "ui_on_device",
    "dub_reperformance",
)

# Catalog Part 7 (§7.2–§7.12) — legal task-type combinations only.
TASK_TYPE_PREFIXES: dict[str, str] = {
    "lipsync": "video editing + audio reuse",  # §7.2
    "face_swap": "video editing + reference generation",  # §7.3 (identity transfer)
    "identity_refine": "video editing + reference generation",  # §7.6 refine path
    "upscale_incontext": "video editing + reference generation",  # §7.4
    "restyle": "video editing",  # §7.11 colorize / look
    "product_swap": "video editing",  # §7.8 product hero (masked local edit)
    "wardrobe": "video editing",
    "environment_weather": "video editing",
    "element_swap": "video editing",
    "ui_on_device": "video editing",  # §7.9 localised device composite
    "dub_reperformance": "video editing + audio reuse",  # §7.5 reperformance
}

RETENTION_MARKERS: frozenset[str] = frozenset(
    {
        "fully_preserved",
        "partially_preserved",
        "attribute_transfer",
        "weak_reference",
        "fully_copy",
        "reference",
    }
)


def section_header(name: str) -> str:
    return f"{name}:"


def parse_six_sections(prompt: str) -> dict[str, str]:
    """Parse bare `section_name:` labels (official H3 format)."""
    text = (prompt or "").strip()
    found: dict[str, str] = {k: "" for k in SECTION_ORDER}
    if not text:
        return found
    current: str | None = None
    buf: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        matched: str | None = None
        for name in SECTION_ORDER:
            if stripped == section_header(name) or stripped.startswith(section_header(name)):
                matched = name
                break
        if matched is not None:
            if current is not None:
                found[current] = "\n".join(buf).strip()
            current = matched
            buf = []
            rest = stripped[len(section_header(matched)) :].strip()
            if rest:
                buf.append(rest)
        elif current is not None:
            buf.append(line)
    if current is not None:
        found[current] = "\n".join(buf).strip()
    return found


def build_six_section_prompt(sections: dict[str, str]) -> str:
    parts: list[str] = []
    for name in SECTION_ORDER:
        body = sections.get(name, "")
        parts.append(f"{section_header(name)}\n{body}")
    return "\n\n".join(parts)


def parse_task_type_prefix(summary: str) -> tuple[list[str], list[str]]:
    """Return (legal parts, illegal parts) from summary `[type + type]` prefix."""
    text = (summary or "").strip()
    m = re.match(r"^\[([^\]]+)\]", text)
    if not m:
        return [], ["<missing>"]
    inner = m.group(1)
    parts = [p.strip() for p in inner.split("+")]
    illegal = [p for p in parts if p and p not in LEGAL_TASK_TYPES]
    return parts, illegal


def subject_labels_from_definitions(text: str) -> list[str]:
    labels = re.findall(r"<Subject\s+\d+>", text or "")
    if labels:
        return labels
    # Fallback: first line subject binding without angle brackets.
    for line in (text or "").splitlines():
        line = line.strip()
        if line.lower().startswith("subject:"):
            return [line.split(":", 1)[1].strip()]
    return []


def _shot_block(action: str, style_note: str = "") -> str:
    style = f"The target video is {style_note}.\n" if style_note else "The target video matches the live-action plate in lens, lighting, and grade.\n"
    return (
        f"{style}"
        f"[Shot 1] {action}"
    )


def _family_sections(
    family: str,
    *,
    subject_name: str,
    task_notes: str,
    style_notes: str,
) -> dict[str, str]:
    subject = subject_name or "primary subject"
    notes = task_notes.strip()
    style = style_notes.strip()
    prefix = TASK_TYPE_PREFIXES[family]

    if family == "lipsync":
        return {
            "subject_definitions": (
                f"<Subject 1> is {subject} whose face identity comes from <Picture 1> "
                f"and whose body, wardrobe, and scene come from <Video 1>.\n"
                "<Video 1> is the source video for the target video edit.\n"
                "<Audio 1> is the approved vocal stem that drives mouth shape for <Subject 1>."
            ),
            "summary": (
                f"[{prefix}] The target video is an edited version of <Video 1>. "
                "Only the mouth, jaw, and immediately adjacent cheeks of <Subject 1> are regenerated "
                "so they follow <Audio 1>. Every unmasked pixel of <Video 1> remains fully_preserved."
            ),
            "retention_analysis": (
                "<Subject 1> (appears in [Shot 1]): partially_preserved - "
                "face identity and head pose stay; visemes follow <Audio 1>.\n"
                "<Video 1> (all pixels outside the mouth-jaw token mask): fully_preserved - "
                "no camera rewrite, no restyle, no crop offset.\n"
                "<Audio 1>: fully_copy - reused 1:1 as the mouth-driving signal."
            ),
            "detailed_description": _shot_block(
                f"Continuous take from <Video 1>. <Subject 1> performs the original head and body motion "
                f"and speaks in sync with <Audio 1>. {notes}".strip()
            ),
            "overall_soundscape": "Room tone and practicals remain those of <Video 1>; they are not regenerated.",
            "non_diegetic_music": "none — external track to be muxed after generation.",
        }

    if family == "face_swap":
        return {
            "subject_definitions": (
                f"<Subject 1> is the replacement facial identity from <Picture 1>.\n"
                f"<Subject 2> is {subject} as the plate body, wardrobe, hands, and environment from <Video 1>.\n"
                "<Video 1> is the source video for the target video edit."
            ),
            "summary": (
                f"[{prefix}] The target video is an edited version of <Video 1>. "
                "<Subject 1>'s face is transferred onto <Subject 2>'s head. "
                "Body performance and camera come from <Video 1>."
            ),
            "retention_analysis": (
                "<Subject 1> (face only, appears in [Shot 1]): attribute_transfer - "
                "identity onto the plate head; do not restyle wardrobe or set.\n"
                "<Subject 2> (body and scene): fully_preserved - performance and wardrobe retained.\n"
                "<Video 1> (camera, timing): fully_preserved - no camera rewrite."
            ),
            "detailed_description": _shot_block(
                f"Continuous take from <Video 1>. <Subject 2> performs as shot. "
                f"The face is <Subject 1>, lighting matched to the plate key. {notes}".strip()
            ),
            "overall_soundscape": "Copied from <Video 1>.",
            "non_diegetic_music": "none.",
        }

    if family == "identity_refine":
        return {
            "subject_definitions": (
                f"<Subject 1> is {subject} whose identity comes from <Picture 1> and <Video 1>.\n"
                "<Video 1> is the source video for the target video edit."
            ),
            "summary": (
                f"[{prefix}] The target video is an edited version of <Video 1> with micro-detail "
                "refinement on <Subject 1> inside the masked region."
            ),
            "retention_analysis": (
                "<Subject 1> (appears in [Shot 1]): partially_preserved - "
                "identity and pose stay; pores and micro-detail are refined only inside the mask.\n"
                "<Video 1> (camera, timing, exterior pixels): fully_preserved - no camera rewrite."
            ),
            "detailed_description": _shot_block(
                f"<Subject 1> matches the plate performance from <Video 1> with refined skin detail. {notes}".strip(),
                style_note="the same lens and grade as the plate",
            ),
            "overall_soundscape": "Unchanged from <Video 1>.",
            "non_diegetic_music": "none.",
        }

    if family == "upscale_incontext":
        return {
            "subject_definitions": (
                "<Video 1> is the draft to be regenerated at the delivery canvas.\n"
                "<Picture 1> is the original identity reference used to make the draft."
            ),
            "summary": (
                f"[{prefix}] The target video is a higher-resolution in-context regeneration of <Video 1>, "
                "reusing the original multimodal context so fine detail is reconstructed rather than hallucinated."
            ),
            "retention_analysis": (
                "<Video 1> (composition, timing, camera): fully_preserved - no restyle, no new action.\n"
                "<Picture 1>: fully_preserved - identity remains binding."
            ),
            "detailed_description": _shot_block(
                f"Identical action and framing to <Video 1>, sharper optical resolution, stable fine detail. {notes}".strip()
            ),
            "overall_soundscape": "Unchanged from the draft's muxed track.",
            "non_diegetic_music": "none.",
        }

    if family == "restyle":
        desc = f"Palette and materials shift per style DNA. {notes}".strip()
        if style:
            desc = f"Style DNA: {style}\n{desc}"
        return {
            "subject_definitions": (
                f"<Subject 1> is {subject} from <Picture 1>.\n"
                "<Video 1> is the source video for the target video edit.\n"
                f"<Picture 2> is the style DNA still{f' — {style}' if style else ''}."
            ),
            "summary": (
                f"[{prefix}] The target video is an edited version of <Video 1> with a look transfer "
                "from <Picture 2> confined to the masked region."
            ),
            "retention_analysis": (
                "<Subject 1> (appears in [Shot 1]): fully_preserved - identity and performance retained.\n"
                "<Video 1> (camera, timing): fully_preserved - no camera rewrite.\n"
                "<Picture 2> (palette and materials): attribute_transfer - look only inside the mask."
            ),
            "detailed_description": _shot_block(desc),
            "overall_soundscape": "Unchanged from <Video 1>.",
            "non_diegetic_music": "none.",
        }

    if family == "product_swap":
        return {
            "subject_definitions": (
                "<Subject 1> is the physical product from <Picture 1>. Geometry, label text, and finish are locked.\n"
                f"<Subject 2> is {subject} and the scene from <Video 1>."
            ),
            "summary": (
                f"[{prefix}] The target video is an edited version of <Video 1> with <Subject 1> "
                "swapped inside the masked region."
            ),
            "retention_analysis": (
                "<Subject 1> (product, appears in [Shot 1]): attribute_transfer - product geometry inside the mask.\n"
                "<Subject 2> (scene and camera): fully_preserved - camera and environment outside the mask.\n"
                "<Video 1> (timing): fully_preserved - no timing rewrite."
            ),
            "detailed_description": _shot_block(
                f"Static or continuous framing from <Video 1>. <Subject 1> appears in the masked region with letter-perfect label text. {notes}".strip()
            ),
            "overall_soundscape": "Unchanged from <Video 1>.",
            "non_diegetic_music": "none.",
        }

    if family == "wardrobe":
        return {
            "subject_definitions": (
                f"<Subject 1> is {subject} whose identity comes from <Picture 1>.\n"
                "<Picture 2> is the target wardrobe reference.\n"
                "<Video 1> is the source video for the target video edit."
            ),
            "summary": (
                f"[{prefix}] The target video is an edited version of <Video 1> with wardrobe from <Picture 2> "
                "applied to <Subject 1> inside the masked region."
            ),
            "retention_analysis": (
                "<Subject 1> (identity and performance): fully_preserved - face, body motion, and timing retained.\n"
                "<Picture 2> (wardrobe): attribute_transfer - garments inside the mask only.\n"
                "<Video 1> (camera): fully_preserved - no camera rewrite."
            ),
            "detailed_description": _shot_block(
                f"<Subject 1> performs as in <Video 1> wearing the wardrobe from <Picture 2> inside the mask. {notes}".strip()
            ),
            "overall_soundscape": "Unchanged from <Video 1>.",
            "non_diegetic_music": "none.",
        }

    if family == "environment_weather":
        return {
            "subject_definitions": (
                f"<Subject 1> is {subject} whose appearance comes from <Picture 1>.\n"
                "<Subject 2> is the environment from <Video 1>.\n"
                "<Video 1> is the source video for the target video edit."
            ),
            "summary": (
                f"[{prefix}] The target video is an edited version of <Video 1>. "
                "Only the masked environment region is regenerated; <Subject 1> and camera remain unchanged."
            ),
            "retention_analysis": (
                "<Subject 1> (appears in [Shot 1]): fully_preserved - identity, wardrobe, and body motion are retained.\n"
                "<Video 1> (camera and timing): fully_preserved - no camera rewrite outside the masked region.\n"
                "<Subject 2> (environment inside mask): attribute_transfer - weather and set dressing only inside the mask."
            ),
            "detailed_description": _shot_block(
                f"Continuous take from <Video 1>. Environment inside the mask changes per the brief; "
                f"<Subject 1> and camera remain locked. {notes}".strip()
            ),
            "overall_soundscape": "Practicals from <Video 1> unless replaced inside the mask.",
            "non_diegetic_music": "none.",
        }

    if family == "element_swap":
        return {
            "subject_definitions": (
                f"<Subject 1> is {subject} from <Picture 1>.\n"
                "<Picture 2> is the replacement element reference.\n"
                "<Video 1> is the source video for the target video edit."
            ),
            "summary": (
                f"[{prefix}] The target video is an edited version of <Video 1> with a single element "
                "swapped inside the masked region."
            ),
            "retention_analysis": (
                "<Subject 1> (appears in [Shot 1]): fully_preserved - identity and performance outside the swap mask.\n"
                "<Video 1> (camera and timing): fully_preserved - no camera rewrite.\n"
                "<Picture 2> (swap element): attribute_transfer - only inside the masked region."
            ),
            "detailed_description": _shot_block(
                f"Continuous take from <Video 1>. The swapped element from <Picture 2> appears only inside the mask. {notes}".strip()
            ),
            "overall_soundscape": "Unchanged from <Video 1>.",
            "non_diegetic_music": "none.",
        }

    if family == "ui_on_device":
        return {
            "subject_definitions": (
                f"<Subject 1> is the device and hand environment from <Picture 2>.\n"
                "<Picture 1> is the flat UI export — every pixel of UI content is locked.\n"
                f"<Video 1> is the plate performance featuring {subject}."
            ),
            "summary": (
                f"[{prefix}] The target video is an edited version of <Video 1> with a photoreal device mockup. "
                "UI content is never redrawn."
            ),
            "retention_analysis": (
                "<Picture 1> (screen surface): fully_preserved - pixel content, typography, and icons.\n"
                "<Subject 1> (device bezel and hand): partially_preserved - reflection and drift may change inside the mask.\n"
                "<Video 1> (camera and timing): fully_preserved - no camera rewrite outside the mask."
            ),
            "detailed_description": _shot_block(
                f"Screen shows the locked UI from <Picture 1>. Subtle camera drift only inside the masked bezel region. {notes}".strip()
            ),
            "overall_soundscape": "Light UI clicks if requested, else silence.",
            "non_diegetic_music": "none.",
        }

    if family == "dub_reperformance":
        return {
            "subject_definitions": (
                f"<Subject 1> is {subject} whose identity comes from <Picture 1>.\n"
                "<Video 1> is the performance source: words, viseme timing, head motion, camera.\n"
                "<Audio 1> is the synchronized audio of <Video 1> — words and timing source only.\n"
                "<Audio 2> is the target voice timbre for <Subject 1>."
            ),
            "summary": (
                f"[{prefix}] <Subject 1> reperforms the exact words of <Audio 1> in the voice of <Audio 2>, "
                "on the camera and body timing of <Video 1>."
            ),
            "retention_analysis": (
                "<Video 1> (camera, body, timing): fully_preserved - outside the face region.\n"
                "<Audio 1>: reference - words and timing only; do not copy timbre.\n"
                "<Audio 2>: fully_copy - delivered vocal; mouth locked to this signal.\n"
                "<Subject 1> (appears in [Shot 1]): partially_preserved - face performance follows <Audio 2>."
            ),
            "detailed_description": _shot_block(
                f"<Subject 1> delivers the source words with lip sync to <Audio 2>. Mouth closed off-lyric. {notes}".strip()
            ),
            "overall_soundscape": "Practicals from <Video 1> unless replaced.",
            "non_diegetic_music": "as approved.",
        }

    raise ValueError(f"Unknown family {family!r}")


def build_family_package(
    family: str,
    *,
    subject_name: str = "",
    task_notes: str = "",
    style_notes: str = "",
    references_manifest: str = "",
    frame_count: int = 22,
    guidance_scale: float = 1.0,
) -> tuple[str, dict[str, Any]]:
    fam = (family or "lipsync").lower()
    if fam not in FAMILY_IDS:
        raise ValueError(f"Unknown family {family!r}")

    sections = _family_sections(
        fam,
        subject_name=subject_name,
        task_notes=task_notes,
        style_notes=style_notes,
    )
    mask_required = fam in ("environment_weather", "element_swap")

    pipeline_notes: dict[str, Any] = {
        "family": fam,
        "catalog_refs": ["#14", "#16"],
        "task_type_prefix": TASK_TYPE_PREFIXES[fam],
        "mask_required": mask_required,
        "guidance_scale": float(guidance_scale),
        "frame_count": int(frame_count),
        "references_manifest": (references_manifest or "").strip(),
        "subjects": subject_labels_from_definitions(sections["subject_definitions"]),
    }

    prompt = build_six_section_prompt(sections)
    return prompt, pipeline_notes


def pipeline_notes_json(notes: dict[str, Any]) -> str:
    return json.dumps(notes, indent=2, sort_keys=True)
