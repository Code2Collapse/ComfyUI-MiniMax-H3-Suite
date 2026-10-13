"""H3 Prompt Writer v2 (C2C L8.24, owner A12 s1.5): the structured plan, its template, the validator and the H3
six-section serialiser.

Nothing here (or in mmx_utils/llm_http.py) imports a native inference runtime. The llama.cpp-based writer this
replaces dies with an illegal instruction INSIDE the ComfyUI process when the prebuilt wheel uses AVX-512 on a CPU
without it (duckyshell/ComfyUI-MiniMaxH3-Prompt-Writer #13, #22); a signal cannot be caught, so the cure is that
models are only ever reached over HTTP. Design: C2C docs/research/l824_prompt_writer_v2.md.

The tag bindings (<Video 1> = the plate, <Picture 1> = the character, the [task type] prefix) always come from the
H3 Family Presets template, so an LLM can describe but never rebind or invent a reference.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from .family_presets import (
    FAMILY_IDS,
    HEAD_SIZE_MODES,
    REPLACEMENT_FAMILIES,
    build_family_package,
    build_six_section_prompt,
    parse_six_sections,
    parse_task_type_prefix,
)

PLAN_FIELDS: tuple[str, ...] = (
    "scene_style",
    "performance_to_preserve",
    "identity_to_transfer",
    "face_motion",
    "body_camera",
    "mask_region",
    "temporal_constraints",
    "negative_constraints",
)

FIELD_HELP: dict[str, str] = {
    "scene_style": "scene and visual style of the target video (lens, light, grade, mood), from the brief only",
    "performance_to_preserve": "what of the source performance must stay exactly as it is",
    "identity_to_transfer": "the replacement character's identity and appearance: face, hair, skin, costume, colours",
    "face_motion": "facial expressions, mouth and lip movement, gaze and head motion",
    "body_camera": "body movement and camera behaviour",
    "mask_region": "which region changes and which stays untouched",
    "temporal_constraints": "consistency across frames and chunks",
    "negative_constraints": "what must not happen (identity drift, frozen expressions, changed lip motion, flicker, "
                            "background changes), written as rules",
}

MAX_FIELD_CHARS = 1200
MAX_DESCRIPTION_WORDS = 500          # H3 reference guide: 350-500 words of detailed_description
TAG_RE = re.compile(r"<(?:Subject|Picture|Video|Audio)\s+\d+>")

# Defaults that carry the owner's performance requirement (A12 s1.1) into every replacement prompt.
_PERFORMANCE = {
    "replace_change_pose": "Body pose, gestures and timing come from <Video 2>; framing and camera stay those of "
                           "<Video 1>.",
    "expression_transfer": "Facial expressions, blinks, mouth shapes, lip and jaw movement and gaze come from "
                           "<Video 2>; head pose, body and timing stay those of <Video 1>.",
}
_PERFORMANCE_DEFAULT = ("The performance is the original performer's in <Video 1>: facial expressions, blinks, mouth "
                        "shapes, lip and jaw movement, head motion, gaze, body pose, gestures and timing stay "
                        "frame-accurate.")
_MASK = "Only the masked region changes; everything outside it is the original plate."
_TEMPORAL = "One constant identity and appearance in every frame; no drift or flicker between frames."
_NEGATIVE = ("No identity drift, no frozen or exaggerated expressions, no changed lip movement or timing, no new "
             "camera motion, no background change.")


@dataclass
class PromptPlan:
    family: str
    fields: dict[str, str]
    source: dict[str, str] = field(default_factory=dict)       # field -> "template" | "llm" | "brief"
    repairs: list[str] = field(default_factory=list)

    def to_json(self, **extra: Any) -> str:
        return json.dumps({"family": self.family, "fields": self.fields, "source": self.source,
                           "repairs": self.repairs, **extra}, indent=2, sort_keys=True)


def _base(family: str, subject_name: str, head_size_mode: str, face_closeup_reference: bool):
    return build_family_package(family, subject_name=subject_name, head_size_mode=head_size_mode,
                                face_closeup_reference=face_closeup_reference)


def template_plan(family: str, brief: str = "") -> PromptPlan:
    """The deterministic plan: same inputs, same plan, no model and no network."""
    fam = (family or "").lower()
    if fam not in FAMILY_IDS:
        raise ValueError(f"Unknown family {family!r}")
    replacement = fam in REPLACEMENT_FAMILIES
    fields = {k: "" for k in PLAN_FIELDS}
    fields["scene_style"] = " ".join((brief or "").split())[:MAX_FIELD_CHARS]
    if replacement:
        fields["performance_to_preserve"] = _PERFORMANCE.get(fam, _PERFORMANCE_DEFAULT)
        fields["mask_region"] = _MASK
        fields["temporal_constraints"] = _TEMPORAL
        fields["negative_constraints"] = _NEGATIVE
    source = {k: ("brief" if k == "scene_style" and fields[k] else "template") for k in PLAN_FIELDS}
    return PromptPlan(fam, fields, source)


def allowed_tags(base_prompt: str) -> set[str]:
    return set(TAG_RE.findall(base_prompt))


def _strip_reasoning(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S | re.I)
    if "</think>" in text:                                   # reasoning without an opening tag
        text = text.rsplit("</think>", 1)[-1]
    return text.strip()


def parse_llm_json(text: str) -> dict[str, Any] | None:
    """The first JSON object in a reply, with reasoning and code fences removed; None when there is none."""
    body = _strip_reasoning(text)
    body = re.sub(r"^```(?:json)?\s*|\s*```$", "", body.strip(), flags=re.I)
    start = body.find("{")
    while start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(body)):
            ch = body[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            elif ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        val = json.loads(body[start:i + 1])
                    except ValueError:
                        break
                    return val if isinstance(val, dict) else None
        start = body.find("{", start + 1)
    return None


def merge_llm_fields(plan: PromptPlan, reply: dict[str, Any] | None, tags: set[str]) -> PromptPlan:
    """Take each field the LLM wrote when it is valid; otherwise keep the template's and say why."""
    if reply is None:
        plan.repairs.append("the reply had no JSON object - every field kept from the template")
        return plan
    for key in PLAN_FIELDS:
        if key not in reply:
            plan.repairs.append(f"{key}: missing - template kept")
            continue
        val = reply[key]
        if isinstance(val, list):
            val = " ".join(str(v) for v in val)
        if not isinstance(val, str):
            plan.repairs.append(f"{key}: not text - template kept")
            continue
        val = " ".join(val.split())
        unknown = sorted(set(TAG_RE.findall(val)) - tags)
        if unknown:
            plan.repairs.append(f"{key}: names {', '.join(unknown)}, which no input provides - template kept")
            continue
        if len(val) > MAX_FIELD_CHARS:
            val = _trim_sentences(val, MAX_FIELD_CHARS)
            plan.repairs.append(f"{key}: longer than {MAX_FIELD_CHARS} characters - trimmed")
        if not val:                                   # "nothing to add": the template's value stays
            continue
        plan.fields[key] = val
        plan.source[key] = "llm"
    return plan


def _trim_sentences(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    end = max(cut.rfind(". "), cut.rfind("! "), cut.rfind("? "))
    return cut[:end + 1] if end > 0 else cut.rstrip()


def _trim_words(text: str, max_words: int) -> tuple[str, bool]:
    words = text.split()
    if len(words) <= max_words:
        return text, False
    kept = " ".join(words[:max_words])
    end = max(kept.rfind(". "), kept.rfind(".\n"))
    return (kept[:end + 1] if end > 0 else kept), True


def _sentence(text: str) -> str:
    text = text.strip()
    return text if not text or text[-1] in ".!?" else text + "."


def serialise(plan: PromptPlan, subject_name: str = "", head_size_mode: str = HEAD_SIZE_MODES[0],
              face_closeup_reference: bool = False) -> tuple[str, list[str]]:
    """The plan as H3's six sections, on top of the family's template. Returns (prompt, problems)."""
    base, _notes = _base(plan.family, subject_name, head_size_mode, face_closeup_reference)
    s = parse_six_sections(base)
    f = plan.fields
    if f["identity_to_transfer"]:
        s["subject_definitions"] = s["subject_definitions"].rstrip() + "\n" + _sentence(
            f"Appearance of <Subject 1>: {f['identity_to_transfer']}")
    if f["performance_to_preserve"]:
        s["summary"] = s["summary"].rstrip() + " " + _sentence(f["performance_to_preserve"])
    rules = [_sentence(f[k]) for k in ("mask_region", "temporal_constraints") if f[k]]
    if f["negative_constraints"]:
        rules.append(_sentence("Must not happen: " + f["negative_constraints"].rstrip(".")))
    if rules:
        s["retention_analysis"] = s["retention_analysis"].rstrip() + "\n" + "\n".join(rules)
    desc = s["detailed_description"]
    if f["scene_style"]:
        lines = desc.split("\n")
        if lines and lines[0].startswith("The target video "):
            lines[0] = _sentence(f"The target video: {f['scene_style']}")
        else:
            lines.insert(0, _sentence(f"The target video: {f['scene_style']}"))
        desc = "\n".join(lines)
    extra = " ".join(_sentence(f[k]) for k in ("body_camera", "face_motion") if f[k])
    if extra:
        desc = desc.rstrip() + " " + extra
    desc, trimmed = _trim_words(desc, MAX_DESCRIPTION_WORDS)
    s["detailed_description"] = desc
    problems = []
    if trimmed:
        plan.repairs.append(f"detailed_description: over {MAX_DESCRIPTION_WORDS} words - trimmed at a sentence")
    prompt = build_six_section_prompt(s)
    back = parse_six_sections(prompt)
    missing = [k for k, v in back.items() if not (v or "").strip()]
    if missing:
        problems.append("empty sections: " + ", ".join(missing))
    _parts, illegal = parse_task_type_prefix(back.get("summary", ""))
    if illegal:
        problems.append("summary task-type prefix not legal: " + ", ".join(illegal))
    return prompt, problems


def llm_messages(plan: PromptPlan, brief: str, subject_name: str, tags: set[str], task_summary: str) -> tuple[str, str]:
    """(system, user) for the backend. The model fills the plan from the brief; it never sees a way to rebind tags."""
    system = (
        "You write the structured plan for a MiniMax H3 video prompt. Return ONE JSON object and nothing else, with "
        "exactly these string keys: " + ", ".join(PLAN_FIELDS) + ". "
        "Use only what the brief supports: never invent actions, expressions, props, locations, cuts or camera moves. "
        "<Video 1> is always the source video: it fixes framing, camera motion, timing and the performance unless "
        "the task says otherwise. Refer to inputs only with these tags: " + (", ".join(sorted(tags)) or "none") + ". "
        "Keep every field short and concrete; write constraints as rules. An empty string means 'nothing to add'."
    )
    user = json.dumps({
        "task": task_summary,
        "subject_name": subject_name,
        "brief": brief,
        "field_meaning": FIELD_HELP,
        "baseline_plan": plan.fields,
    }, ensure_ascii=False, indent=1)
    return system, user
