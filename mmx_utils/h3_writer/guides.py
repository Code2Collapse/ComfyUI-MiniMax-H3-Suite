# PORTED FROM: third_party/ComfyUI-MiniMaxH3-Prompt-Writer/backend/guides.py (MIT, H3 Prompt Writer contributors)
"""The official MiniMax H3 prompt-writing guides: pinned revision, SHA-256 check, shared-rule excerpt.

Changed here: the guide files are NOT shipped with this pack. They are MiniMax "Documentation" under the MiniMax H3
Community License, whose redistribution terms are a doubtful fit for a GPL-3.0 repository (owner decision pending,
docs/PLAN_h3_prompt_writer_node.md). They are read from ``<models>/LLM/h3_prompt_guides/`` when present, fetched there
from the pinned Hugging Face revision when downloads are allowed, and otherwise replaced by a short format contract
written for this pack (BUILTIN_*), which states the structure but does not copy the guide prose.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from typing import Callable

SOURCE_REVISION = "bfc8ed0353f5a9733be73e6b2c98ec0948195b86"
SOURCE_ROOT = f"https://huggingface.co/MiniMaxAI/MiniMax-H3/resolve/{SOURCE_REVISION}/docs"
MAX_GUIDE_BYTES = 200_000
GUIDE_SUBDIR = "h3_prompt_guides"


@dataclass(frozen=True)
class GuideSpec:
    id: str
    title: str
    filename: str
    source_sha256: str

    @property
    def source_url(self) -> str:
        return f"{SOURCE_ROOT}/{self.filename}"


GUIDES = {
    "base": GuideSpec(
        id="base",
        title="MiniMax H3 Video Prompt Writing Guide",
        filename="VIDEO_PROMPT_WRITING_GUIDE_base_en.md",
        source_sha256="2cfebc096a6e08370f288d468d90b60f7f9bcb938f94bf090816e910e48e75fc",
    ),
    "reference": GuideSpec(
        id="reference",
        title="MiniMax H3 Reference Prompt Writing Guide",
        filename="VIDEO_PROMPT_WRITING_GUIDE_ref_en.md",
        source_sha256="1e574f356716ad55612247ffb7bbccbcdb484ad96599d63c7dca1af186b1fab7",
    ),
}

# The three instruction lines are a format requirement of the guide (section 2.1), quoted exactly so a prompt the
# built-in contract produces is the form H3 was trained on.
I2VA_INSTRUCTION = "For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced."
FL2VA_INSTRUCTION = ("How the reference pictures align with the target video — Picture 1 (from Shot 1) aligns with the "
                     "0.00-second mark of the target video; Picture 2 (from Shot N) aligns with the S.SS-second mark of "
                     "the target video.")
L2VA_INSTRUCTION = ("How the reference pictures align with the target video — <Picture 1> (from [Shot N]) aligns with "
                    "the S.SS-second mark of the target video.")

# Written for this pack: the structure the official guides require, not their text.
BUILTIN_BASE = f"""# H3 prompt format (built-in summary; the official guide was not available)

Modes: T2VA builds the whole audiovisual timeline from text. I2VA starts from <Picture 1> as the first frame and
develops forward. FL2VA travels continuously from <Picture 1> (first frame) to <Picture 2> (last frame). L2VA invents
the least preceding state needed and lands on <Picture 1> as the last frame.

Line one (not for T2VA) is the mode's alignment instruction, followed by one blank line:
- I2VA: {I2VA_INSTRUCTION}
- FL2VA: {FL2VA_INSTRUCTION}
- L2VA: {L2VA_INSTRUCTION}
N is the index of the final shot; S.SS is the video duration with exactly two decimals.

Then exactly three fields, in this order:
integrated_multimodal_description: begins with [Shot 1]; visuals, actions, camera, speakers, dialogue, singing and
in-scene sound along the timeline. A new shot starts with [Shot N] and its cut time as MM:SS.mmm.
overall_soundscape: ambience, physical action sounds and non-verbal human sounds over the whole video.
non_diegetic_music: music only the audience hears; N/A when none was asked for.

Camera motion is written as type + amplitude + speed. A speaker gets a stable ID such as (S1) right before each
<d>spoken words</d>; spoken words and visible text stay verbatim in their original language. Everything else is
English. Keep one continuous shot unless the brief asks for cuts.
"""

BUILTIN_REFERENCE = """# H3 full-reference prompt format (built-in summary; the official guide was not available)

Six sections, in this order, each starting with its name and a colon:
subject_definitions: one line per tracked item. Labels: <Subject N> = visible content abstracted from the references
that the target reuses or changes; <Picture N> = a reference image; <Video N> = a reference video (editing source,
continuation start or whole-video structure); <Audio N> = an audio signal that is copied or referenced. A label keeps
one meaning in every section.
summary: one short English paragraph that starts with a square-bracketed task type. Types: keyframe completion,
reference generation, video editing, video continuation, audio reuse, audio reference; several are joined with " + ".
For video editing it continues with "The target video is an edited version of <Video 1>."
retention_analysis: one line per label: preserved, partly preserved, transferred, copied or referenced, and where.
detailed_description: one or two style sentences, then [Shot 1] ... in playback order with composition, subject look
and position, setting, light, actions, camera, sound and where each reference takes effect. Later shots start with
[Shot N] and the cut time as MM:SS.mmm. For generation tasks 350-500 English words; editing tasks scale with the
source.
overall_soundscape: ambience and physical sounds; cite <Audio N> where an audio relationship is active.
non_diegetic_music: music only the audience hears; N/A when none was asked for.

Speakers get a stable ID such as (S1) before each <d>spoken words</d>; spoken words and visible text stay verbatim.
"""

def _normalized(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n").rstrip() + "\n"


def _default_fetch(url: str) -> bytes:
    import urllib.request
    req = urllib.request.Request(url, headers={"User-Agent": "mmx-h3-prompt-writer"})
    with urllib.request.urlopen(req, timeout=20) as r:      # noqa: S310 - fixed https URL at a pinned revision
        data = r.read(MAX_GUIDE_BYTES + 1)
    if len(data) > MAX_GUIDE_BYTES:
        raise ValueError("guide larger than expected")
    return data


def load_guide(guide_id: str, guide_dir: str | None, allow_fetch: bool,
               fetch: Callable[[str], bytes] | None = None, notes: list[str] | None = None) -> tuple[str, str]:
    """(content, source). source: 'official (cached)' | 'official (fetched)' | 'built-in summary'. Never raises."""
    spec = GUIDES[guide_id]
    notes = notes if notes is not None else []
    builtin = BUILTIN_REFERENCE if guide_id == "reference" else BUILTIN_BASE
    path = os.path.join(guide_dir, spec.filename) if guide_dir else None
    if path and os.path.isfile(path):
        try:
            with open(path, encoding="utf-8-sig") as fh:
                content = _normalized(fh.read())
            if hashlib.sha256(content.encode("utf-8")).hexdigest() == spec.source_sha256:
                return content, "official (cached)"
            notes.append(f"{spec.filename} in {guide_dir} does not match the pinned revision (SHA-256); not used")
        except OSError as e:
            notes.append(f"cannot read {path}: {e}")
    if allow_fetch and path:
        try:
            content = _normalized((fetch or _default_fetch)(spec.source_url).decode("utf-8-sig"))
            if hashlib.sha256(content.encode("utf-8")).hexdigest() != spec.source_sha256:
                notes.append(f"fetched {spec.filename} does not match the pinned SHA-256; not used")
            else:
                os.makedirs(guide_dir, exist_ok=True)
                tmp = path + ".part"
                with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
                    fh.write(content)
                os.replace(tmp, path)
                return content, "official (fetched)"
        except Exception as e:  # noqa: BLE001 - offline / blocked / disk: fall back, say why
            notes.append(f"could not fetch {spec.filename} ({type(e).__name__}: {e})")
    elif not allow_fetch:
        notes.append(f"{spec.filename} not cached and downloads are off")
    return builtin, "built-in summary"


def reference_base_excerpt(base_content: str) -> str:
    """Return only the shared base-guide rules referenced by full-reference mode (official content only)."""
    sections = base_content.split("\n### ")
    selected: list[str] = []
    paragraph_limits = {
        "4.2 Shots and Cuts": 1,
        "4.3 Camera Motion: Motion Type + Amplitude + Speed": 1,
        "4.4 Speakers, Dialogue, and Singing": 2,
        "4.5 On-Screen Text": 1,
        "4.6 overall_soundscape": 1,
        "4.7 non_diegetic_music": 1,
    }
    for section in sections:
        title, _, body = section.partition("\n")
        limit = paragraph_limits.get(title.strip())
        if limit is None:
            continue
        paragraphs = [part.strip() for part in body.split("\n\n") if part.strip()]
        prose = [part for part in paragraphs if not part.startswith(("```", "|", "## "))]
        selected.append(f"### {title.strip()}\n\n" + "\n\n".join(prose[:limit]))
    if len(selected) != len(paragraph_limits):
        return ""
    return (
        "# Shared official base-guide rules used by full-reference mode\n\n"
        + "\n\n".join(selected)
        + "\n"
    )
