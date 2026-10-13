"""N12 — Family presets: six-section H3 prompt (#14) + style DNA (#16)."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

from comfy_api.latest import io

_PKG = Path(__file__).resolve().parents[1]
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from mmx_utils.family_presets import FAMILY_IDS, HEAD_SIZE_MODES, build_family_package, pipeline_notes_json


class MiniMaxH3_FamilyPresets(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MiniMaxH3_FamilyPresets",
            display_name="H3 Family Presets",
            category="MiniMax H3/Authoring",
            description=(
                "Emit catalog #14 six-section prompt shape and #16 style DNA fields. "
                "environment_weather and element_swap set mask_required in pipeline_notes."
            ),
            inputs=[
                io.Combo.Input("family", options=list(FAMILY_IDS), default="lipsync"),
                io.String.Input("subject_name", default="primary subject", multiline=False),
                io.String.Input("task_notes", default="", multiline=True),
                io.String.Input("style_notes", default="", multiline=True, tooltip="Style DNA (#16)."),
                io.String.Input("references_manifest", default="", multiline=True),
                io.Int.Input("frame_count", default=22, min=5, max=512),
                # C2C D8.21: appended; only the six replace_* / expression families read it
                io.Combo.Input("head_size_mode", options=list(HEAD_SIZE_MODES), default=HEAD_SIZE_MODES[0],
                               optional=True,
                               tooltip="Character replacement families: how big the new head is - match character proportions (the reference's own head-to-body ratio) or keep original composition (the original head's size and position). Set the same mode on H3 Subject Crop."),
                # C2C L8.22 BUILD (prompt helper): appended, default off = the prompt is unchanged
                io.Boolean.Input("face_closeup_reference", default=False, optional=True,
                                 tooltip="Character replacement families: you also wire a tight face crop of the character as ref_image_1 (e.g. H3 Face Track + Crop on the character image). The prompt then names it <Picture 2>, the authority for facial identity. Ignored by expression_transfer. Experimental: not A/B-tested on H3 weights."),
            ],
            outputs=[
                io.String.Output("prompt"),
                io.String.Output("pipeline_notes"),
                io.String.Output("report"),
            ],
        )

    @classmethod
    def fingerprint_inputs(
        cls,
        family="lipsync",
        subject_name="primary subject",
        task_notes="",
        style_notes="",
        references_manifest="",
        frame_count=22,
        head_size_mode=HEAD_SIZE_MODES[0],
        face_closeup_reference=False,
    ):
        blob = (f"{family}|{subject_name}|{task_notes}|{style_notes}|{references_manifest}|{frame_count}|"
                f"{head_size_mode}|{bool(face_closeup_reference)}")
        return hashlib.md5(blob.encode()).hexdigest()

    @classmethod
    def execute(
        cls,
        family="lipsync",
        subject_name="primary subject",
        task_notes="",
        style_notes="",
        references_manifest="",
        frame_count=22,
        head_size_mode=HEAD_SIZE_MODES[0],
        face_closeup_reference=False,
    ) -> io.NodeOutput:
        prompt, notes = build_family_package(
            family,
            subject_name=subject_name,
            task_notes=task_notes,
            style_notes=style_notes,
            references_manifest=references_manifest,
            frame_count=int(frame_count),
            head_size_mode=head_size_mode or HEAD_SIZE_MODES[0],
            face_closeup_reference=bool(face_closeup_reference),
            # PINNED, never a widget. H3 is guidance-distilled: anything above 1.0
            # applies guidance twice and degrades output. A 0-4 slider here was the
            # exact trap N13 EditValidator exists to refuse, offered by the node that
            # authors the notes N13 reads.
            guidance_scale=1.0,
        )
        notes_str = pipeline_notes_json(notes)
        report = (
            f"family={notes['family']} catalog=#14,#16 "
            f"mask_required={notes['mask_required']} "
            f"guidance_scale={notes['guidance_scale']} frame_count={notes['frame_count']}"
        )
        return io.NodeOutput(prompt, notes_str, report)
