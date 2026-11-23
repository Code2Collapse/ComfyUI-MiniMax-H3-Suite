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
                io.Combo.Input("family", options=list(FAMILY_IDS), default="lipsync",
                               tooltip="Edit task whose six-section prompt template is used (lipsync, face_swap, "
                                       "restyle, ... and the replace_* character families). environment_weather, "
                                       "element_swap and the replacement families set mask_required."),
                io.String.Input("subject_name", default="primary subject", multiline=False,
                                tooltip="Words that describe <Subject 1> in the prompt (empty = 'primary subject')."),
                io.String.Input("task_notes", default="", multiline=True,
                                tooltip="Free text appended to the family's task sentence in the prompt (stripped; "
                                        "empty adds nothing)."),
                io.String.Input("style_notes", default="", multiline=True, tooltip="Style DNA (#16)."),
                io.String.Input("references_manifest", default="", multiline=True,
                                tooltip="Free-text list of the reference assets; not used in the prompt, only copied "
                                        "(stripped) into pipeline_notes JSON as 'references_manifest'.",
                                extra_dict={'c2c_section': 'References'}),
                io.Int.Input("frame_count", default=22, min=5, max=512,
                             tooltip="Clip length in frames, written to pipeline_notes only (not used in the "
                                     "prompt); Edit Validator flags it when it is off H3's 17k+5 grid."),
                # C2C D8.21: appended; only the six replace_* / expression families read it
                io.Combo.Input("head_size_mode", options=list(HEAD_SIZE_MODES), default=HEAD_SIZE_MODES[0],
                               optional=True,
                               tooltip="Character replacement families: how big the new head is - match character proportions (the reference's own head-to-body ratio) or keep original composition (the original head's size and position). Set the same mode on H3 Subject Crop.", extra_dict={'c2c_show_if': {'family': ['replace_keep_composition', 'replace_change_pose', 'replace_keep_pose_change_look', 'expression_transfer', 'replace_masked_subject', 'replace_across_video']}}),
                # C2C L8.22 BUILD (prompt helper): appended, default off = the prompt is unchanged
                io.Boolean.Input("face_closeup_reference", default=False, optional=True,
                                 tooltip="Character replacement families: you also wire a tight face crop of the character as ref_image_1 (e.g. H3 Face Track + Crop on the character image). The prompt then names it <Picture 2>, the authority for facial identity. Ignored by expression_transfer. Experimental: not A/B-tested on H3 weights.", extra_dict={'c2c_show_if': {'family': ['replace_keep_composition', 'replace_change_pose', 'replace_keep_pose_change_look', 'expression_transfer', 'replace_masked_subject', 'replace_across_video']}}),
            ],
            outputs=[
                io.String.Output("prompt", tooltip="The six-section H3 prompt text (section headers + bodies) for "
                                               "the text encoder."),
                io.String.Output("pipeline_notes", tooltip="JSON: family, task-type prefix, mask_required, "
                                                       "guidance_scale (pinned 1.0), frame_count, references_"
                                                       "manifest, subject labels (+ head size / face close-up for "
                                                       "replacement families); read by Edit Validator."),
                io.String.Output("report", tooltip="One line: family, mask_required, guidance_scale and "
                                               "frame_count."),
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
