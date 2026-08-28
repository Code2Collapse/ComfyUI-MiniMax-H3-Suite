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

from mmx_utils.family_presets import FAMILY_IDS, build_family_package, pipeline_notes_json


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
    ):
        blob = f"{family}|{subject_name}|{task_notes}|{style_notes}|{references_manifest}|{frame_count}"
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
    ) -> io.NodeOutput:
        prompt, notes = build_family_package(
            family,
            subject_name=subject_name,
            task_notes=task_notes,
            style_notes=style_notes,
            references_manifest=references_manifest,
            frame_count=int(frame_count),
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
