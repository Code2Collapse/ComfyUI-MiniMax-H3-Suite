import json

import torch

from mmx_utils.edit_validator import validate_edit_package
from mmx_utils.family_presets import build_family_package, pipeline_notes_json


def _valid_env_package():
    prompt, notes = build_family_package("environment_weather", subject_name="hero")
    return prompt, pipeline_notes_json(notes)


def _partial_mask():
    m = torch.zeros(8, 8)
    m[2:6, 2:6] = 1.0
    return m


def test_refuse_mask_required_without_region_mask():
    prompt, notes = _valid_env_package()
    result = validate_edit_package(prompt=prompt, pipeline_notes=notes, region_mask=None)
    assert result.gate is False
    assert "region_mask" in result.report.lower()


def test_refuse_full_frame_mask():
    prompt, notes = _valid_env_package()
    result = validate_edit_package(prompt=prompt, pipeline_notes=notes, region_mask=torch.ones(8, 8))
    assert result.gate is False
    assert "whole-frame" in result.report.lower() or "entire frame" in result.report.lower()


def test_refuse_missing_section():
    prompt, notes = _valid_env_package()
    bad = prompt.split("detailed_description:")[0]
    result = validate_edit_package(prompt=bad, pipeline_notes=notes, region_mask=_partial_mask())
    assert result.gate is False
    assert "detailed_description" in result.report


def test_refuse_markdown_headers():
    prompt, notes = _valid_env_package()
    bad = prompt.replace("subject_definitions:", "## subject_definitions")
    result = validate_edit_package(prompt=bad, pipeline_notes=notes, region_mask=_partial_mask())
    assert result.gate is False
    assert "Markdown" in result.report or "##" in result.report


def test_refuse_illegal_task_type_prefix():
    prompt, notes = _valid_env_package()
    bad = prompt.replace(
        "[video editing]",
        "[video editing + invented_task]",
        1,
    )
    result = validate_edit_package(prompt=bad, pipeline_notes=notes, region_mask=_partial_mask())
    assert result.gate is False
    assert "Illegal task type" in result.report


def test_refuse_subject_without_retention_line():
    """A subject declared in subject_definitions must have a retention line.

    The rename has to be scoped to the subject_definitions SECTION. A global
    prompt.replace() renames the token in retention_analysis too, so the subject
    stays covered, the validator correctly passes, and the test proves nothing —
    which is exactly how this test failed before: it was asserting a refusal that
    should not happen.
    """
    prompt, notes = _valid_env_package()
    head, sep, tail = prompt.partition("retention_analysis:")
    assert sep, "fixture prompt is missing the retention_analysis section"
    # orphan <Subject 9> in definitions only; retention still talks about <Subject 1>
    orphaned = head.replace("<Subject 1>", "<Subject 9>") + sep + tail
    assert "<Subject 9>" not in tail, "rename leaked into retention_analysis"

    result = validate_edit_package(
        prompt=orphaned, pipeline_notes=notes, region_mask=_partial_mask()
    )
    assert result.gate is False, result.report
    assert "Subject 9" in result.report

    # and the complementary case: fully covered subjects must PASS, so the rule
    # is not simply refusing everything.
    ok = validate_edit_package(
        prompt=prompt, pipeline_notes=notes, region_mask=_partial_mask()
    )
    assert ok.gate is True, ok.report


def test_refuse_off_grid_frame_count():
    prompt, notes = _valid_env_package()
    data = json.loads(notes)
    data["frame_count"] = 20
    result = validate_edit_package(prompt=prompt, pipeline_notes=json.dumps(data), region_mask=_partial_mask())
    assert result.gate is False
    assert "17n+5" in result.report


def test_refuse_guidance_scale_not_one():
    prompt, notes = _valid_env_package()
    data = json.loads(notes)
    data["guidance_scale"] = 1.5
    result = validate_edit_package(prompt=prompt, pipeline_notes=json.dumps(data), region_mask=_partial_mask())
    assert result.gate is False
    assert "guidance_scale" in result.report


def test_refuse_drift_above_threshold():
    prompt, notes = _valid_env_package()
    drift = "exterior drift peak=5.0px threshold=2.0px frames=1 — FAIL"
    result = validate_edit_package(
        prompt=prompt,
        pipeline_notes=notes,
        region_mask=_partial_mask(),
        drift_report=drift,
    )
    assert result.gate is False
    assert "drift" in result.report.lower()


def test_passes_when_partial_mask_connected():
    prompt, notes = _valid_env_package()
    result = validate_edit_package(prompt=prompt, pipeline_notes=notes, region_mask=_partial_mask())
    assert result.gate is True
