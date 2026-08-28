import json
import re

from mmx_utils.family_presets import (
    FAMILY_IDS,
    LEGAL_TASK_TYPES,
    SECTION_ORDER,
    TASK_TYPE_PREFIXES,
    build_family_package,
    parse_six_sections,
    parse_task_type_prefix,
    section_header,
)


def test_all_families_emit_six_sections_in_order():
    for fam in FAMILY_IDS:
        prompt, notes = build_family_package(fam, subject_name="hero", task_notes="test")
        for i, name in enumerate(SECTION_ORDER):
            assert section_header(name) in prompt
            assert f"## {name}" not in prompt
            if i > 0:
                assert prompt.index(section_header(SECTION_ORDER[i - 1])) < prompt.index(section_header(name))
        sections = parse_six_sections(prompt)
        assert sections["summary"].startswith("[")


def test_all_families_use_legal_task_type_prefix():
    for fam in FAMILY_IDS:
        prompt, notes = build_family_package(fam)
        parts, illegal = parse_task_type_prefix(parse_six_sections(prompt)["summary"])
        assert not illegal, f"{fam}: illegal task types {illegal}"
        for part in parts:
            assert part in LEGAL_TASK_TYPES
        assert notes["task_type_prefix"] == TASK_TYPE_PREFIXES[fam]


def test_official_subject_bindings_present():
    prompt, _ = build_family_package("face_swap", subject_name="hero")
    assert "<Subject 1>" in prompt
    assert "<Picture 1>" in prompt
    assert "<Video 1>" in prompt


def test_official_retention_shape():
    prompt, _ = build_family_package("environment_weather", subject_name="hero")
    retention = parse_six_sections(prompt)["retention_analysis"]
    assert re.search(r":\s*fully_preserved\s*-\s*.+", retention)
    assert "[Shot 1]" in retention


def test_detailed_description_has_shot_marker():
    prompt, _ = build_family_package("lipsync")
    detailed = parse_six_sections(prompt)["detailed_description"]
    assert "[Shot 1]" in detailed


def test_environment_and_element_swap_require_mask():
    for fam in ("environment_weather", "element_swap"):
        _prompt, notes = build_family_package(fam)
        assert notes["mask_required"] is True
        prompt, _ = build_family_package(fam)
        retention = parse_six_sections(prompt)["retention_analysis"]
        assert "fully_preserved -" in retention
        assert "camera:" not in retention


def test_pipeline_notes_json_roundtrip():
    _prompt, notes = build_family_package("lipsync", guidance_scale=1.0, frame_count=22)
    data = json.loads(json.dumps(notes))
    assert data["catalog_refs"] == ["#14", "#16"]
    assert data["guidance_scale"] == 1.0
