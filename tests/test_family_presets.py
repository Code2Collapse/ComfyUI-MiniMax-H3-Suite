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


# ── character replacement families (owner 2026-10-13, C2C D8.21b): deterministic, editable templates ───────────

def test_replacement_families_are_deterministic_and_declare_the_plate():
    from mmx_utils.family_presets import REPLACEMENT_FAMILIES
    assert len(REPLACEMENT_FAMILIES) == 6
    for fam in REPLACEMENT_FAMILIES:
        a, notes = build_family_package(fam, subject_name="the actor", task_notes="x")
        b, _ = build_family_package(fam, subject_name="the actor", task_notes="x")
        assert a == b                                                   # same settings, same prompt
        defs = parse_six_sections(a)["subject_definitions"]
        assert "<Video 1> is the source video for the target video edit" in defs
        assert "framing, camera motion, timing and composition" in defs
        assert notes["mask_required"] is True
        assert "fully_preserved" in parse_six_sections(a)["retention_analysis"]


def test_head_size_mode_changes_only_the_head_wording():
    for fam in ("replace_keep_composition", "replace_change_pose", "replace_keep_pose_change_look",
                "replace_masked_subject", "replace_across_video"):
        match, n1 = build_family_package(fam, head_size_mode="match character proportions")
        keep, n2 = build_family_package(fam, head_size_mode="keep original composition")
        assert "head-to-body proportions of <Picture 1>" in match and "never stretched" in match
        assert "size and screen position of the original performer's head" in keep
        assert n1["head_size_mode"] != n2["head_size_mode"]


def test_expression_source_differs_from_identity_replacement():
    expr, _ = build_family_package("expression_transfer")
    swap, _ = build_family_package("replace_keep_composition")
    assert "facial expression, eye and lip movement" in expr and "<Video 2>" in expr
    assert "identity and appearance stay unchanged" in expr          # expression transfer keeps the person
    assert "Pose, expression, lip movement and timing follow <Subject 2>" in swap
    assert "<Video 2>" not in swap


def test_existing_families_unchanged_by_the_head_mode():
    for fam in ("lipsync", "face_swap", "element_swap"):
        assert build_family_package(fam)[0] == build_family_package(fam, head_size_mode="keep original composition")[0]
