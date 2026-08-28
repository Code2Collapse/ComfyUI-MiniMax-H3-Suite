"""Pre-flight gate for H3 edit packages — catalog #8."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from .family_presets import (
    LEGAL_TASK_TYPES,
    RETENTION_MARKERS,
    SECTION_ORDER,
    parse_six_sections,
    parse_task_type_prefix,
    section_header,
    subject_labels_from_definitions,
)
from .h3_grid import is_h3_compatible

MASK_FULL_FRAME_EPSILON = 1e-3

RETENTION_LINE_RE = re.compile(
    r":\s*(" + "|".join(re.escape(m) for m in sorted(RETENTION_MARKERS, key=len, reverse=True)) + r")\s*-\s*.+"
)


@dataclass(frozen=True)
class ValidationResult:
    gate: bool
    report: str


def _parse_pipeline_notes(raw: str) -> dict[str, Any]:
    text = (raw or "").strip()
    if not text:
        return {}
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError as exc:
        return {"_parse_error": str(exc)}


def _section_line_positions(prompt: str) -> dict[str, int]:
    positions: dict[str, int] = {}
    for i, line in enumerate((prompt or "").splitlines()):
        stripped = line.strip()
        for name in SECTION_ORDER:
            if stripped == section_header(name) or stripped.startswith(section_header(name)):
                positions[name] = i
    return positions


def _section_order_ok(prompt: str) -> tuple[bool, list[str]]:
    issues: list[str] = []
    positions = _section_line_positions(prompt)
    for name in SECTION_ORDER:
        if name not in positions:
            issues.append(f"Missing section {name!r} — add a line starting with {section_header(name)!r}.")
    ordered = [positions[n] for n in SECTION_ORDER if n in positions]
    if ordered != sorted(ordered):
        issues.append(
            "Sections out of order — use the official order: "
            + ", ".join(SECTION_ORDER) + "."
        )
    return (len(issues) == 0, issues)


def _reject_markdown_headers(prompt: str) -> list[str]:
    if "## " in (prompt or ""):
        return [
            "Prompt uses Markdown `##` section headers — H3 expects bare labels like "
            f"{section_header('subject_definitions')!r} on their own line."
        ]
    return []


def _validate_task_prefix(summary: str) -> list[str]:
    parts, illegal = parse_task_type_prefix(summary)
    issues: list[str] = []
    if illegal == ["<missing>"]:
        issues.append(
            "summary is missing the [task type] prefix — start with a legal combination such as "
            f"[{ ' + '.join(sorted(LEGAL_TASK_TYPES)[:2])} ]."
        )
        return issues
    for part in illegal:
        issues.append(
            f"Illegal task type {part!r} in summary prefix — legal values are: "
            f"{', '.join(sorted(LEGAL_TASK_TYPES))}."
        )
    return issues


def _validate_retention_shape(retention: str) -> list[str]:
    issues: list[str] = []
    lines = [ln.strip() for ln in (retention or "").splitlines() if ln.strip()]
    if not lines:
        issues.append(
            "retention_analysis is empty — add one line per referenced asset using "
            "`<Subject N> (appears in [Shot 1]): fully_preserved - <what is retained>`."
        )
        return issues
    for line in lines:
        if not RETENTION_LINE_RE.search(line):
            issues.append(
                "retention_analysis line does not match the official shape "
                "(label: marker - prose): "
                f"{line!r}"
            )
    return issues


def _mask_mean(region_mask) -> float | None:
    if region_mask is None:
        return None
    import torch

    m = region_mask.float()
    if m.ndim == 3:
        m = m[0]
    if m.ndim != 2:
        return None
    return float(m.mean())


def _parse_drift_report(raw: str | None) -> tuple[float | None, float | None]:
    if not raw or not str(raw).strip():
        return None, None
    text = str(raw).strip()
    threshold = None
    peak = None
    m = re.search(r"threshold[=:]?\s*([0-9.]+)", text, re.I)
    if m:
        threshold = float(m.group(1))
    m = re.search(r"peak[=:]?\s*([0-9.]+)", text, re.I)
    if m:
        peak = float(m.group(1))
    if peak is None:
        try:
            data = json.loads(text)
            if isinstance(data, dict):
                vals = data.get("drift_px") or data.get("drift_per_frame")
                if isinstance(vals, list) and vals:
                    peak = float(max(vals))
        except json.JSONDecodeError:
            pass
    return peak, threshold


def validate_edit_package(
    *,
    prompt: str,
    pipeline_notes: str,
    region_mask=None,
    drift_report: str | None = None,
) -> ValidationResult:
    issues: list[str] = []
    notes = _parse_pipeline_notes(pipeline_notes)
    if "_parse_error" in notes:
        issues.append(
            f"pipeline_notes is not valid JSON — fix the syntax: {notes['_parse_error']}"
        )

    issues.extend(_reject_markdown_headers(prompt))

    if notes.get("mask_required") is True and region_mask is None:
        issues.append(
            "mask_required is true in pipeline_notes but no region_mask is connected — "
            "wire N10 Region Mask into this validator (and the spine) before a whole-frame "
            "environment edit can run."
        )

    mask_mean = _mask_mean(region_mask)
    if notes.get("mask_required") is True and mask_mean is not None and mask_mean >= 1.0 - MASK_FULL_FRAME_EPSILON:
        issues.append(
            "region_mask covers the entire frame (mean≈1.0) — R18 requires a proper subset mask "
            "for environment/element edits, not a whole-frame replace."
        )

    order_ok, order_issues = _section_order_ok(prompt)
    if not order_ok:
        issues.extend(order_issues)

    sections = parse_six_sections(prompt)
    issues.extend(_validate_task_prefix(sections.get("summary", "")))
    issues.extend(_validate_retention_shape(sections.get("retention_analysis", "")))

    labels = subject_labels_from_definitions(sections.get("subject_definitions", ""))
    retention = sections.get("retention_analysis", "")
    for label in labels:
        if label and label not in retention:
            issues.append(
                f"{label} appears in subject_definitions but has no line in "
                "retention_analysis — add a preservation line for every declared subject."
            )

    frame_count = notes.get("frame_count")
    if frame_count is not None:
        try:
            fc = int(frame_count)
            if not is_h3_compatible(fc):
                issues.append(
                    f"frame_count {fc} is off the H3 17n+5 grid — snap with H3 Frame Handles "
                    f"(remainder {fc % 17}, expected 5)."
                )
        except (TypeError, ValueError):
            issues.append(f"frame_count in pipeline_notes must be an integer, got {frame_count!r}.")

    gs = notes.get("guidance_scale")
    if gs is not None:
        try:
            if float(gs) != 1.0:
                issues.append(
                    f"guidance_scale must be 1.0 for distilled H3 sampling (got {gs}) — "
                    "remove CFG / second cond; control strength via masks and hints only."
                )
        except (TypeError, ValueError):
            issues.append(f"guidance_scale must be a number, got {gs!r}.")

    peak, threshold = _parse_drift_report(drift_report)
    if peak is not None and threshold is not None and peak > threshold:
        issues.append(
            f"drift_report peak {peak:.4f}px exceeds threshold {threshold:g}px — "
            "fix tracking or tighten the edit mask before sampling."
        )

    if issues:
        return ValidationResult(gate=False, report="\n".join(f"• {i}" for i in issues))
    return ValidationResult(gate=True, report="All pre-flight checks passed.")
