# PORTED FROM: third_party/ComfyUI-MiniMaxH3-Prompt-Writer/backend/h3_pipeline.py (MIT, H3 Prompt Writer contributors)
"""Generate, audit, repair once.

Changed here: no thinking pass, no sequence chunks, no context planner (each provider reports its own limits);
the audit and the one repair pass cover the base modes too (this pack's audit_base_prompt); the capability check
tells the user in words which input the model cannot take and the two ways out.
"""

from __future__ import annotations

import time
from typing import Any, Callable

from .audit import audit_prompt, camera_structure_requested
from .media import Reference
from .modes import WriterError, is_reference_mode
from .references import reference_policy, reference_tags
from .text import strip_fences
from .repair import (
    audit_failures,
    dialogue_lines,
    explicit_constraint_violations,
    multimodal_repair_messages,
    narrow_repair_messages,
    unexpected_audio_task,
)

MEDIA_CHOICES = (
    "send connected media (error if the model cannot take it)",
    "images and video only (audio as a label)",
    "labels only (send no media)",
)


def media_plan(provider: Any, refs: list[Reference], choice: str) -> tuple[bool, bool, list[str]]:
    """(send_visual, send_audio, notes). Raises WriterError in words when a connected input cannot be taken."""
    notes: list[str] = []
    visual = [r for r in refs if r.kind in ("image", "video") and r.data]
    audio = [r for r in refs if r.kind == "audio" and r.data]
    if choice == MEDIA_CHOICES[2]:
        if visual or audio:
            notes.append("media: labels only - no picture, video sheet or audio was sent to the model")
        return False, False, notes
    caps = provider.capabilities() if provider is not None else {"images": False, "audio": False}
    name = getattr(provider, "model", "this model")
    send_visual = bool(visual)
    if visual and not caps.get("images"):
        why = provider.why_no_images() + ". " if hasattr(provider, "why_no_images") else ""
        inputs = ", ".join(sorted({r.source.split("[")[0] for r in visual}))
        raise WriterError(f"{name} cannot take images or video sheets (connected: {inputs}). {why}Pick a vision "
                          f"model, or set media_to_model to '{MEDIA_CHOICES[2]}'.")
    send_audio = False
    if audio and choice == MEDIA_CHOICES[0]:
        if not caps.get("audio"):
            raise WriterError(f"{name} cannot take audio (connected: audio). Pick a model that hears audio (for "
                              f"example Gemma 4 E2B / E4B or the Gemini API), or set media_to_model to "
                              f"'{MEDIA_CHOICES[1]}'.")
        send_audio = True
    elif audio:
        notes.append("media: audio sent as a label only; its role comes from the brief")
    return send_visual, send_audio, notes


def _intent(assembled: dict[str, Any]) -> str:
    return "\n".join(str(assembled["input"].get(k, "")) for k in ("creative_brief",) if assembled["input"].get(k))


def audit(prompt: str, assembled: dict[str, Any]) -> dict[str, Any]:
    inp = assembled["input"]
    mode_key = inp["mode_key"]
    intent_text = _intent(assembled)
    result = audit_prompt(prompt, mode_key, inp.get("duration_seconds"), camera_structure_requested(intent_text))
    if is_reference_mode(mode_key):
        policy = reference_policy(inp)
        actual = reference_tags(prompt)
        result["missing_reference_tags"] = sorted(policy.required - actual)
        result["unexpected_reference_tags"] = sorted(actual - policy.allowed)
        result["required_reference_tags"] = sorted(policy.required)
        result["allowed_reference_tags"] = sorted(policy.allowed)
        result["unexpected_audio_task"] = unexpected_audio_task(result.get("task_label"), actual)
        result["explicit_constraint_violations"] = explicit_constraint_violations(intent_text, prompt)
        result["repair_required"] = bool(
            result.get("repair_required")
            or result["missing_reference_tags"]
            or result["unexpected_reference_tags"]
            or result["unexpected_audio_task"]
            or result["explicit_constraint_violations"]
        )
        result["official_format_pass"] = bool(result.get("official_format_pass")) and not result["repair_required"]
    return result


def run(provider: Any, assembled: dict[str, Any], *, repair: bool = True, seed: int | None = None,
        temperature: float = 1.0, max_tokens: int = 2048,
        check_interrupt: Callable[[], None] | None = None) -> dict[str, Any]:
    messages = assembled["messages"]
    t0 = time.perf_counter()
    response = provider.complete(messages, temperature=temperature, top_p=0.95, max_tokens=max_tokens, seed=seed,
                                 check_interrupt=check_interrupt)
    usage = dict(response.get("usage") or {})
    if response.get("finish_reason") == "length":
        raise WriterError(f"The model reached max_tokens ({max_tokens}) before finishing the prompt; raise max_tokens.")
    prompt = strip_fences(response.get("text") or "")
    if not prompt:
        raise WriterError("The model returned no prompt text.")
    result = audit(prompt, assembled)
    out: dict[str, Any] = {"repair_attempted": False, "repair_applied": False, "repair_failure": None,
                           "repair_method": None, "repair_reason": None}
    if repair and result.get("repair_required"):
        inp = assembled["input"]
        policy = reference_policy(inp)
        expected, allowed = policy.required, policy.allowed
        repair_tags = (reference_tags(prompt) & allowed) | expected
        failed = audit_failures(result)
        out["repair_attempted"] = True
        out["repair_reason"] = ", ".join(failed) or "official format audit"
        has_visual = any(p["type"] == "image" for m in messages for p in m["parts"])
        if result.get("missing_reference_tags") and has_visual:
            out["repair_method"] = "multimodal reference correction"
            repair_messages = multimodal_repair_messages(messages, prompt, failed, repair_tags,
                                                         inp.get("duration_seconds"), allowed)
        else:
            out["repair_method"] = "narrow text correction"
            repair_messages = narrow_repair_messages(assembled["user_text"], prompt, failed, repair_tags,
                                                     inp.get("duration_seconds"), allowed)
        if check_interrupt is not None:
            check_interrupt()
        fixed = provider.complete(repair_messages, temperature=0.3, top_p=0.9, max_tokens=max_tokens, seed=seed,
                                  check_interrupt=check_interrupt)
        repaired = strip_fences(fixed.get("text") or "")
        fixed_audit = audit(repaired, assembled) if repaired else {"repair_required": True}
        ru = fixed.get("usage") or {}
        usage["prompt_tokens"] = int(usage.get("prompt_tokens", 0)) + int(ru.get("prompt_tokens", 0))
        usage["completion_tokens"] = int(usage.get("completion_tokens", 0)) + int(ru.get("completion_tokens", 0))
        tags_match = (reference_tags(repaired) == repair_tags) if is_reference_mode(inp["mode_key"]) else True
        if (repaired and fixed.get("finish_reason") != "length" and fixed_audit.get("repair_required") is False
                and tags_match and dialogue_lines(repaired) == dialogue_lines(prompt)):
            prompt, result = repaired, fixed_audit
            out["repair_applied"] = True
        elif not repaired:
            out["repair_failure"] = "empty repair"
        elif fixed.get("finish_reason") == "length":
            out["repair_failure"] = "repair reached its output limit"
        elif fixed_audit.get("repair_required"):
            out["repair_failure"] = "repaired draft still failed: " + ", ".join(audit_failures(fixed_audit))
        else:
            out["repair_failure"] = "correction changed the reference inventory or user dialogue"
    out.update(prompt=prompt, audit=result, usage=usage, seconds=round(time.perf_counter() - t0, 3))
    return out
