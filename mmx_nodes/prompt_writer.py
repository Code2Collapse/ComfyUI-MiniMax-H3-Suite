"""MiniMax H3 Prompt Writer v2 (C2C L8.24, owner A12 s1.5).

Writes the structured H3 prompt for a task from a plain-language brief. The deterministic template (H3 Family
Presets + the 8-field plan) needs no model at all; an LLM backend - Ollama, an OpenAI-compatible server such as
llama-server or LM Studio, or a cloud API - only refines the descriptive fields, always over HTTP, so a broken local
runtime can fail a request but never take ComfyUI down. Every failure falls back to the template unless asked to
stop, and the report says what came from where.
"""

from __future__ import annotations

import hashlib
import logging
import sys
import time
from pathlib import Path

from comfy_api.latest import io

_PKG = Path(__file__).resolve().parents[1]
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from mmx_utils.family_presets import FAMILY_IDS, HEAD_SIZE_MODES, build_family_package, parse_six_sections  # noqa: E402
from mmx_utils.llm_http import (  # noqa: E402
    BACKENDS,
    OFFLINE_ENV,
    TEMPLATE,
    BackendConfig,
    BackendError,
    chat,
    offline,
    preflight,
    scrub,
)
from mmx_utils.prompt_writer import (  # noqa: E402
    PLAN_FIELDS,
    allowed_tags,
    llm_messages,
    merge_llm_fields,
    parse_llm_json,
    serialise,
    template_plan,
)

_LOG = logging.getLogger(__name__)
ON_FAILURE = ("use the template", "stop with an error")


def _interrupt_check():
    try:
        import comfy.model_management as mm
        return mm.throw_exception_if_processing_interrupted
    except Exception:  # noqa: BLE001 - outside ComfyUI (tests) there is nothing to interrupt
        return None


class MiniMaxH3_PromptWriter(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MiniMaxH3_PromptWriter",
            display_name="MiniMax H3 Prompt Writer v2",
            category="MiniMax H3/Prompt",
            description=(
                "Writes the structured H3 prompt for a task from a plain-language brief. 'template (no LLM)' is "
                "deterministic and needs no model; an LLM backend (Ollama, an OpenAI-compatible server such as "
                "llama-server or LM Studio, or a cloud API) refines the descriptive fields over HTTP. Any failure "
                "falls back to the template and the report says why."
            ),
            inputs=[
                io.Combo.Input("family", options=list(FAMILY_IDS), default="replace_keep_composition",
                               tooltip="The H3 task. Its H3 Family Presets template fixes the reference roles "
                                       "(<Video 1> = the plate, <Picture 1> = the character)."),
                io.String.Input("brief", default="", multiline=True,
                                tooltip="What you want, in plain words: look, light, mood, what the character "
                                        "does. Only what the brief says is described - nothing is invented."),
                io.String.Input("subject_name", default="the character"),
                io.Combo.Input("head_size_mode", options=list(HEAD_SIZE_MODES), default=HEAD_SIZE_MODES[0],
                               tooltip="Same as on H3 Subject Crop / H3 Family Presets."),
                io.Boolean.Input("face_closeup_reference", default=False,
                                 tooltip="A face crop of the character is wired as ref_image_1 (<Picture 2>)."),
                io.Combo.Input("backend", options=list(BACKENDS), default=TEMPLATE,
                               tooltip="template (no LLM): deterministic, offline. The others run OUTSIDE ComfyUI "
                                       "and are reached over HTTP - a crashing model server cannot take ComfyUI "
                                       "down. llama.cpp: run llama-server and pick 'openai-compatible server'."),
                io.String.Input("base_url", default="", tooltip="Blank = the backend's default "
                                "(Ollama http://127.0.0.1:11434, llama-server http://127.0.0.1:8080/v1)."),
                io.String.Input("model", default="", tooltip="Blank = the server's first model (Ollama, "
                                "OpenAI-compatible). Cloud APIs need a model name."),
                io.String.Input("api_key_env", default="",
                                tooltip="The NAME of the environment variable that holds the API key (blank = "
                                        "ANTHROPIC_API_KEY / GEMINI_API_KEY / OPENAI_API_KEY). The key itself is "
                                        "never stored in the workflow."),
                io.Int.Input("timeout_s", default=90, min=5, max=600, tooltip="Per request."),
                io.Int.Input("retries", default=1, min=0, max=3,
                             tooltip="Extra attempts after a timeout, 429 or 5xx (1 s, 2 s apart)."),
                io.Float.Input("temperature", default=0.4, min=0.0, max=1.5, step=0.05),
                io.Int.Input("max_tokens", default=1200, min=256, max=8192),
                io.Combo.Input("on_failure", options=list(ON_FAILURE), default=ON_FAILURE[0],
                               tooltip="What to do when the backend fails: use the deterministic template "
                                       "(the report says why) or stop the run with the reason."),
            ],
            outputs=[
                io.String.Output("prompt", display_name="PROMPT"),
                io.String.Output("plan", display_name="PLAN",
                                 tooltip="The 8-field plan as JSON, with where each field came from."),
                io.String.Output("report", display_name="REPORT"),
            ],
        )

    @classmethod
    def fingerprint_inputs(cls, **kw):
        blob = "|".join(f"{k}={kw[k]}" for k in sorted(kw))          # never contains a key: only its variable name
        return hashlib.md5(blob.encode("utf-8")).hexdigest()

    @classmethod
    def execute(cls, family="replace_keep_composition", brief="", subject_name="the character",
                head_size_mode=HEAD_SIZE_MODES[0], face_closeup_reference=False, backend=TEMPLATE, base_url="",
                model="", api_key_env="", timeout_s=90, retries=1, temperature=0.4, max_tokens=1200,
                on_failure=ON_FAILURE[0]) -> io.NodeOutput:
        t0 = time.perf_counter()
        report: list[str] = []
        plan = template_plan(family, brief)
        base, _notes = build_family_package(family, subject_name=subject_name, head_size_mode=head_size_mode,
                                            face_closeup_reference=bool(face_closeup_reference))
        tags = allowed_tags(base)
        used = TEMPLATE
        cfg = BackendConfig(backend, base_url.strip(), model.strip(), api_key_env.strip(), float(timeout_s),
                            float(temperature), int(max_tokens))
        if backend != TEMPLATE and offline():
            report.append(f"{OFFLINE_ENV} is set: no network - template used instead of {backend}")
            backend = TEMPLATE
        if backend != TEMPLATE:
            secret = ""
            try:
                cfg = preflight(cfg)
                secret = cfg.key
                summary = parse_six_sections(base).get("summary", "")
                system, user = llm_messages(plan, brief, subject_name, tags, summary)
                reply = chat(cfg, system, user, retries=int(retries), check_interrupt=_interrupt_check())
                merge_llm_fields(plan, parse_llm_json(reply), tags)
                used = f"{backend} ({cfg.model} @ {cfg.base_url})"
            except BackendError as e:
                why = scrub(e.message, secret, cfg.key)
                if on_failure == ON_FAILURE[1]:
                    raise RuntimeError(f"H3 Prompt Writer: {backend} failed - {why}") from None
                report.append(f"{backend} failed - {why}; the deterministic template was used")
                plan = template_plan(family, brief)
            for line in cfg.log:
                report.append("  " + scrub(line, cfg.key))
            if cfg.api_key_env:
                report.append(f"key: from ${cfg.api_key_env} ({'set' if cfg.key else 'not set'})")
        prompt, problems = serialise(plan, subject_name, head_size_mode, bool(face_closeup_reference))
        ms = (time.perf_counter() - t0) * 1000.0
        from_llm = [k for k in PLAN_FIELDS if plan.source.get(k) == "llm"]
        report.insert(0, f"backend used: {used} - {ms / 1000.0:.2f} s")
        report.append("from the LLM: " + (", ".join(from_llm) if from_llm else "nothing"))
        report.extend("repair: " + r for r in plan.repairs)
        report.extend("problem: " + p for p in problems)
        if backend != TEMPLATE:
            report.append("an LLM reply is not deterministic: change an input to get a new one (same inputs = cached)")
        text = "\n".join(scrub(r, cfg.key) for r in report)
        return io.NodeOutput(prompt, plan.to_json(backend_used=scrub(used, cfg.key)), text)
