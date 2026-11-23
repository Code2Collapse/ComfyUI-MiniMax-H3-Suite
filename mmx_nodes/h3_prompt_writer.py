"""H3 Prompt Writer (owner order 2026-11-23; design docs/PLAN_h3_prompt_writer_node.md).

The H3 Prompt Writer's prompt logic (third_party/ComfyUI-MiniMaxH3-Prompt-Writer, MIT, H3 Prompt Writer contributors)
as a workflow node: pictures, a video or an image sequence and audio come in as sockets, the mode picks the official
H3 prompt structure, the provider is any of a deterministic template, local Hugging Face transformers (any model id),
local GGUF, Ollama, an OpenAI-compatible server, or the OpenAI / Anthropic / Gemini APIs. The written prompt is audited
against the official format and repaired once when an objective check fails. No floating window.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from comfy_api.latest import io

_PKG = Path(__file__).resolve().parents[1]
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from mmx_utils.h3_writer import assembly, catalog, guides, media, pipeline, template
from mmx_utils.h3_writer import downloads as dl  # noqa: E402
from mmx_utils.h3_writer.hf_local import DEVICES, DTYPES, QUANTIZATIONS  # noqa: E402
from mmx_utils.h3_writer.modes import ASPECT_RATIOS, MODES, WriterError, mode_for  # noqa: E402
from mmx_utils.h3_writer.providers import (  # noqa: E402
    GGUF_LOCAL,
    HF_LOCAL,
    HTTP_PROVIDERS,
    PROVIDERS,
    TEMPLATE,
    Settings,
    open_provider,
)
from mmx_utils.llm_http import OFFLINE_ENV, offline, scrub  # noqa: E402

DEFAULT_NEGATIVE = "blurry, low quality, distorted faces, extra limbs, flicker, watermark, subtitles, garbled text"
_LLM = [p for p in PROVIDERS if p != TEMPLATE]
_LOCAL = [HF_LOCAL, GGUF_LOCAL]
_HTTP = list(HTTP_PROVIDERS)
_EDIT_MODES = [m for m in MODES if m.split(" - ")[0] in ("V2V edit", "Replacement", "Continuation")]


def _show(*providers: str, section: str | None = None) -> dict:
    d: dict = {"c2c_show_if": {"provider": list(providers)}}
    if section:
        d["c2c_section"] = section
    return d


def _interrupt_check():
    try:
        import comfy.model_management as mm
        return mm.throw_exception_if_processing_interrupted
    except Exception:  # noqa: BLE001 - outside ComfyUI (tests)
        return None


def _node_id() -> str | None:
    try:
        from comfy_execution.utils import get_executing_context
        ctx = get_executing_context()
        return str(ctx.node_id) if ctx is not None else None
    except Exception:  # noqa: BLE001
        return None


class MiniMaxH3_H3PromptWriter(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MiniMaxH3_H3PromptWriter",
            display_name="H3 Prompt Writer",
            category="MiniMax H3/Prompt",
            description=(
                "Writes a MiniMax H3 prompt in the official format for the chosen mode, looking at the connected "
                "pictures, video and audio. Any model: the offline template, local Hugging Face or GGUF models, "
                "Ollama, an OpenAI-compatible server, or the OpenAI / Anthropic / Gemini APIs."
            ),
            inputs=[
                io.Combo.Input("mode", options=list(MODES), default=MODES[0],
                               tooltip="The H3 task. It decides the prompt structure and which references it needs."),
                io.String.Input("brief", default="", multiline=True,
                                tooltip="What should happen, in plain words. Name references as <Picture 1>, "
                                        "<Video 1>, <Audio 1> to give each a role."),
                io.Float.Input("duration_seconds", default=5.0, min=1.0, max=20.0, step=0.5,
                               tooltip="Target video length in seconds."),
                io.Combo.Input("aspect_ratio", options=list(ASPECT_RATIOS), default=ASPECT_RATIOS[0]),
                io.Combo.Input("provider", options=list(PROVIDERS), default=TEMPLATE,
                               tooltip="template (no LLM) is offline and deterministic. The others write the prompt "
                                       "with a model: local (transformers, GGUF) or over HTTP."),
                io.Combo.Input("model_preset", options=list(catalog.preset_labels()), default=catalog.CUSTOM,
                               tooltip="A tested model list. model_id, when filled, wins.",
                               extra_dict=_show(*_LOCAL)),
                io.String.Input("model_id", default="",
                                tooltip="Any model: a Hugging Face repo id (Qwen/Qwen3-VL-4B-Instruct, "
                                        "google/gemma-4-E4B-it, Jackrong/Qwopus3.5-4B-v3 ...), a local folder or "
                                        ".gguf file, org/repo::file.gguf, an Ollama tag or an API model name.",
                                extra_dict=_show(*_LLM)),
                io.String.Input("base_url", default="", tooltip="Blank = the provider's default address.",
                                extra_dict=_show(*_HTTP, section="Provider")),
                io.String.Input("api_key_env", default="",
                                tooltip="The NAME of the environment variable holding the API key (blank = the "
                                        "provider's usual one). The key is never stored in the workflow.",
                                extra_dict=_show(*_HTTP, section="Provider")),
                io.Combo.Input("device", options=list(DEVICES), default="auto", extra_dict=_show(HF_LOCAL)),
                io.Combo.Input("dtype", options=list(DTYPES), default="auto", extra_dict=_show(HF_LOCAL)),
                io.Combo.Input("quantization", options=list(QUANTIZATIONS), default="none",
                               tooltip="8-bit / 4-bit need bitsandbytes and a CUDA GPU.", extra_dict=_show(HF_LOCAL)),
                io.Boolean.Input("trust_remote_code", default=False,
                                 tooltip="Run the model repo's own Python code. Only for repos you trust.",
                                 extra_dict=_show(HF_LOCAL)),
                io.Int.Input("gguf_gpu_layers", default=-1, min=-1, max=999,
                             tooltip="Layers on the GPU (-1 = all, 0 = CPU only).", extra_dict=_show(GGUF_LOCAL)),
                io.Int.Input("context_tokens", default=16384, min=2048, max=131072, step=1024,
                             tooltip="llama.cpp context size.", extra_dict=_show(GGUF_LOCAL)),
                io.Boolean.Input("unload_after", default=True, tooltip="Free the local model after the run.",
                                 extra_dict=_show(HF_LOCAL)),
                io.Combo.Input("downloads", options=list(dl.DOWNLOAD_CHOICES),
                               default=dl.DOWNLOAD_CHOICES[0],
                               tooltip="Download a missing local model (and the two official H3 guides, 40 KB) into "
                                       "models/LLM when the node runs. The C2C setting 'Never download models' always "
                                       "wins.", extra_dict=_show(*_LLM)),
                io.Combo.Input("media_to_model", options=list(pipeline.MEDIA_CHOICES),
                               default=pipeline.MEDIA_CHOICES[0],
                               tooltip="What the model receives besides the reference labels.",
                               extra_dict=_show(*_LLM)),
                io.Int.Input("video_samples", default=6, min=2, max=24,
                             tooltip="Frames on each video's contact sheet."),
                io.Float.Input("temperature", default=1.0, min=0.0, max=2.0, step=0.05,
                               extra_dict=_show(*_LLM, section="Advanced")),
                io.Int.Input("max_tokens", default=2048, min=256, max=16384,
                             extra_dict=_show(*_LLM, section="Advanced")),
                # control_after_generate=False: the frontend would otherwise add a "randomize" control to any
                # input named seed, so every queue would ask the model again and the prompt would never be cached.
                io.Int.Input("seed", default=0, min=0, max=2**31 - 1, control_after_generate=False,
                             tooltip="Change it for a different reply from the same inputs.",
                             extra_dict=_show(*_LLM, section="Advanced")),
                io.Boolean.Input("repair", default=True,
                                 tooltip="When an objective format check fails, ask the model for one correction.",
                                 extra_dict=_show(*_LLM, section="Advanced")),
                io.Int.Input("timeout_s", default=180, min=5, max=1800,
                             tooltip="Longest wait for one reply (HTTP request or llama.cpp child process).",
                             extra_dict=_show(*_HTTP, GGUF_LOCAL, section="Provider")),
                io.Int.Input("retries", default=1, min=0, max=3,
                             tooltip="Extra attempts after a timeout, 429 or 5xx.",
                             extra_dict=_show(*_HTTP, section="Provider")),
                io.String.Input("system_prompt_override", default="", multiline=True,
                                tooltip="Blank = the ported mode instructions.",
                                extra_dict=_show(*_LLM, section="Advanced")),
                io.String.Input("negative", default="", multiline=True,
                                tooltip="Blank = a short default negative.", extra_dict={"c2c_section": "Advanced"}),
                io.Boolean.Input("keep_plate_audio", default=True,
                                 tooltip="V2V edit / Replacement / Continuation: the source video's own sound is "
                                         "kept (audio reuse) when it has one.",
                                 extra_dict={"c2c_show_if": {"mode": _EDIT_MODES}}),
                io.Image.Input("images", optional=True, tooltip="Pictures: each batch item is the next <Picture N>."),
                io.Image.Input("images_2", optional=True, tooltip="More pictures (another size), numbered after images."),
                io.Image.Input("images_3", optional=True, tooltip="More pictures, numbered after images_2."),
                io.Video.Input("video", optional=True, tooltip="<Video 1>: the source or reference video."),
                io.Image.Input("video_frames", optional=True,
                               tooltip="A video as an IMAGE sequence: the next <Video N>."),
                io.Float.Input("video_frames_fps", default=24.0, min=1.0, max=240.0, optional=True,
                               tooltip="Frame rate of video_frames, for the timestamps."),
                io.Audio.Input("audio", optional=True, tooltip="<Audio 1>."),
            ],
            outputs=[
                io.String.Output("prompt", display_name="PROMPT"),
                io.String.Output("negative", display_name="NEGATIVE"),
                io.String.Output("report", display_name="REPORT"),
            ],
        )

    @classmethod
    def execute(cls, mode=MODES[0], brief="", duration_seconds=5.0, aspect_ratio=ASPECT_RATIOS[0], provider=TEMPLATE,
                model_preset=catalog.CUSTOM, model_id="", base_url="", api_key_env="", device="auto", dtype="auto",
                quantization="none", trust_remote_code=False, gguf_gpu_layers=-1, context_tokens=16384,
                unload_after=True, downloads=dl.DOWNLOAD_CHOICES[0], media_to_model=pipeline.MEDIA_CHOICES[0],
                video_samples=6, temperature=1.0, max_tokens=2048, seed=0, repair=True, timeout_s=180, retries=1,
                system_prompt_override="", negative="", keep_plate_audio=True, images=None, images_2=None,
                images_3=None, video=None, video_frames=None, video_frames_fps=24.0, audio=None) -> io.NodeOutput:
        t0 = time.perf_counter()
        report: list[str] = []
        notes: list[str] = []
        secret = ""
        prov = None
        try:
            m = mode_for(mode)
            refs = media.collect_references(images, images_2, images_3, video, video_frames, video_frames_fps, audio,
                                            int(video_samples), notes)
            if m.key in ("V2VEdit", "Replacement", "Continuation") and keep_plate_audio:
                if media.add_plate_audio(refs) is not None:
                    notes.append("plate audio: <Video 1> has sound; it is kept as an audio reference")
                elif any(r.ref == "<Video 1>" and r.has_audio for r in refs):
                    notes.append("plate audio: not added - the 3-audio / 12-reference limit is reached")
            refs = assembly.validate(m, brief, float(duration_seconds), aspect_ratio, refs, notes)
            negative_out = negative.strip() or DEFAULT_NEGATIVE

            if provider == TEMPLATE:
                for r in refs:
                    r.sent = False
                prompt = template.write(m, brief, float(duration_seconds), refs)
                assembled = assembly.build(m, brief, float(duration_seconds), aspect_ratio, refs,
                                           {"base": guides.BUILTIN_BASE, "reference": guides.BUILTIN_REFERENCE},
                                           send_visual=False, send_audio=False)
                audit = pipeline.audit(prompt, assembled)
                report.append(f"H3 Prompt Writer - {m.label} - template (no LLM), deterministic, offline")
                report.extend("reference: " + r.manifest_line() for r in refs)
                report.extend(notes)
                report.append(_audit_line(audit))
                report.append(f"time: {time.perf_counter() - t0:.2f} s")
                return io.NodeOutput(prompt, negative_out, "\n".join(report))

            if provider in HTTP_PROVIDERS and offline():
                raise WriterError(f"{OFFLINE_ENV} is set (no network): {provider} is not used. Pick template (no LLM) "
                                  f"or a local provider, or unset {OFFLINE_ENV}.")
            progress = dl.Progress(_node_id(), report)
            preset = catalog.lookup(model_preset) if provider in (HF_LOCAL, GGUF_LOCAL) else None
            if preset is not None and preset.get("provider") != ("hf" if provider == HF_LOCAL else "gguf"):
                raise WriterError(f"The preset {model_preset} is a {preset.get('provider').upper()} model; pick the "
                                  f"matching provider or clear model_preset.")
            settings = Settings(provider=provider, model_id=model_id, preset=preset, base_url=base_url,
                                api_key_env=api_key_env, timeout_s=float(timeout_s), retries=int(retries),
                                device=device, dtype=dtype, quantization=quantization,
                                trust_remote_code=bool(trust_remote_code), gguf_gpu_layers=int(gguf_gpu_layers),
                                context_tokens=int(context_tokens), downloads=downloads)
            report.append(f"H3 Prompt Writer - {m.label} - {provider}")
            prov = open_provider(settings, progress)
            secret = getattr(getattr(prov, "cfg", None), "key", "") or ""
            send_visual, send_audio, media_notes = pipeline.media_plan(prov, refs, media_to_model)
            notes.extend(media_notes)

            gdir = dl.guide_dir()
            allow_fetch = dl.downloads_allowed(downloads)
            guide_notes: list[str] = []
            base_text, base_src = guides.load_guide("base", gdir, allow_fetch, notes=guide_notes)
            ref_text, ref_src = (guides.load_guide("reference", gdir, allow_fetch, notes=guide_notes)
                                 if m.guide == "reference" else ("", ""))
            report.append(f"guide: {ref_src if m.guide == 'reference' else base_src}"
                          + (f" (+ shared base rules: {base_src})" if m.guide == "reference" else ""))
            report.extend("guide note: " + n for n in guide_notes)

            assembled = assembly.build(m, brief, float(duration_seconds), aspect_ratio, refs,
                                       {"base": base_text, "reference": ref_text,
                                        "base_official": base_src.startswith("official")},
                                       system_prompt_override, send_visual, send_audio)
            progress(f"writing with {prov.name}")
            result = pipeline.run(prov, assembled, repair=bool(repair), seed=int(seed),
                                  temperature=float(temperature), max_tokens=int(max_tokens),
                                  check_interrupt=_interrupt_check())
            sent_imgs = sum(1 for r in refs if r.sent and r.kind == "image")
            sent_vids = sum(1 for r in refs if r.sent and r.kind == "video")
            sent_aud = sum(1 for r in refs if r.sent and r.kind == "audio")
            report.extend("reference: " + r.manifest_line() for r in refs)
            report.append(f"sent to the model: {sent_imgs} picture(s), {sent_vids} video sheet(s), "
                          f"{sent_aud} audio clip(s)" + (" ; custom system prompt" if assembled["system_custom"] else ""))
            report.extend(notes)
            report.append(_audit_line(result["audit"]))
            if result["repair_attempted"]:
                report.append(f"repair: {result['repair_method']} for {result['repair_reason']} - "
                              + ("applied" if result["repair_applied"] else f"not applied ({result['repair_failure']})"))
            u = result["usage"]
            report.append(f"tokens: {u.get('prompt_tokens', 0)} in, {u.get('completion_tokens', 0)} out; "
                          f"generation {result['seconds']:.1f} s; total {time.perf_counter() - t0:.1f} s")
            report.extend("http: " + line for line in getattr(prov, "log", []))
            report.append("a model reply is not deterministic: change seed for another (same inputs = cached)")
            text = "\n".join(scrub(line, secret) for line in report)
            return io.NodeOutput(result["prompt"], negative_out, text)
        except WriterError as e:
            raise RuntimeError("H3 Prompt Writer: " + scrub(e.message, secret)) from None
        finally:
            if prov is not None:
                try:
                    prov.close(bool(unload_after))
                except Exception:  # noqa: BLE001 - freeing must not mask the real result
                    pass


def _audit_line(audit: dict) -> str:
    from mmx_utils.h3_writer.repair import audit_failures
    failures = audit_failures(audit)
    status = "pass" if audit.get("official_format_pass") else "FAIL"
    line = f"format audit: {status}"
    if failures:
        line += " - " + "; ".join(failures)
    if audit.get("quality_warnings"):
        line += " ; warnings: " + ", ".join(audit["quality_warnings"])
    if audit.get("detailed_description_words") is not None:
        line += f" ; detailed_description {audit['detailed_description_words']} words"
    return line
