# H3 Prompt Writer as a workflow node

Status: design, 2026-11-23 (owner order of the same day). Written before the code.

Source: `third_party/ComfyUI-MiniMaxH3-Prompt-Writer` (MIT, "H3 Prompt Writer contributors", extension 0.4.8). It is a
floating-window UI extension, not a node: it keeps media in a server-side session store and talks to the model through
its own HTTP routes. The owner wants the same prompt-writing logic as a real workflow node: image, video and audio
come in as sockets, a mode chooses the H3 prompt structure, any provider and any model, prompt and report come out.

## Decision: a new node beside Prompt Writer v2, not more inputs on v2

`MiniMax H3 Prompt Writer v2` (`MiniMaxH3_PromptWriter`) and this node solve different problems:

| | Prompt Writer v2 | H3 Prompt Writer (new) |
|---|---|---|
| Organised by | 21 replacement / edit **families** (Family Presets, the 8-field plan) | the official H3 **modes** (T2VA, I2VA, FL2VA, L2VA, Reference) plus edit / replacement / continuation tasks |
| Media | none: the reference roles are fixed by the family | real IMAGE / VIDEO / IMAGE-sequence / AUDIO sockets that the model looks at |
| LLM job | fills descriptive JSON fields inside a fixed template | writes the whole prompt to the official guide, then the audit and the repair pass check it |
| Providers | HTTP only (Ollama, OpenAI-compatible, cloud APIs) | the same HTTP providers **plus** local Hugging Face transformers and local GGUF |

Putting the new behaviour on v2 would give one node two unrelated sets of controls (a family dropdown and a mode
dropdown, most widgets meaningful for one and not the other). That is worse UX, and appending more than a dozen inputs
to a saved node risks the frontend's positional widget restore (see memory: "Frontend 1.53 widget restore order").
v2 stays exactly as it is, so every saved workflow loads unchanged; the new node gets its own id.

- node id `MiniMaxH3_H3PromptWriter`, display name **H3 Prompt Writer**, category `MiniMax H3/Prompt`, V3
  `io.ComfyNode`, registered as one row in `_PORTED_SPECS` (import-guarded like every other row).
- v2's description gains one sentence pointing to the new node (no input change).

## Inputs and outputs

Widgets (order fixed now, new widgets are only ever appended):

| Input | Type | Notes |
|---|---|---|
| `mode` | combo | `T2VA - text to video`, `I2VA - first frame`, `FL2VA - first and last frame`, `L2VA - last frame`, `Reference - reference to video`, `V2V edit - edit the source video`, `Replacement - replace a subject in the plate`, `Continuation - extend the source video` |
| `brief` | multiline | plain-language creative brief; may use `<Picture N>`, `<Video N>`, `<Audio N>` |
| `duration_seconds` | float 1-20 | target length (H3 makes 4-15 s; the guide allows up to 20) |
| `aspect_ratio` | combo | the source's eight ratios |
| `provider` | combo | `template (no LLM)`, `huggingface transformers (local)`, `gguf llama.cpp (local)`, `ollama`, `openai-compatible server`, `openai api`, `anthropic api`, `gemini api` |
| `model_preset` | combo | `custom (use model_id)` + the curated list in `mmx_utils/h3_writer/models.json` (HF and GGUF entries) |
| `model_id` | string | free text, wins over the preset: any HF repo id (`Qwen/Qwen3-VL-4B-Instruct`, `google/gemma-4-E4B-it`, `Jackrong/Qwopus3.5-4B-v3`, ...), a local folder or `.gguf` file, `repo/id::file.gguf` for a GGUF repo, an Ollama tag, or an API model name |
| `base_url`, `api_key_env` | string | HTTP providers only; the key is read from the named environment variable |
| `device`, `dtype`, `quantization`, `trust_remote_code` | | HF transformers only (`quantization` 8-bit / 4-bit needs bitsandbytes; words-only error otherwise) |
| `gguf_gpu_layers`, `context_tokens` | int | GGUF only |
| `unload_after` | bool | free the local model after the run (default on) |
| `downloads` | combo | `download if missing` / `never download` |
| `media_to_model` | combo | `send connected media (error if the model cannot take it)`, `images and video only (audio as a label)`, `labels only (send no media)` |
| `video_samples` | int 2-24 | frames on each video contact sheet (source default 6) |
| `temperature`, `max_tokens`, `seed`, `repair`, `timeout_s`, `retries` | | sampling and HTTP limits; `repair` runs the audit-driven correction pass |
| `system_prompt_override` | multiline | blank = the ported mode wrapper |
| `negative` | multiline | blank = a short default negative for H3 |
| `keep_plate_audio` | bool | V2V edit / Replacement / Continuation: `<Video 1>`'s own sound becomes the next `<Audio N>` ("synchronized audio track", guide 2.5) and the task label gains `audio reuse` / `audio reference` |

Optional sockets: `images`, `images_2`, `images_3` (IMAGE; every batch item is one `<Picture N>`, numbered in that
order, so different-sized pictures go to different sockets), `video` (VIDEO, `<Video 1>`), `video_frames` (IMAGE
sequence, the next `<Video N>`), `video_frames_fps` (float, timestamps for the sequence), `audio` (AUDIO,
`<Audio 1>`). Limits follow the source: up to 9 pictures, 3 videos, 3 audio, 12 in total; base modes take exactly
the pictures they need (I2VA 1, FL2VA 2, L2VA 1, T2VA 0).

Outputs: `prompt` (STRING), `negative` (STRING), `report` (STRING: provider, model and where it came from, download
and load times, media manifest and what was sent, capability decisions, audit result, repair attempted / applied /
why it failed, warnings). No floating window. The node panel shows the run's progress text (download, load,
generate) through core's `send_progress_text` when the server provides it.

Caching: the default (same inputs, same output). `seed` gives a new LLM reply on purpose. No `fingerprint_inputs`
that hashes tensors or keys.

## Modes and the H3 prompt structure

| Mode | Guide | Structure | Required references |
|---|---|---|---|
| T2VA / I2VA / FL2VA / L2VA | base | instruction line (I2VA / FL2VA / L2VA, exact forms from the guide) + `integrated_multimodal_description`, `overall_soundscape`, `non_diegetic_music` | 0 / 1 / 2 / 1 pictures |
| Reference | reference | six sections: `subject_definitions`, `summary` ([task label]), `retention_analysis`, `detailed_description` ([Shot 1], 350-500 words), `overall_soundscape`, `non_diegetic_music` | at least one reference |
| V2V edit | reference | as Reference, classified `video editing` (+ `audio reuse` when the plate keeps its sound); summary starts "The target video is an edited version of <Video 1>." | `<Video 1>` |
| Replacement | reference | `video editing + reference generation` (+ `audio reuse`): `<Video 1>` keeps composition, camera, motion, timing, setting, light and sound; the `<Picture N>` refs give only the new subject's identity and look | `<Video 1>` + at least one picture |
| Continuation | reference | `video continuation` (+ `audio reference`) | `<Video 1>` |

The final grounding contract for the four base modes and Reference is ported verbatim (`assembly._final_contract`);
V2V edit, Replacement and Continuation add one classification sentence each in the same style.

## The official guides (licence)

The two guides are MiniMax "Documentation" under the MiniMax H3 Community License (territory limits, NOTICE and
agreement-copy duties, "no additional terms"). Bundling them inside this GPL-3.0 repository is a doubtful case, so
they are **not committed** (NEEDS_OWNER note in the report). Instead:

1. `guides.py` keeps the source's pinned revision and SHA-256 values and loads the guides from
   `models/LLM/h3_prompt_guides/` when present (a copy the user placed, or one fetched earlier).
2. When missing and downloads are allowed, the run fetches the two files (about 40 KB) from
   `huggingface.co/MiniMaxAI/MiniMax-H3` at the pinned revision and verifies the SHA-256 before use.
3. Otherwise (never download, offline, hash mismatch) it uses a short format contract written here in our own words:
   the section names, the three instruction-line forms, the task-label list and the timestamp form. The report says
   which guide text was used.

The system wrappers (`system_prompts.py`), the audit, the repair prompts, the reference policy, the contact-sheet
layout and the final contracts are the Writer's own MIT code and text: ported with
`# PORTED FROM: third_party/ComfyUI-MiniMaxH3-Prompt-Writer/<file> (MIT, H3 Prompt Writer contributors)` headers,
credited in CREDITS.md, licence copy in THIRD_PARTY_LICENSES/. Nothing imports `third_party/` at runtime.

## Providers

One neutral message format (`[{"role", "parts": [text | image(jpeg bytes) | audio(wav bytes)]}]`); each provider
translates it and reports what it can take.

| Provider | Images / video sheet | Audio | Notes |
|---|---|---|---|
| template (no LLM) | labels only | label only | deterministic, offline, always runs; output passes the structural audit |
| HF transformers (local) | when the processor has an image processor | when the processor has an audio feature extractor (e.g. Gemma 4 E2B/E4B, Qwen2.5-Omni) | any repo id or local folder; AutoProcessor + AutoModelForImageTextToText / AutoModelForMultimodalLM / AutoModelForCausalLM; device, dtype, 8/4-bit; `trust_remote_code` off by default; model cached between runs unless `unload_after` |
| GGUF llama.cpp (local) | only with an mmproj beside the model **and** a matching llama-cpp-python chat handler (Qwen2.5-VL, LLaVA 1.5/1.6, MiniCPM-V 2.6) | no | runs in a **child process** (`gguf_worker.py`): a llama-cpp-python wheel with AVX-512 code dies with an illegal instruction on CPUs without it, which in-process would kill ComfyUI (the reason v2 never loads a model in-process); the child's death becomes a sentence. Gemma 4 / Qwen3.x vision GGUFs need a newer runtime: run `llama-server` and use `openai-compatible server` (the error says so) |
| Ollama | when `/api/show` lists `vision` | no | `/api/chat` with base64 images |
| OpenAI-compatible | sent as `image_url` data URIs; a 4xx is reported in words ("this server/model may not accept images") | `input_audio` only when asked | llama-server, LM Studio, vLLM, OpenRouter |
| OpenAI / Anthropic / Gemini | yes | Gemini yes (inline WAV); OpenAI only for audio-capable model names; Anthropic no | key from the environment only |

A model that cannot take a connected modality stops the run with a words-only error that names the input, the model
and the two ways out (pick a capable model, or set `media_to_model` to the label-only choice). No stack trace text.

Video understanding is an ordered contact sheet (source layout: 2-4 columns, numbered cells, endpoints included),
never every frame. Audio sent to a model is mono 16 kHz WAV, at most 30 s.

## Downloads and keys

- Never at import or test time. Only inside `execute`, only for a local provider whose model is not on disk.
- Where: ComfyUI's `models/LLM/` (the folder ComfyUI-QwenVL and the Writer use; a registered `LLM` / `llm`
  folder-path wins). HF repos go to `models/LLM/<org>--<name>/`, GGUF repos to `models/LLM/<org>--<name>/<file>`.
- Allowed only when `downloads` = `download if missing` AND the C2C setting `c2c.models.neverDownload` is off (read
  with the Suite's `_c2c_saved_setting`, so it works with or without CustomNodePacks; `C2C_NO_DOWNLOADS=1` still wins)
  AND `MMX_PROMPT_WRITER_OFFLINE` is not set. Otherwise the error says what is missing and where to put it.
- Progress text per file goes to the node panel and the report.
- Keys: the existing `mmx_utils/llm_http` rules (L9.288) are reused, not copied: the key comes only from the named
  environment variable, goes only to loopback, the key's own service or `MMX_PROMPT_WRITER_HOSTS`, travels in a
  header, and is scrubbed from every message. The workflow stores only the variable name. HF gated repos use the
  user's own `huggingface_hub` login / `HF_TOKEN`; the node never reads or logs it.

## Robustness

- `mmx_utils/h3_writer/` imports only the standard library at module level; torch, PIL, numpy, transformers,
  huggingface_hub and llama_cpp are imported inside the functions that need them, each guarded, so a missing or
  broken optional library fails one provider with a sentence and never the pack.
- Interrupt is honoured between steps and inside generation (stopping criteria / per-token callback).
- Context budget: estimated before the call; a too-small context is a words-only error.

## Tests (local only, never committed)

`tests/test_h3_prompt_writer_node.py`: schema and registration; template output for every mode passes the ported
audit; reference numbering and limits; contact sheet layout; audit and repair (fake `complete`); provider payloads
built from mocked HTTP (no network, no paid call); capability errors; download gate (setting, env, input) with a fake
downloader; key never in report or payload logs; HF provider end to end on a tiny random model created in a temp
folder (no download); guides loaded from a temp folder with hash check.

## Not in this slice

Sequence / multi-chunk writing, Music 3 captions, the media editor and composer, the "Refine" chat loop (re-running
with an edited brief covers it), streaming text into the node.
