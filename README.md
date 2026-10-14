# ComfyUI-MiniMaxSuite — H3 geometric spine + sampling guards

GPL-3.0 node pack (see `LICENSE`) for MiniMax H3 production workflows (V3 `io.ComfyNode` API).

## Nodes

| Node | Role |
|------|------|
| **H3 Frame Handles** (N7) | Editorial head/tail handles + H3/WAN/LTX2 grid padding; audio resync |
| **H3 Track + Crop** (N1) | TV/L1 crop planner; `crops` + `H3_TRANSFORM` |
| **H3 Control Hints** (N3) | Crop-space canny / passthrough hints (feed core's Apply MiniMax H3 Fun ControlNet, or H3 Mask-Aware ControlNet below) |
| **H3 Mask Prep** (N2) | Video-side latent mask |
| **H3 Masked Replace** (N4) | Inject crops + `noise_mask` NestedTensor (lipsync via audio lock) |
| **H3 Stitch Back** (N5) | Inverse affine paste; exterior = literal plate |
| **H3 Detail Reinject** (N6) | Plate texture transfer inside mask |
| **H3 Protected Layer Guard** (P10) | Fail if protected DiT blocks / FinalLayer paths are patched |
| **H3 Accelerator Conflict** (P11) | Fail on incompatible cache stacks; passes MODEL through |
| **H3 Legal Scheduler** (P3) | Validate sigma schedule; rejects SplitSigmas |
| **H3 Sigma Shift (Ratio Locked)** (P1) | ModelSamplingAV with video/audio ratio lock (default 4 = 12/3) |
| **H3 Per-Frame Denoise** (P6) | Face-size-driven per-frame noise_mask strength |
| **H3 Block Cache (T8)** (P9) | Dual-metric F1B0 block cache; `cache_device` default `cpu` |
| **H3 Sigma Inspector** (P2) | Read-only σ table report (no JS plot in this slice) |
| **H3 OCIO Bridge** (I1) | Scene-linear ↔ display via OCIO (HDR-safe, no clamp) |
| **H3 Color Round-Trip QC** (W4) | Display round-trip error probe with HDR gate |
| **H3 Strongest Pose** (N8) | ViTPose-H plate detect → crop warp pose hints |
| **H3 Temporal Depth** (N9) | Depth hints (passthrough default) + optional normals |
| **H3 Drift QC** (N11) | Exterior-only RAFT drift gate |
| **H3 Region Mask** (N10) | Depth-horizon or painted region MASK for N2 spine |
| **H3 Family Presets** (N12) | Six-section prompt (#14) + style DNA (#16) |
| **H3 Edit Validator** (N13) | Pre-flight gate (#8) — mask_required, grid, drift |
| **H3 DCC Bridge** (N14) | External EXR sim ingest (no solver) |
| **H3 Pose Puppeteer** (N15) | Driving pose → motion hints (Union blocked) |
| **H3 HDR Roundtrip** (N16) | OCIO view↔scene with HDR tag propagation |
| **H3 Hybrid HDR LoRA** | Apply extracted HDR LoRA to Ref2VA (constant strength) |
| **H3 Hybrid HDR Extract** | Extract LoRA from Ref2VA/FL2VA/Singularity triple on GPU box |
| **H3 Color QC** | Hasler-Süsstrunk + luma DR + texture/flicker for Hybrid A/B |

N1 is a **planner**, not a detector — wire `subject_mask` and/or `bboxes_json`.

## Character replacement: room for the new head

H3 has no face or expression input: identity comes from `<Picture N>` references, structure and expression only
from pixels it sees (the kept plate, a reference video, or the Fun ControlNet's control video). Every geometric
signal of a crop -> mask -> regenerate -> uncrop replacement comes from the ORIGINAL performer, so the new head can
be no bigger than the room the crop and the mask leave. Measured with MVEx / H3 Subject Crop `combined`
`crop_scale` 1.2, mask expand 40 px, 32 px tokens (`docs/research/l820_minimax_character_replacement.md` in the
C2C workspace): a head 1.3x the original has 86 % of its area in the regenerated region, 1.5x only 67 % - the model
has to draw it smaller.

**Head Size Mode** on **H3 Subject Crop** / **H3 Subject Crop (Advanced)** (owner decision 2026-10-13) makes the
room explicit:
- `head_size_mode`: **match character proportions** (default) - room for the new character's own head-to-body ratio,
  anchored at the chin; or **keep original composition** - the original performer's head size and position.
- `head_scale`: new head / original head; 0 = measure it from `original_pose` and `reference_pose` (any
  POSE_KEYPOINT: DWPose, SDPose, controlnet_aux) as reference face-width/shoulder-width over the original's, so
  resolution and framing cancel. Without poses it is 1 and the report says why.
- `head_room_masks` (new output): the subject mask plus the target head, padded by 10 % of the head height - feed it
  to the latent mask and to **H3 Subject Uncrop**'s `cropped_masks`, so the regenerated and the composited region
  match. The head is scaled uniformly about the chin, never stretched.
With nothing connected and `head_scale` 0 the crops are bit-identical to before. When a pose / depth control runs
next to a latent mask, use **H3 Mask-Aware ControlNet** instead of the core apply (core adds the control to the kept
plate too).
**H3 Swap Control** renders only brows / eyes / nose / mouth so the control carries the performance, not the
original skull. Quality effects are not yet A/B-tested on H3 weights.

### Six character-replacement workflows

`workflows/character_replacement/` holds one ready-to-load graph per task: replace keeping the composition, keep
identity and change the pose, keep the pose and change the look, transfer an expression, replace a masked subject,
and replace across a whole take. Each carries its guide as a note on the canvas, and the full step-by-step guide is
[`docs/character_replacement.md`](docs/character_replacement.md) (inputs, mask and reference requirements, Head Size
Mode, starting settings, and the GPU A/B to run before trusting the result). **H3 Family Presets** has six matching
families that write a fixed, structured prompt per task. `<Video 1>` is always the cropped plate. The `head_size_mode`
there must match the one on H3 Subject Crop. **MiniMax H3 Prompt Preview & Edit** now shows that prompt after a run;
typing in it turns `use_edited_text` on, so your edit is what runs next. Optional and experimental: `face_closeup_reference` names a tight face crop of the character, wired as `ref_image_1`, as `<Picture 2>`, the authority for facial identity. The graphs are checked in both renderers and
by the server's validation, but **not yet run on H3 weights**; the settings are the official r2v template's, not tuned.

### Measuring a replacement: `tools/h3_replace_bench.py`

Answers the question for character replacement: does the result keep the source performance while the identity
changes? Run it with ComfyUI's Python. Nothing is downloaded.

- `score --plate P --output O --mask M [--performance D] [--reference R]` measures:
  - face performance against the source (or the driving clip for expression transfer): mouth, eyes, gaze and head
    pose, each as Pearson r and MAE, plus mouth open/close event F1 and a frozen-expression flag;
  - background preservation outside the mask (PSNR), leakage at the mask edge, excess flicker inside the mask, and
    colour drift;
  - identity, as ArcFace cosine, only where insightface and its `buffalo_l` model are already installed.
- Face landmarks come from mediapipe FaceMesh, whose models ship inside the mediapipe package. A frame without a
  face is excluded and counted, never guessed.
- `run --manifest bench.json --server …` queues each workflow of a matrix (e.g. the A12 T0–T7 set) one at a time. It
  times each run from ComfyUI's own timestamps, records peak GPU memory with `nvidia-smi` (whole GPU, over idle) and
  scores the newest output.
- `report results/*.json` prints one table.

Checked on a real face clip with a known head roll: an identical output scores r = 1 on every signal; the same clip
4 frames late drops mouth r to 0.31 and event F1 to 0.38. No H3 output has been scored yet; that is the owner's GPU
run.

## MiniMax H3 Prompt Writer v2

Writes the structured H3 prompt for any H3 Family Presets task from a plain-language brief. It plans 8 fields (scene
and style, performance to preserve, identity to transfer, face motion, body and camera, mask region, temporal rules,
negative rules) and writes them into H3's six sections. The reference roles come from the task's template, so a model
cannot rebind `<Video 1>` or invent a `<Picture 3>`.

- **`template (no LLM)`** (default): deterministic, offline, no model. Replacement tasks state the plate's performance
  (expressions, mouth, head, body, timing) as the authority, plus the mask and consistency rules.
- **LLM backends**: Ollama, any OpenAI-compatible server (llama-server, LM Studio, vLLM), or the OpenAI / Anthropic /
  Gemini APIs. They refine the descriptive fields over HTTP only. **No model runtime is loaded inside ComfyUI.** The
  llama.cpp-based writer died with an *illegal instruction* when its prebuilt wheel met a CPU without AVX-512, and
  that kills the whole ComfyUI process. To use llama.cpp, run `llama-server` and pick `openai-compatible server`. `python tools/llama_cpp_probe.py [--model x.gguf]`, run with ComfyUI's Python, checks whether a llama-cpp-python install would hit that crash on this CPU, safely in a child process.
- Each backend is checked before the request: is it reachable, is the model present, is the key set. Transient errors
  (429 / 5xx / network) are retried. On any failure the node falls back to the template, or stops if `on_failure`
  says so. The report names the backend used, the time taken, which fields came from the model, and every repair.
- API keys are read from an environment variable that you name (`api_key_env`), are sent only in request headers,
  and never appear in the workflow, the outputs or the report. `MMX_PROMPT_WRITER_OFFLINE=1` forces the template
  everywhere.
- The server settings appear only when an LLM backend is selected. The Promptor LLM director still covers free-form
  prompts.

## H3 Hybrid HDR (Singularity colour without FL2VA damage)

Method: for matched checkpoints, `D = FL−Ref`, `Δ = Sing−Ref`, `a = clamp(⟨Δ,D⟩/⟨D,D⟩,0,1)`, `H = Δ−a·D`; compress `H` per layer to rank‑r LoRA.

**Extract** (once, on GPU box):

```bash
python tools/h3_hybrid_extract.py \
  --ref ref2va.safetensors --fl fl2va.safetensors --sing singularity.safetensors \
  --out models/loras/h3_hybrid_hdr.safetensors --rank 64 --blocks 25-49 --report hybrid_report
```

Or use node **MiniMaxH3_HybridExtract** in ComfyUI.

**Apply**: wire **MiniMaxH3_HybridHDR** on a Ref2VA MODEL before sampling (`strength` 0.5–1.0, blocks 25–49 default).

**Late-step HDR** (two-stage — not hook keyframes): sample early steps with plain Ref2VA, then continue with HybridHDR MODEL via `KSamplerAdvanced` start/end steps or `SamplerCustomAdvanced`. Do **not** use SplitSigmas on H3.

**A/B workflow**: `workflows/h3_hybrid_hdr_ab.json` — Ref2VA vs Singularity vs HybridHDR @0.5/1.0 into **MiniMaxH3_ColorQC**.

See `docs/PLAN_h3_hybrid_hdr.md` for limits and report interpretation.

## Dual-clock shim (P4 — not shipped)

On ComfyUI core `180060c` and later, `ModelSamplingAV` is native (`comfy/model_sampling.py`). A dual-clock sampler shim is a no-op — use core **MiniMaxH3SigmaShift** or our **MiniMaxH3_SigmaShiftLocked** for ratio-locked shifts.

## Workflow

See `workflows/h3_masked_face_pipeline.json` and `workflows/README.md` (runtime untested without H3 weights).

Recommended sampling chain after N4:

`N4` → `P10 ProtectedLayerGuard` → `P11 AcceleratorConflict` → `P1 SigmaShiftLocked` → `P3 LegalScheduler` → sampler

## Tests

```bash
cd ComfyUI-MiniMaxSuite
python run_tests.py
```

One test file per `mmx_nodes/` and `mmx_utils/` module. CPU-only; no weights required.

## Licence

Ported utilities carry upstream licence headers and `# PORTED FROM:` provenance lines.
