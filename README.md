# ComfyUI-MiniMaxSuite — H3 geometric spine + sampling guards

GPL-3.0 node pack (see `LICENSE`) for MiniMax H3 production workflows (V3 `io.ComfyNode` API).

## Nodes

Every node shows its stage, its run status and H3 errors in plain English. That status follows the node inside (nested) subgraphs too: the server names such a node `outer:inner`, and the root graph alone never finds it.

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

## Masked replacement without a spatial offset (A16)

When the sampler starts from the SOURCE video latent (masked editing) and references are attached, every stream that
shows the same shot as the output - the source, a pose or depth video of it, a guide frame - has to be on the output's
own token grid. ComfyUI's Reference to Video sizes reference videos to its 768-short-edge canvas; H3 gives each token
the coordinate of its top-left edge, so a reference with bigger tokens reaches the model half a token up and left of
its content: **-14 px** for a 1440x1440 crop with its 768x768 reference, 7-18 px for a 1344x1536 crop (measured with
ComfyUI's own layout code; the effect on real outputs still needs a GPU run).

- **H3 Geometry Report** (new): wire the positive conditioning and the latent the sampler gets. It lists every
  reference and guide with its size, token grid and the offset in pixels (also in plate pixels with the crop boxes),
  flags guides that are not on the output grid, audio that does not match the frame count, and says what the noise mask
  keeps once ComfyUI has pooled it onto tokens. Shapes only, instant.
- **H3 Image + Reference to Video**: new `ref_video_fit` = `target grid (zero offset)` puts reference videos on the
  output grid (cover-crop if the aspect differs; costs more reference tokens). Default `core canvas` is unchanged. Its
  report now states the offset. Fixed: on ComfyUI 0.39 this node refused every run (its canvas constant was 16 where
  ComfyUI's is 32).
- **H3 Source AV Latent** (new): the source frames + sound as the masked joint latent, without the H3 VAE's silent
  centre-crop of off-grid sizes (refused, or padded and recorded), with the frame count checked (17k+5), the mask reduced
  on H3's own (1,4,4,4,4) frame grouping (a pixel mask given to the sampler is resampled linearly in time: measured one
  latent row early at half strength) and the sound kept / regenerated / following the mask per 1/40 s.

- **H3 Subject Uncrop** `protect` (appended; default unchanged): `exact outside mask` keeps every pixel outside the
  crop mask bit-exact (the feather runs inside the mask edge), `soft mask edge` blends by the mask's own values,
  `whole crop` replaces the box. A `report` / `ledger_json` say per frame what was kept exactly, blended and replaced
  and the largest change measured where the original was kept (0 = bit-exact).
- **H3 Drift QC** placement (appended inputs `source_subject_mask`, `output_subject_mask`, `bboxes`; output
  `placement_json`): where the new subject sits against the old one - centre and box-edge shift per frame, in plate
  pixels with the crop boxes.

- **H3 Mask-Aware ControlNet** (appended): `control_region` (where this control may steer, multiplied with the
  sampler's denoise mask), `schedule` (constant / fade out / fade in / smooth fade out over the sigma window) and
  `crop_bboxes` (a plate-sized control / source / mask gets the source's own crop instead of ComfyUI's cover-crop).
  A gate that cannot be built is now an error instead of a silent fall back to ungated control. What can run together
  in one pass, and what wins when constraints disagree: `docs/h3_mask_control_matrix.md`.

Caching fixed for 36 nodes: their `fingerprint_inputs` read linked tensors (or used their id()), which ComfyUI never passes (they arrive as
None), so ComfyUI stored NaN and the node - and everything after it, the sampler included - ran again on every queue.
Measured live with classic caching: H3 Motion Context Trim re-ran every time before, is cached after.

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

## H3 Identity Fusion Controller (experimental)

RefMod and the reference size decide **what** reference goes in. This node decides **where**, **when** and **how
strongly** the generated video looks at the character pictures (`<Picture N>`). The cropped plate (`<Video 1>`)
stays the authority for the performance.

- **Where.** Identity attention only inside the replacement region (`replacement_mask`, the latent mask of
  `head_room_masks`), so none on the background or on another person. It is eased on mouth and eye rows
  (`expression_mask`), where the plate's own performance is strengthened instead. It is off on an occluder in front
  of the face (`occluder_mask`).
- **When.** A block range and a step window. The expression easing starts later (`expression_from`), because
  mouth and eye shape are fine detail.
- **How.** A multiplier per token row on the attention that goes to the identity pictures. It is applied as an
  exact bias on padded attention channels: no dense mask, no extra model pass, no reserved DiT block. NegPiP, Block
  Cache and Spectrum keep working. An optional asymmetric mode keeps the references from reading the noisy video.

Wire it after the model loader (after H3 Mask-Aware ControlNet if you use one) and before the sampling guards. Its
`positive` comes from Reference to Video. **Not A/B-tested on H3 weights**: it is run T5 of the benchmark in
`docs/character_replacement.md`. Measure it with `tools/h3_replace_bench.py` before you rely on it. Tested on CPU:
the bias equals a dense-mask attention to 1e-5, and through core's own H3 attention module to 1e-4.

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
- The Promptor's Gemini provider also sends its key only in the `x-goog-api-key` header (until 2026-10-15 it put
  the key in the request URL, where a connection error could carry it into the error text and the log). Any
  error text it returns is cleaned of the key.

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
