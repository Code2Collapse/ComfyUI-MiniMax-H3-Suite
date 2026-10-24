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

## Fixes 2026-10-24: nodes that did less than they said

- **QC gates now gate.** H3 Audio Quality Gate, H3 Color Round-Trip QC and H3 Edit Validator run even with nothing
  after them, show their verdict on the node, and pass the checked audio, image or prompt through. Put one before
  your Save and set `on_fail = stop the run`: a failed check stops the queue before anything is saved. Their pass/fail
  output fed nothing before, and they did not run unless something was wired after them.
- **Stitch Continuation checks the overlap.** Prepare Continuation stores the overlap it actually used (after grid
  snapping) in the latent; a different `overlap_frames` on Stitch now stops the run instead of joining at the wrong
  place.
- **AV Bridge takes the clips' frame rates** (`start_fps`, `end_fps`) and converts each clip to 24 fps on the
  constant-frame-rate grid (upstream's conversion); Assemble converts the same way.
- **Legal Scheduler measures the schedule.** With the model connected, it works out which scheduler and denoise
  produced the sigmas and checks those. The widgets were taken at their word before.
- **Latent Upscale (Model) sizes by `scale` or `megapixels`** too (`size_by`), keeping the latent's aspect, and only
  the widgets of the chosen mode are shown.
- **Tone Compensate shows its drift curve** (R, G, B over the overlap frames). The measurement was sent to a panel
  that did not exist.
- **Turbo LoRA works on every install:** the time-embedding grid it needs now ships with the pack (Apache-2.0,
  ComfyUI-MiniMax-H3-Turbo by Larryvrh; see CREDITS). **Hybrid Extract finds ComfyUI-GGUF** in ComfyUI's custom-nodes
  folders under any folder name.
- **Tagged Reference to Video** says when no VAE / audio VAE is connected (the references then reach the text encoder
  only). **Strongest Pose** passthrough reports the real share of frames with a hint, not a flat 1.0.
- **Dual-Clock Shim is deprecated:** every supported core has native ModelSamplingAV. It still loads in saved
  workflows.
- **OCIO nodes list the views of every display** (OCIO Bridge, HDR Round-Trip, Color Round-Trip QC). They showed the
  first display's only, and a view the chosen display lacks is now refused with the ones it has.
- **Region Mask maps crop-space depth through the crop** instead of stretching it over the plate, so the horizon
  lands where it is when the crop is not the full frame.
- **Control Hints / Pose Puppeteer** name their consumer (core's Apply MiniMax H3 Fun ControlNet, or H3 Mask-Aware
  ControlNet). They said no consumer existed.
- **Promptor settings moved out of the pack:** `user/default/minimax_suite/promptor_config.json`. The old in-pack
  `config.json` is tracked by git; an edited copy is moved over once. API keys can come from environment variables:
  set a provider's `api_key_env` to the variable name, write `api_key` as `env:NAME`, or leave it empty for
  `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` / `GEMINI_API_KEY`.

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
  Its panel now ends with **Last run**: how many generated rows the gate fully controlled, held back or blended in
  the run that just sampled. The gate is only known once sampling starts, and this count used to be computed and
  never shown (2026-10-24).
- **MiniMax H3 Preview Override** shows the animated preview on the node. With the default target (`node`) it sent
  its frames to a panel that did not exist, and also hid ComfyUI's own preview, so nothing appeared at all. The
  panel shows the step, frame count, playback rate and what each preview cost (2026-10-24).

Caching fixed for 36 nodes: their `fingerprint_inputs` read linked tensors (or used their id()), which ComfyUI never passes (they arrive as
None), so ComfyUI stored NaN and the node - and everything after it, the sampler included - ran again on every queue.
Measured live with classic caching: H3 Motion Context Trim re-ran every time before, is cached after.

### One node: H3 Region Edit (plate + mask + ControlNet)

**H3 Region Edit** does a masked edit steered by a ControlNet in one node. Connect the plate, the mask (1 =
regenerate) and, optionally, the H3 Fun ControlNet and its control video. It outputs the patched `model` and the
`latent`; feed them to the sampler.
- Inside the mask the ControlNet steers what is regenerated. Outside it the latent keeps the plate and the ControlNet
  adds nothing, so the two never fight. (Core's ControlNet adds to every video row, also the ones the latent holds.)
- `hint`: the control video, inpainting from the plate, or both; *auto* picks the control video when one is
  connected.
- `controlnet_where = everywhere` gives core's behaviour, for comparison.
- It is H3 Source AV Latent + H3 Mask-Aware ControlNet in one node; both remain for finer control.

## Long video: H3 Continuation (A16)

**H3 Continuation Next** (before the sampler) and **H3 Continuation Commit** (after the decoders) make a film of any
length one segment per queue. Queue the same graph again and again: each run continues where the last accepted segment
ended, and a crash, a restart of ComfyUI or a closed browser loses nothing. Design and reasoning:
`docs/h3_continuation_design.md`.

- The context is the previous segment's last frames AS THE START OF THE NEW CLIP: their exact latent tokens (picture
  rows and sound ticks), pinned by the denoise mask. No extra tokens, no position tricks; Commit checks the pinned head
  came back bit-exact and refuses the segment if a node in between dropped the mask.
- `overlap_frames` 39 (default; exactly 65 sound ticks), 22, 56 or 5.
- Nothing lost, nothing doubled: frames are numbered on one global timeline (`frame_000001.png` ...), sound is cut on
  global time (one WAV per segment, total exact to the sample). A segment's sound starts on H3's 25 ms tick grid,
  within 16.7 ms of its picture for 124-frame segments, and the error never adds up from segment to segment.
- Session on disk (`output/h3_continuation/<session>/`): `session.json` written atomically after every change, the
  stored tails with SHA-256 (a damaged tail is refused, not used), statuses running / awaiting approval / accepted /
  failed / rejected. A segment left running by a crash is marked failed and its interval is generated again.
- Drift is measured on every segment: colour against segment 0 (Lab), sharpness ratio, seam jump against the clip's
  usual frame change, frozen frames. `approval` = accept unless drift (default), accept, or always ask; a held
  segment waits in `pending/` until `decision` accepts or rejects it (rejected = the same interval again).
- The overlap is decoded again in every new segment. Commit compares it with the frames already accepted (PSNR in
  dB, mean and worst frame) to show any seam the VAE adds when it decodes the same tokens in a new clip. The report
  ends with the worst value of every measure so far. The overlap PSNR is reported, not used to hold a segment.
- Positions: H3 starts every pass's timeline at its own origin, and no RoPE offset is applied. Continuity comes from
  the pinned tail (reasoning from core's source: `docs/h3_continuation_design.md`, "Temporal positions").
  `H3 Context Windows` prints each window's 40 Hz start for reference only.
- Guides inside the pinned head are dropped (those frames are fixed); guides later in the segment stay. Guides use
  the segment's own frame numbers, so the same guide applies to every segment.
- Long inputs are cut per segment by Next itself (only that segment's frames are decoded from Load Video):
  `source_video` + `source_mask` = edit a long plate segment by segment (the plate is kept where the mask is black;
  `segment_source` / `segment_mask` feed core's Image Composite Masked for an exact composite before Commit),
  `control_video` -> `segment_control` for H3 Mask-Aware ControlNet, `soundtrack` with `sound = keep the soundtrack` =
  the picture follows a fixed long soundtrack (music, dialogue). `sound` can also keep the plate's own sound or
  regenerate it only where the mask covers a frame. The session ends with the plate / soundtrack and says so.
- Frames are taken by index from the file's own first frame: core's time-based trim returns frame 84 for 85 on a file
  whose first frame is 2 frames late; Next does not. A 23.976 fps plate is taken frame for frame (reported); other
  rates are refused. Cropping a long plate (H3 Subject Crop) happens before, as its own stage.

Checked: 30 segments without a model (every frame once and in order, every sound sample in place), and live through a
real ComfyUI server (core 0.39, frontend 1.53.10) over 10 queues - accept, hold, refuse while held, reject, a crash
mid-queue, an unchanged re-queue (runs the next segment, never served from the cache), accept a held segment - and
inside a subgraph in both renderers. A 300-frame plate with a mask video, edited live in 4 segments through Load
Video and Image Composite Masked: every frame outside the mask is the plate's own pixel, generated pixels appear on
exactly the masked frames, and the session ends at frame 300. Picture and sound quality of long runs need the GPU and
H3 weights: not measured.

The older **H3 Motion Context** nodes stay for saved workflows (manual Save / Load by clip index; their audio context
starts before the clip by default and ends 8.3 ms after the video context). **H3 Motion Context Chain** never chained
anything and is now deprecated: hidden from the search, it says to use H3 Continuation.

### Whole clip in one pass: H3 Context Window Sampling (experimental)

**H3 Context Window Sampling** patches the model so every denoising step splits the clip into overlapping time
windows, runs the model once per window and cross-fades the predictions. Unlike H3 Continuation (one segment per
queue), the whole clip stays in one sampling pass; attention runs per window, so its cost grows with the window, not
with the clip. Design: `docs/h3_context_window_executor_design.md`.

- Windows start on 17-frame cycles and are legal H3 lengths (`window_frames` snapped to 17k + 5, `overlap_frames` to
  17m + 5); the last window is cut at the clip's end. A clip no longer than one window runs untouched.
- The sound is cut at the same moments, both channels. With the defaults (141 / 39: stride 102 frames) every edge
  falls on H3's 25 ms tick grid exactly; other settings are within half a tick and the report says so.
- `positions`: `global` (default) - each window keeps the clip's own time positions, so it predicts what the whole
  clip would there (no offset, D0.26). `local (experimental)` lays every window out as a clip of its own and
  re-indexes the keyframes inside it, for A/B tests.
- Video and sound denoise masks and core's H3 Fun ControlNet hint (encoded once) are cut per window; a ControlNet in
  the conditioning and block / step caches are refused with a sentence. Combines with H3 Tiled Diffusion (windows
  in time, tiles in space). Kill switch: `MMX_CONTEXT_WINDOWS=off`.

Checked on core's real H3 model and sampler with tiny random weights: one window is bit-identical to the plain model;
three windows run through core's sampler with masked rows kept exactly; global and local positions reach the model
(different outputs); with Tiled Diffusion every window runs every tile. A frame-local stand-in model comes back
exactly (nothing lost, doubled, reordered or shifted). Picture and sound quality over long clips need the GPU and H3
weights: not measured.

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

## Masks, keyframes and long-take repair (ported 2026-10-22)

These nodes are ported from community H3 packs; CREDITS.md has the sources and licences.

- **H3 Set AV Noise Mask / H3 Clear AV Noise Mask** set or remove the denoise masks of an H3 AV latent, one per
  stream (video and audio).
  - Core's Set Latent Noise Mask writes one tensor, which H3 reads as video only, so protected audio gets
    regenerated.
  - Here the frame mask is snapped to H3's own grid: latent steps of 1 or 4 frames and 32 px tokens.
  - `video_lock` keeps every frame, so only the sound is re-sampled; `audio_lock` keeps the sound.
- **H3 Hard Keyframes (protected stills)** writes still images into the latent at chosen frames and protects them,
  where core's Add Guide only suggests them. Single-frame positions are 1, 18, 35, 52, ... Anywhere else protects the
  4-frame latent step the frame falls in.
- **H3 Blend Latents By Frames, H3 Keyframe Mask, H3 Mask Levels** fix the "photocopy" drift of long chained shots:
  run a short refine pass (denoise about 0.5) on the new segment, then blend it back in by frames. For example,
  `0:0, 22:0, 44:1` keeps the frames that join the previous segment and fades to the refined detail by frame 44.
- **H3 Context Taper Noise (frames / latent)**, experimental: tapered noise on the carried context of a Motion
  Context chain. H3 then keeps the motion but stops copying the old colour, which matters in character-swap chains.
  Validated upstream, not yet on these workflows. It is not for H3 Continuation, whose pinned head must come back
  unchanged.

## Bridges, frozen endings and seams (ported 2026-10-22)

- **H3 AV Bridge / H3 AV Bridge Assemble** generate the MIDDLE between two existing clips.
  - The end of clip A and the start of clip B are encoded into the target and protected, picture and sound.
  - Only the middle is generated, so it starts exactly where A ends and lands exactly where B begins.
  - Kept frames are a shared video / audio boundary: 39, 90, 141 ... (39 + 51k).
  - Assemble joins A + middle + B with the sound cut on exact sample boundaries.
- **H3 Continuation Commit `freeze_tail`**: H3 often ends a clip by freezing on its last image (FL2VA), and a frozen
  tail carried into the next segment carries the freeze on.
  - `report` measures it.
  - `cut before it` shortens the segment to the longest 17k+5 length that ends before the freeze, so the next
    segment continues from moving frames. Frames and sound stay exact (tested).
- **H3 Seam Correction** works on the decoded film.
  - It finds the seams between segments and removes their exposure / colour steps (affine or MKL).
  - Options: motion jolts (optical-flow warp + blend at the seam) and single flash frames.
  - Real shot cuts are left alone when PySceneDetect is installed (`pip install scenedetect`). Without it, every
    exposure step counts as a seam, and the report says so.
- **Refining only the sound:** H3 Set AV Noise Mask with `video_lock` on, then a sampler at denoise 0.4-0.6. The
  picture stays as it is, and the sound is re-sampled around the original.

## Tagged references: write @tags in the prompt (ported 2026-10-23)

- **H3 Tagged Picture / Video / Audio** register references under @tags. Chain them in the order you want them
  numbered. A video can carry its soundtrack, with its own tag (blank: `@<tag>_audio`).
- **H3 Tagged Reference to Video** is core's MiniMax H3 Reference to Video driven by those tags.
  - Only the references the prompt mentions are sent to H3.
  - Each @tag is replaced by the label core gives it: pictures first, then each video (its soundtrack's
    `<Audio N>` just before its `<Video N>`), then standalone audio.
  - Example: `<Subject 1> has the face of @hero_face and moves like @performance.` becomes
    `<Subject 1> has the face of <Picture 1> and moves like <Video 1>.`
  - `compliance`: strict stops on an unknown @tag or more than 9 pictures / 3 videos / 3 audios; soft turns those
    into warnings; disabled leaves the @tags as written.
  - The `compiled_prompt` and `report` outputs show what was sent.

## High resolution: tiled diffusion and the 3D latent upscaler (ported 2026-10-22)

Design: `docs/h3_tiled_sampling_design.md`. Both nodes are experimental until measured on a GPU.

- **H3 Latent Upscale (Model)** resizes the video half of an H3 latent to a new pixel size; the audio half passes
  through untouched.
  - `method = model` uses the trained H3 3D latent upscaler. Put `minimax_h3_latent_upscaler_3d_conv_v1_fp16.safetensors`
    (or `_bf16`) from [LBH-123-AI/Minimax_h3_latent_Upscaler](https://huggingface.co/LBH-123-AI/Minimax_h3_latent_Upscaler)
    (Apache-2.0) into `models/latent_upscale_models`. Pickle files (`.pth`) are refused.
  - The other methods interpolate without a model, like core's Upscale Latent.
  - The whole clip runs in one pass. `chunk_latent_steps` trades memory for an approximate result: the network's
    normalisation sees the whole clip.
  - Then re-sample at low denoise to add detail, with conditioning made for the new size.
- **H3 Tiled Diffusion** patches the model so every denoising step runs tile by tile and the predictions are blended.
  - All tiles share the same sigmas and noise, so no overlap is frozen and no seam is repaired afterwards.
  - Each tile keeps its global frame positions, so it predicts what the full frame would there.
  - Masks and core's Apply MiniMax H3 Fun ControlNet are split per tile; the audio is whole in every tile.
  - Use it with any H3 sampler, or with H3 Split Upscale with `spatial_split_param` unconnected.
  - It costs more time than one full-frame call (the report gives the factor) and bounds memory to one tile.
  - Not with a block cache (First Block Cache, T8) or the Mask-Aware ControlNet: the node says so and stops.
  - Kill switch: `MMX_TILED_DIFFUSION=off`.
- **H3 TST Attention (experimental)** is a training-free correction against temporal flicker and small faces that
  break up (Spectral Transport Homeostasis, [arXiv 2609.08505](https://arxiv.org/abs/2609.08505)).
  - For every video attention call it measures, per head, whether the frames over-mix or fragment, and scales the
    video queries to pull them back; strongest in deep layers and early steps.
  - `tau` 0 passes the model through (the A/B baseline); 0.1-0.3 is the usual range.
  - Put it after other attention patches. One summary line per sampling run in the console shows how many calls it
    corrected and by how much. Kill switch: `MMX_TST=off`.
  - Upstream calls its effect limited; it is unmeasured on H3 here.

- **H3 Split Upscale** (samples each tile to completion, then stitches):
  - Odd latent sizes (720p has 45 latent rows, 2160p 135) are now sampled on the DiT's 2x2 patch grid: the last
    tile no longer starts on an odd row, and an odd-sized edge tile is padded by repeating its edge, with the pad
    frozen. Before, H3 padded such tiles circularly, so a tile's top row shared a patch with its bottom row.
  - New options on H3 Spatial Split Params (advanced, off by default, so saved workflows behave as before):
    `brightness_match` (per frame), `dynamic_fade` (narrowing / widening over the steps), `overlap_blend`
    (linear / smoothstep / midpoint), `masked_area_noise`, and equal tiles with `grid_rows` / `grid_cols`.

## Sigma Presets (official templates)

**Sigma Presets** gives the sigma schedule, sampler, cfg and steps of a model as ComfyUI's official workflow
templates ship them, with a graph and a how-to.
- Models: MiniMax H3 Ref2VA / FL2VA (both with their turbo LoRAs) and FastH3; Wan 2.1, Wan 2.2 (lightx2v), Wan
  Animate 2; Flux.1 dev / schnell, Flux.2 dev / Klein; SDXL, SDXL Turbo, SD 1.5, SD 3.5; LTX-2 / 2.3 / 2.5 (each
  stage); HunyuanVideo 1 / 1.5; Qwen-Image; Z-Image / Turbo; Chroma; HiDream.
- Each preset names the template it comes from (comfyui-workflow-templates-json 0.1.104), the model's trained shift
  from core, and any speed LoRA or distilled model its step count needs.
- The curves are computed by ComfyUI's own scheduler code; connect your model to use its own shift. For H3 the graph
  also draws the audio clock.
- Outputs: `sigmas`, `sampler` (for SamplerCustom / SamplerCustomAdvanced), `cfg`, `steps`, the `graph` and the
  `how_to` text.

## Dual-clock shim (P4 — not shipped)

On ComfyUI core `180060c` and later, `ModelSamplingAV` is native (`comfy/model_sampling.py`). A dual-clock sampler shim is a no-op — use core **MiniMaxH3SigmaShift** or our **MiniMaxH3_SigmaShiftLocked** for ratio-locked shifts.

## Workflow

**Task workflows** (A16): 14 ready-to-load graphs in `workflows/tasks/`, one per task:
- text / image to video, reference to video with guides, reference image editing;
- masked character replacement (zero offset, bit-exact outside the mask), masked ControlNet, exact protected edits;
- stable high-resolution crops;
- long-form continuation (from text, editing a long plate, following a long soundtrack);
- face / head replacement, whole-frame object / background edits, tiled enhancement;
- latent persistence.

Each graph carries a guide note (models, variants, inputs, controls, output, limits) and titled stage groups.
They are built in ComfyUI's own front end and validated by the server, but not run on H3 weights:
`workflows/tasks/README.md`.

Older pipelines: `workflows/h3_masked_face_pipeline.json` and `workflows/README.md` (runtime untested without H3
weights).

Recommended sampling chain after N4:

`N4` → `P10 ProtectedLayerGuard` → `P11 AcceleratorConflict` → `P1 SigmaShiftLocked` → `P3 LegalScheduler` → sampler

## Tests

The test suite stays on the developer's machine and is not in this repository (owner rule 2026-10-23). Where it
exists locally:

```bash
cd ComfyUI-MiniMaxSuite
python run_tests.py
```

One test file per `mmx_nodes/` and `mmx_utils/` module. CPU-only; no weights required. CI boots ComfyUI with this
pack and fails on an import failure, no registered node or a traceback from the pack (`.github/c2c_boot_check.py`).

## Licence

Ported utilities carry upstream licence headers and `# PORTED FROM:` provenance lines.
