# Changelog

## Unreleased

### Fixed (masked replacement audit L8.28, 2026-11-26)
- **H3 Mask Prep / mask-to-token / Identity Fusion put mask frames in the wrong latent step.** The frame grouping
  restarted H3's 1,4,4,4,4 cycle at frame 5. Core counts it from frame 0: 0 | 1-4 | 5-8 | 9-12 | 13-16 | 17 | ...
  As a result, 4 of every 5 latent steps took mask frames that were up to 4 frames off, so a moving subject was
  regenerated in the wrong frames. The grid now matches core exactly (tested against core's own spans).
- **Per-Frame Denoise dropped a spatial noise mask.** The [1,1,T,h,w] mask from Set AV Noise Mask / Hard Keyframes
  was replaced by a whole-frame strength, so the background was regenerated too. It is now multiplied in.
  A mask that does not fit the latent is a plain error. Strength per latent step is now the mean over that step's
  frames on H3's grid.
- **H3 Masked Replace:**
  - A canvas-pixel mask is now reduced on H3's grid (max per latent step and per 32 px token). Before, core shrank
    it trilinearly, so thin strands vanished.
  - Crops shorter than the latent are now refused. Before, they were padded with the template's opening frames,
    with only a warning in the report.
- The pixel-frame count of a latent, used in messages, was wrong above 2 steps. It is now computed on H3's grid.
- Mask To Latent Space's report gives `grow_tokens` in the real pixel size.
- The six character-replacement workflows (`workflows/character_replacement/`) now set H3 Subject Uncrop to `protect = exact outside mask`. They were saved before that setting existed, so they loaded with "feathered box", where the model's background replaced the plate inside the crop box. Every pixel outside the head mask now stays the original, bit-exact.

### Added (2026-11-26)
- `tools/h3_vae_drift.py`: measures the H3 video VAE round trip on your footage. Reports colour shift per channel (red drift), softness, chunk-seam dips and out-of-range values over 1, 3 and 5 rounds, for several VAEs side by side (e.g. the original and H3-MT-VAE), and what colour_lock removes. Run it on a GPU: on a 16 GB CPU box the H3 VAE cannot finish one small encode.
- **H3 Open Regen (Regenerate + IC)** (`MiniMaxH3_OpenRegen`): one node regenerates a draft video. Draft frames and
  fps in; finished frames, soundtrack, refined latent and a report out.
  - `mode` switches between rdh9616's Regenerate recipe and slmonker's 2K IC recipe. Only the chosen recipe's inputs
    are shown.
  - Any size and aspect ratio on the 32 px grid, through `size_by` (megapixels / width / height / scale), `aspect`
    (source, presets, custom) and `fit`. The default keeps the draft's own ratio at about 3.7 MP.
  - The draft is encoded at its own ratio (768p area, nearest 32 px pair).
  - Two upscale passes run above 2K. The refine uses core's own sampler.
  - Decoding uses the core H3 VAE (default) or the fast decoder.
  - `colour_lock` (default on) matches each frame's colour and broad contrast back to the draft. Nothing is clamped.
- The 9 Regenerate STD and 5 Regenerate IC nodes moved to the menu category `MiniMax H3/DEPR`, with names prefixed
  `DEPR ·`. They stay registered, so saved workflows load unchanged.
- Ported from mo-akkakk's ComfyUI-MiniMaxH3-TemporalWindow-SpatialTiles 1.7.0 (MIT):
  - **Minimax H3 V2V Temporal Window & Spatial Tiles** (`MiniMaxH3_V2VWindows`): long and high-resolution V2V in
    windows and tiles. `prefix` carry puts the previous window's last frames into the next window, frozen, so the
    hand-off is exact.
  - **Minimax H3 Long Generation** (`MiniMaxH3_LongGeneration`).
  - **Minimax H3 Multi Guide** (`MiniMaxH3_MultiGuide`), with per-guide `until_step`.
  - **Minimax H3 Reference to Video HiRes** (`MiniMaxH3_ReferenceToVideoHiRes`): reference videos given to the model
    at the generation's size.
  - Upstream's own test script runs against the Suite copy (0 failures on core 0.46.3).
  - Suite addition: optional `start_frames` / `start_audio` (plus `vae` / `audio_vae`) on V2V and Long Generation.
    These frames are frozen at the start of the video, kept exactly, and continued from. Use it to extend a previous
    clip or to pin an exact first frame.

### Added (Open Regen: one pipeline, 2026-11-25)
- The H3 Open Regen nodes (rdh9616's Open-H3 Regenerate) now also run slmonker's Open-H3-Regenerate-2K-IC recipe, so
  STD and IC are one pipeline. New inputs, appended at the end with defaults that keep today's results:
  - **LoRA:** `speed_lora` / `speed_strength` and `sharp_lora` / `sharp_strength`, applied in the IC order
    (speed -> LMS Sharp -> `lora_name`). `lora_name` also takes the IC LoRA with Diffusers keys or ComfyUI keys.
  - **Sigmas:** `grid` = IC up3 / up4 / hf8 / custom, plus `custom_sigmas`.
  - **Prepare Draft:** `fps_map` = floor (the IC preprocessing), plus `min_real_frames` (IC: 113).
  - **Refine Latent:** `reference_latent` (the Reference to Video LATENT: the IC target, audio pinned), the
    `audio_mode` option "silence asset file (IC, 124 frames)", and `silence_file`.
- New example `workflows/h3_open_regen_ic_draft_to_2k_unified.json`: the IC draft-to-2K graph built on the Open Regen
  nodes, with Finish Frames.
- The five `MiniMaxH3_OpenRegenIC*` nodes stay registered for saved workflows. Their descriptions now point to the
  Open Regen IC options.
- The shipped STD and HDR example workflows save the new widgets with their default values. Their results are
  unchanged.

### Fixed (L9.290)
- Node panels refill when ComfyUI serves a node from its cache. A re-queue with unchanged inputs used to leave the
  window plan, seam drift, token grid, crop plan, drift curve, box trajectory and every text report empty after a
  page reload or tab switch, because core sends a cached node's result only for output nodes. The 44 non-output
  nodes that show a result now declare core's `has_intermediate_output` flag. Core then resends their last result at
  the end of a run that finishes, without running them again; an unwired node still never runs. The status strip
  says "cached" and then "cached · last result", and it explains when a stopped run left a panel empty. Node ids,
  inputs, outputs and defaults are unchanged. Design: docs/cached_node_panels_design.md.

### Fixed (L9.312)
- H3 Per-Frame Denoise: an incoming noise mask is now multiplied by the per-frame strength once, as its tooltip says.
  It was applied twice (mask x mask x strength), so a 0.5 mask with strength 0.5 gave 0.125 instead of 0.25. Saved
  workflows that feed this node a latent which already has a noise mask now get a weaker mask reduction (higher
  values) than before; without an incoming mask nothing changes.
- H3 Strongest Pose: `gap_fill` now works. A low-confidence gap longer than `gap_fill` frames (between two
  confident detections) keeps the detector's own x/y instead of being interpolated; 0 interpolates every gap. Before,
  every gap was interpolated whatever the value. Only `keypoints_json` changes (for points with score 0 inside such
  gaps); the hints image is unchanged because those points are not drawn.
- H3 Region Mask (painted mode): each frame now uses its own frame of `painted_mask`. Before, the first painted
  frame was applied to every frame. Saved workflows with a multi-frame painted mask change from frame 2 on; a
  single-frame mask behaves as before.
- H3 Region Mask: with no depth wired, `sky` is now the top of the frame and `ground` the bottom (a level, centred
  horizon, as the report says). They were swapped. Saved workflows that use sky or ground mode without depth get the
  opposite half from before; with depth wired nothing changes.
- H3 Track + Crop: `canvas_mode` auto_no_downscale now sizes the square canvas from the largest crop (rounded up
  to 32, 128-1344) and auto_capped_768 does the same but stops at 768, as in H3 Face Track Crop. Both always gave
  768x768 before. Saved workflows that use an auto mode get a different crop size (output frames and transform
  canvas) unless the largest crop rounds to 768.
- H3 Color Round-Trip QC: every frame is now checked, not only the first. `max_error` and `passed` cover the whole
  clip, `error_map` has one frame per input frame, the report names the frame with the largest error, and the
  "clamped to SDR" check reads the peak of the whole clip (a dark first frame in an HDR clip no longer fails it).
  Saved workflows with multi-frame input can now fail where they passed, and `error_map` returns more frames. Each
  extra frame costs one more OCIO round trip on the CPU.
- MiniMax H3 Vision: `REF_IMAGES` is now a list output (one IMAGE per reference), as in the original Promptor node.
  Before, it handed the whole Python list to the next node as a single image, which image nodes cannot use. A node
  wired to it now runs once per reference image unless it takes lists.
- Spectrum H3 Objective Media Capture (Sequential) and Capture Reset: they now re-run when the in-memory capture
  store they act on has changed, and are cached when it has not. Before, a Capture Reset queued again after a new
  capture could be served from cache and release nothing.

## 1.2.0 - 2026-11-23

### Fixed (replacement, round 2 - docs/PLAN_h3_replacement_quality.md s9)
- The six character-replacement workflows put the plate (`<Video 1>`) on the target grid: node 15 migrated in place
  to H3 Image + Reference to Video (same node ids); new `ref_video_fit` choice *target grid for <Video 1>, core canvas
  for the rest* keeps a driving clip on core's canvas (R5).
- H3 Temporal Subject Tracker: `masks` carries accepted masks only (T5); a precomputed track no longer loops at a
  review (T4); `reid_interval` 12 for new nodes and an appended `jump_2d` loss rule, off (T6); refuses a pick made in
  a window that does not start at frame 0 when the Selector gets the new `window_json` (T1).
- H3 Face Mask SAM: the report names rectangle-fallback frames; appended `sam_fallback` hold option (T7).
- H3 Subject Memory: appended `track_json` cuts approved frames to the subject (R3).
- H3 Identity Fusion Controller: stays on for the positive half under batched CFG (R7).
- Anchor Studio / Tracker: frames go to numpy through `.cpu()` (device scan L8.33).

### Added
- **H3 Prompt Writer** node (`MiniMaxH3_H3PromptWriter`, category `MiniMax H3/Prompt`; design
  `docs/PLAN_h3_prompt_writer_node.md`): the H3 Prompt Writer extension's prompt logic (MIT, ported) as a workflow node
  with image / video / image-sequence / audio sockets, eight modes, and any provider - offline template, local Hugging
  Face transformers (any repo id), local GGUF (child process), Ollama, OpenAI-compatible, OpenAI, Anthropic, Gemini.
  Official-format audit plus one repair pass. Prompt, negative and report outputs. Example workflow
  `workflows/h3_prompt_writer_reference.json`.
- Prompt Writer v2: its description points to the new node (no input change; saved workflows load unchanged).
- **Viggle Remake on H3** (category `MiniMax H3/Replace`; design `docs/PLAN_h3_replacement_quality.md`): H3 Remake
  Conditioning (Viggle), H3 Remake Keyframe Until, H3 Remake Noise, H3 Remake Prefix - ported from the reference
  nodes (Apache-2.0, CREDITS.md) - and **H3 Remake Replace**, the adapter from an approved anchor + a plate window to
  Remake conditioning in the window's own framing (zero offset), chunked by the Job Controller (anchor pinned on the
  first window, the accepted context held clean on later ones), with character, face / head and item / prop modes.
  Experimental: not run on H3 weights or the Remake LoRA.
- Workflows `workflows/replace/remake_character_replace.json`, `remake_head_face_replace.json`,
  `remake_item_replace.json` (job-chunked; Plate Restore puts the literal plate back outside the mask).
- `tools/h3_replace_bench.py`: re-entry identity drop and multi-shot identity spread (`--shots`, manifest
  `"cases"`), with the measure named (ArcFace when installed, otherwise masked colour histogram).
- H3 Image + Reference to Video: `image_batches` (appended, default `first image only` = unchanged) - `every image`
  makes each image of a reference batch its own `<Picture N>`; the report now says when images were left out.

### Fixed
- **Zero offset on V2V** (owner order 2026-11-23; `docs/PLAN_h3_zero_offset.md`, bench `tools/h3_zero_offset_bench.py`):
  - Subject Uncrop / Plate Restore / Stitch Back: a decode that the H3 VAE cut to a multiple of 16 is put back at
    core's offset with the margins from the plate instead of being stretched into the box (5.02 px -> 0.0005 px).
    Subject Crop records the crop size in each box (`metadata.mmx_crop_size`) and says so in its report.
  - Lanczos resizes (Subject Crop / Uncrop, Source AV bridge, Motion Context, AV noise mask) run in float: core's went
    through 8-bit PIL (HDR clipped at 1.0, 256 levels; 129.5/255 -> 7/255 on an HDR plate).
  - Mask Prep and Mask To Latent Space reduce masks that are off the 16 px grid on the cells the VAE actually encodes
    (182 / 118 masked cells left "keep" -> 0). On-grid masks are unchanged.
  - Detail Reinject combines the generated low band with the plate's high band (the detail was counted twice: an
    unchanged region came back 8.8/255 different and 0.43 px moved -> 0.025/255, 0 px).
  - Source AV Latent, Masked Replace and the native audio lock pad the soundtrack to whole 1/40 s ticks before the
    audio VAE: core cut up to 12.5 ms off the head (8.3 ms for 121 frames).
  - The OpenCV fast paths of mask dilate / grow move a mask to the CPU before `.numpy()`.
  - Subject Crop: appended optional `bboxes` / `bboxes_first_frame` - crop at given boxes, one plan for a whole job
    (each window planned on its own moved the crop between segments).
  - Face Stitch no longer fades the face out for one missed detection (the old face ghosted through ~9 frames).
  - Temporal Subject Tracker keeps the soft mask edge until the plate-size matte (4K edge error 8.0 -> 2.6 px).
  - Matte Chatter's dropout bridge uses 1 byte per cell instead of 9 (bit-exact).
  - H3 Frame Handles: appended `handles` output + optional `undo_handles` input - a second Frame Handles after the
    model removes exactly the frames and sound the first one added (L8.56). The full-spine and masked-face workflows
    now trim back before Create Video: the delivered clip starts on the plate's first frame.
  - H3 Remake resizes references in float (no 8-bit rounding, HDR above 1.0 kept) (L8.57).
- H3 Subject Memory: the approved anchor (H3 Anchor Approval) is now in its `anchors` output, as Approval promised,
  without routing Approval's bank (R2).
- H3 Identity Fusion Controller: a pixel-frame mask is reduced with H3's own frame groups, so a frame where the head
  enters or leaves is no longer dropped (R6).
- H3 Temporal Subject Tracker: refuses a subject picked on a frame other than 0 instead of putting that mask on frame
  0 and tracking the wrong spot (T1).
- H3 Face Track + Crop: with an identity anchor, a lone face that is not the performer (an extra while the performer
  is out of frame) and a first-frame face below `identity_threshold` are no longer taken; a connected
  `identity_reference` that cannot run (insightface missing) is an error instead of silently ignored; the report
  says whether identity was on (T2, T3).
- Compact-node layout engine: a node removed right after it was created (paste then undo, a script that adds and
  removes) is no longer kept alive by the engine (about 185 KB each, measured in CNP). Same fix as CNP's engine (L9.01).

## 1.1.0 - 2026-11-17

### Added
- **HDR** (owner order A29; category `MiniMax H3/HDR`, 18 nodes; how-to `docs/HDR_HOWTO.md`, workflows
  `workflows/hdr/`):
  - `mmx_utils/hdr_math.py`: the single source of truth for every HDR encode / decode (ported from
    ComfyUI-H3HDR-Test; default `diffhdr_log_16384`), plus sRGB, PQ, gamut (Rec.709 / Rec.2020 / ACEScg), the
    editable VAE bias table, soft-shoulder preview, clip masks and the guided gain-map join.
  - Inference and I/O: HDR Log Encode / Decode, Bias Correct, Gamut Convert, Clip Mask, Gain Map Join, HDR Prompt,
    Tonemap Preview, Load / Save EXR Sequence (half or float, ZIP, never clamped), Save Log 16-bit (PNG / TIFF),
    Save HDR10 (HEVC 10-bit, PQ, BT.2020, mastering metadata), Eval Report.
  - Dataset and training: Fetch Poly Haven, Fetch Hugging Face (size cap), HDRI -> Clips, Make Pairs, Train IC-LoRA
    (ai-toolkit `minimax_h3_ref2va`, SDR as reference video, loss on the target only). They run as background jobs
    with progress, cancel and skip-what-is-done; a jobs panel shows the log and the loss curve.
  - Workflows: (a) SDR -> HDR EXR, (b) + Regenerate 4K gain-map join, (c) dataset build, (d) training, (e) eval.
- **Open-H3 Regenerate STD** (9 nodes) and **Open-H3 Regenerate 2K IC** (5 nodes), two distinct pipelines.
- `__version__`, `pyproject.toml`, this changelog.

### Changed
- H3 Latent Upscale (Model) accepts a plain H3 video latent (from VAE Encode).

### Tested / not tested
- Tested here on synthetic data (CPU): log round trip < 1e-4 relative from 2^-10 to 16384, EXR keeps values > 1 (cv2
  and OpenImageIO), bias table, gamut, PQ / HDR10 tags, gain-map join, job cancel / resume / skip, workflow wiring.
- Not tested here: the real H3 model, the HDR LoRA, a GPU, real downloads, a real training run; picture quality is
  not verified.

### Compatibility
- No existing node was renamed or removed; saved workflows load unchanged. 196 nodes registered.

## 1.0.0 - before 2026-11-16

The Suite as it stood before versioning (164 nodes), unversioned.
