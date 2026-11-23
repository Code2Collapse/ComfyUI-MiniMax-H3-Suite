# H3 replacement quality: audit, the Remake path, and the fixes

Status: design note, 2026-11-23 (owner order: "the nodes related to character-replacement or face/head replacement or
item replacement is really bad"). Written before the code, as the repo requires. Branch `replace-quality`.

**What this machine can and cannot show.** There are no H3 weights here and no Viggle Remake LoRA. Everything below is
read from the code, checked against ComfyUI core 0.39 source (`comfy/ldm/minimax/model.py`, `comfy/model_base.py`,
`comfy/text_encoders/minimax.py`, `comfy_extras/nodes_minimax_h3.py`) and unit-tested on CPU with stand-in CLIP/VAE.
**Image quality is NOT VERIFIED.** New behaviour ships opt-in or in new nodes marked experimental. The bench cases in
section 6 are how the owner's GPU box measures it.

## 1. How a VFX artist hits the problems

The artist's job: one person (or a head, or a prop) in a plate becomes someone/something else, the rest of the plate
is untouched, identity holds for the whole take, including when the subject leaves frame and comes back, and long
takes are rendered in chunks without seams.

The Suite has every part (Subject Crop/Uncrop, Image+Reference to Video, Identity Fusion, Selector, Re-ID, Tracker,
Anchor Studio/Approval, Prop Checker, Plate Restore, Face Stitch/Refine, the Job Controller, six workflows), but the
parts do not add up to that job. The defects, worst first. Severity: P1 = wrong or silently degraded output on a
normal path; P2 = degraded in a common case; P3 = edge case or a misleading report.

### 1a. Identity and references (conditioning side - fixed on this branch unless marked)

| ID | Sev | Where (file:line at 75db635) | Defect | Cause | Fix |
|---|---|---|---|---|---|
| R1 | P1 | `mmx_nodes/image_and_reference.py:224` (also core `nodes_minimax_h3.py:308`) | A reference **batch** silently becomes one picture. H3 Reference Bank `identity` (`h3_job_nodes.py:1237`) and H3 Subject Memory `anchors` (`:1345`) are IMAGE batches; wired into a `ref_image_N` slot, only image 0 reaches the model. Face + profile + outfit, or identity + approved frames: H3 sees one. | `img[:1]` per slot; the report says nothing. | New optional input `image_batches` (appended last, default `first image only` = old behaviour): `every image` turns each image of a batch into its own `<Picture N>`. The report now always says when images were dropped. |
| R2 | P1 | `mmx_nodes/h3_job_nodes.py:1316-1345` | Subject Memory's `anchors` never contains the **approved anchor**, although Anchor Approval's description promises "Subject Memory and every segment use it". The anchor reaches the model only if the user also routes Approval's `bank`. | `subject_memory()` lists approved anchors in the JSON (`:1292`) but `execute` only loads bank identity + examples. | Approved anchor images (deduplicated against the bank) are loaded right after the bank identity images. |
| R3 | P2 | `mmx_nodes/h3_job_nodes.py:1274,1343` | Memory "examples" are whole generated frames (other people, background) fed back as identity: drift compounds segment by segment. | Last frame of each accepted segment, uncropped. | Report-level for now: the notes say examples are whole frames. A subject-cropped example needs the segment's matte; listed as follow-up (section 8). |
| R4 | P1 | `workflows/tasks/16_character_swap_job.json` (subgraph node 12) | The approved anchor is a **full plate frame** while the generation runs in **crop space**: `<Picture 1>` shows the new character small, in a different framing from `<Video 1>` (the crop). Fewer identity tokens on the face; the model is shown two framings of one subject. | Wiring. | The new Remake workflows crop the anchor to the same window as the plate (`MiniMaxH3_RemakeReplace` `bboxes` input). Task 16 is left as is (its own subgraph); see section 5. |
| R5 | P1 | `workflows/character_replacement/1..6` node 15 | The cropped plate goes in as `<Video 1>` through **core** Reference to Video, which re-sizes it to its own 768-short-edge canvas: when the crop is not that canvas the plate reaches the model **offset by up to half a token** against the target (the A16 offset; `H3 Image + Reference to Video` reports e.g. -14 px for 1440x1440). The task workflows already use the Suite node with `target grid (zero offset)`; the six did not. | Wiring. | Documented in `docs/character_replacement.md`; the new workflows (section 5) put `<Video 1>` on the target canvas by construction (Remake: source canvas == target canvas). Rewriting the six graphs is deferred - they are reference examples validated in the live front end and a JSON rewrite cannot be re-validated there in this task. |
| R6 | P3 | `mmx_utils/identity_fusion.py:79` | A **pixel-frame** mask (124 frames) given to Identity Fusion is sub-sampled with `linspace` to the latent frames: a frame where the head enters or leaves can be dropped, so identity is off on those latent rows. | `linspace(...).round()` instead of H3's 1-4-4-4-4 groups. | When the mask length is a 17n+5 count whose latent length matches, it is reduced with the H3 frame groups (max). Latent-space masks are unchanged. |
| R7 | P3 | `mmx_nodes/identity_fusion.py:95` | Under real CFG (cond+uncond batched) the controller switches itself off for the whole batch. | `0 not in cou or 1 in cou`. | Report only (H3 runs CFG-free; changing it needs a per-row bias split). |

### 1b. Picking and tracking the right subject (identity side - fixed on this branch unless marked)

| ID | Sev | Where | Defect | Cause | Fix |
|---|---|---|---|---|---|
| T1 | P1 | `mmx_nodes/subject_track.py:726` | Temporal Subject Tracker always seeds the Selector's mask at **frame 0**, ignoring the frame the Selector picked on (`subject["frame"]`, `:213`). Pick the performer on frame 40 and the frame-40 mask is placed on frame 0; SAM 3 then tracks whatever is under it. | `t.reseed(0, seed0)`. | Refuse with a plain message when `subject.frame != 0` (tracking backward from a later seed is not implemented; trim the clip or pick on frame 0). |
| T2 | P1 | `mmx_utils/face_refine.py:313-322` | H3 Face Track + Crop accepts a **single detected face with no identity check** even when an identity reference is connected, and on the first frame takes the best match **without** `identity_threshold`. The performer leaves frame, an extra is the only face: the track jumps to the extra. | Branches `len(boxes)==1` and `last is None` skip the threshold. | With an identity anchor: a single face is accepted only if it matches (`>= identity_threshold`); the first-frame pick also needs the threshold; otherwise the frame counts as not found (gap-filled like any miss). |
| T3 | P1 | `mmx_utils/face_refine.py:299-300` | If InsightFace cannot load, identity tracking silently falls back to size/position; the report does not say so. | `print` only. | The node raises when `identity_reference` is connected but identity cannot run (a reference that is ignored is worse than an error). |
| T4 | P1 | `mmx_nodes/subject_track.py:743-745` | With backend = masks input, a bad frame at the cursor deadlocks the review loop. | `track_fn` ignores the confirmed seed. | Not fixed here (review-loop state machine; needs its own test plan) - follow-up. |
| T5 | P1 | `mmx_nodes/subject_track.py:498-501` | Tracker `masks` output carries frames rejected as "another person". | Raw window output. | Follow-up; the `matte` output (used by the workflows) is correct. |
| T6 | P2 | `mmx_nodes/subject_track.py:652`, `mmx_utils/subject_track.py:220` | Drift check off by default; loss rule horizontal only - two people crossing at similar x swap silently. | Defaults. | Follow-up (changing a default alters saved workflows; needs an appended option). |
| T7 | P2 | `mmx_utils/face_refine.py:748-759` | SAM failure frames fall back to a hard rectangle: seam pop. | Fallback order. | Follow-up (Face Mask SAM). |

### 1c. The backend itself

| ID | Sev | Defect | Fix |
|---|---|---|---|
| B1 | P1 | Every Suite replacement path asks **plain ref2va** to replace a subject from text + a reference picture, i.e. to infer what to change. Viggle Remake is a pair of LoRAs on H3 trained exactly for "source clip + painted keyframe(s) -> the same clip with the change", and Anchor Studio already makes the painted keyframe it needs. The Suite had no way to use it. | Port Remake (section 2) and add the adapter (section 3). |
| B2 | P1 | Long takes: the replacement paths continue segments through Motion Context; Remake's own continuation (prefix rows held clean) was not available. | `MiniMaxH3_RemakePrefix` + the adapter uses the Job Controller's `context_images` as the prefix. |

### 1d. Geometry defects seen during the audit (NOT edited here - owned by the zero-offset task)

| ID | Where | Defect |
|---|---|---|
| G1 | `mmx_nodes/subject_crop.py` (no external box input) | In a job, every segment runs Subject Crop on its own window: the crop window **changes at every segment boundary**, so the continuation context (previous segment, previous crop) is pinned at a different position from the new crop. A pop at each segment start. Needs a job-wide crop plan (Subject Crop accepting `bboxes`, or a plan computed once from the whole matte). |
| G2 | `mmx_utils/face_refine.py:462,677-681` | Face Stitch `fade_out` opacity is a smoothing of `detected`: one missed detection lowers opacity for ~9 frames and the original face ghosts through. |
| G3 | `mmx_nodes/subject_track.py:577-580` | Plate-size matte = binary proxy mask upscaled and thresholded: ~4 px staircase edges at 4K. |
| G4 | `mmx_utils/body_double.py:275,285` | Matte Chatter without bboxes allocates int64 T x H x W (2 GB at 124 frames 1080p, >8 GB at 4K). |

## 2. Viggle Remake, ported (new nodes, category `MiniMax H3/Replace`)

Source: `third_party/ComfyUI-Viggle-Remake/nodes.py`, SPDX Apache-2.0 header (no LICENSE file in the folder; the SPDX
line is the licence statement). Apache-2.0 -> GPL-3.0 is permitted. Ported faithfully as V3 `io.ComfyNode`, with a
`# PORTED FROM:` header and CREDITS.md / THIRD_PARTY_LICENSES entries. No runtime import of third_party.

| Node | What it does (unchanged from the reference) | Additions |
|---|---|---|
| `MiniMaxH3_RemakeConditioning` | Source clip as `<Video 1>` (labels + `<T.T seconds>` stamps, **text-only** presentation, each segment tokenized on its own), refs as `<Picture N>`, optional `pins` (one output frame per ref -> keyframes), source soundtrack as **clean target audio** (noise_mask 0), Viggle's canvas policy (as is when short edge = 480/736 and sides /32, else cover onto 832x480 / 1280x736). | `report` output (appended): canvas, refs, pins, audio. |
| `MiniMaxH3_RemakeKeyframeUntil` | DIFFUSION_MODEL wrapper: drops the keyframes once sigma < `keyframe_until`; rebuilds `cond_video_latents` from the refs, layout rebuilt by the model. | - |
| `MiniMaxH3_RemakeNoise` | One CPU generator; draws once per ref/keyframe latent before the target noise, so a seed gives the reference script's video. | - |
| `MiniMaxH3_RemakePrefix` | Previous take's frames open this render, latents clean with noise_mask 0. | - |

Verified against core 0.39: `minimax_refs` / `minimax_keyframes` keys and block fields (`model_base.py:2199-2227`),
`PackedLayout` order (keyframe `cond` rows, then refs as `ref_img` in list order, `model.py:351-471`), wrapper call
signature (`model.py:625-631`), `comfy.audio.resample`, `node_helpers.conditioning_set_values`, the `qwen3vl_32b`
token key.

The LoRA is loaded with core **LoraLoaderBypassModelOnly** (the reference: a merge into bf16 rounds part of the update
away). The LoRA file name and HF repo are NOT VERIFIED here (no network check); the workflows name a placeholder.

## 3. The adapter: `MiniMaxH3_RemakeReplace` (digest 4.5)

"Approved anchor + plate window -> Remake conditioning, chunked by the Job Controller", one node, three modes on the
same path.

Inputs: `clip`, `vae`, `audio_vae`, `window` (IMAGE, the segment's plate frames: Job Controller `window` -> Frame
Window, or Subject Crop's crops), `anchor` (IMAGE, Anchor Approval's `anchor`: frame 0 of the take repainted), `mode`
(`character`, `face / head`, `item / prop`), `prompt` (empty = a six-section prompt written for the mode),
`subject` (text used by the default prompt), `short_edge` (480 / 736), optional `mask` (per frame; required by face /
head and item / prop), `references` (extra identity pictures, each its own `<Picture N>`, max 4 pictures in total -
Remake's trained range), `context_images` (Job Controller: the previous accepted frames), `audio`, `bboxes` (Subject
Crop's boxes when `window` is a crop: the plate-sized anchor is cut with the box of `anchor_frame`), `anchor_frame`
(window frame the anchor repaints, default 0), `hold_outside_mask` (experimental, off).

Rules, each one a test:
1. **Canvas = the window's own framing**: short edge -> `short_edge`, each side rounded to a multiple of 32, a pure
   stretch (no crop). Source and target share the canvas, so `<Video 1>` lands token-for-token on the target (zero
   offset) and the decoded frames map back to the window by the inverse stretch. Viggle's own 16:9 cover crop would
   cut the window's sides and shift the composite; it stays available in `MiniMaxH3_RemakeConditioning` only.
2. **First window** (no usable context): the anchor is `<Picture 1>` **and** pinned as a keyframe at `anchor_frame`
   (Viggle's "painted keyframe"). **Later windows**: the previous accepted tail opens the render as clean prefix rows
   (RemakePrefix); the anchor stays `<Picture 1>`, unpinned (its frame is not in this window). The prefix length must
   be a multiple of 17 (Task Spec context 17, 34, ...); anything else is refused in words, because the window's first
   `context` frames ARE those accepted frames and a shorter prefix would regenerate accepted frames. When `window`
   is a crop, the Job Controller's plate-sized context frames are cut with THIS window's boxes (frame t with box t),
   so a crop window that moved since the previous segment (G1) still lines up for the prefix.
3. **Identity**: anchor first, then `references` (Reference Bank `identity`, Subject Memory `anchors`) - every image
   of a batch is its own picture (R1 does not apply here), duplicates of the anchor dropped, capped at 4 with a report
   line when pictures were left out.
4. **Mask modes**: the mask is resized to the canvas with the window and returned as `edit_mask` plus a full-window
   `bboxes`, so **H3 Plate Restore** (unchanged) puts the decoded frames back with registration, colour fit and the
   literal plate outside the mask. With `hold_outside_mask` on, the target latent is the encoded window and its
   noise_mask is the H3-token mask (1-4-4-4-4 frame groups, 2x2 patch max): the model is told to keep the plate
   outside. Off by default because Remake was trained to regenerate the whole target (experimental).
5. **Audio**: the window's soundtrack is held clean (Remake's own rule), so replacement never re-voices the plate.
6. Report: canvas, stretch, pins, prefix, pictures, mask coverage, what was dropped.

Outputs: `positive`, `latent`, `plate_canvas` (the window on the canvas), `edit_mask`, `bboxes`, `report`.

## 4. Fixes on this branch (section 1 IDs)

R1, R2, R6, T1, T2, T3. All optional/appended or refusals of silently wrong output; no saved workflow changes output
unless it was producing a wrong result (T1/T3 now stop with a message instead).

## 5. Workflows

New, in `workflows/replace/` (the six old ones and the task workflows stay as they are):
- `remake_character_replace.json` - whole-window character swap, Job Controller chunked, anchor pinned on window 1,
  prefix on later windows, Plate Restore, QC Gate, Commit.
- `remake_head_face_replace.json` - head mask from the Tracker matte, Subject Crop window, anchor cut with the crop box,
  mode face / head, Plate Restore with the mask.
- `remake_item_replace.json` - item mask, mode item / prop, Prop Consistency Checker on the references.

Each has a note: what to load, what is NOT VERIFIED, which LoRA (placeholder name).

## 6. Bench (tools/h3_replace_bench.py)

New `score` options and report columns for the two failure modes Viggle itself names:
- **re-entry**: `--reentry` (auto from the mask: frames where the subject is absent then present again) scores
  identity before vs after each re-entry (ArcFace cosine when insightface + buffalo_l are present; otherwise the
  colour-histogram similarity of the masked subject, labelled as such) -> `reentry_drop`.
- **multi-shot**: `--shots 0,124,248` (or auto from the plate's cuts) scores identity per shot and the worst
  shot-to-shot change -> `shot_spread`.
The manifest gains `"cases": ["reentry", "multishot"]`. Both run on CPU against saved videos.

## 7. Tests (CPU, stand-ins)

`tests/test_remake.py`: stamps against the reference values; canvas policy (as-is / cover); tokens are one segment
per label; pins -> keyframes at the right frames, first stretched, rest cover; audio padding to 800-sample frames;
latent shapes and noise masks; KeyframeUntil drops keyframes below the threshold and rebuilds cond latents from refs;
Noise draws in order (same seed -> same noise); Prefix holds the first 5m latent rows clean; adapter: canvas stretch,
first-window pin vs later-window prefix, refusal of a non-17m context, picture cap and batch expansion, anchor cut by
bbox, mask -> edit_mask, hold_outside_mask token mask, audio held. Plus tests for R1, R2, R6, T1, T2, T3. Tests stay
local (owner rule: never pushed).

## 8. Follow-ups (not done here)

T4-T7, R3 (subject-cropped memory examples), R5 (rewrite the six workflows in the live front end), G1-G4 (geometry
agent), real-weight A/B of Remake vs the spine with the bench on the GPU box.

## 9. Round 2 (branch replace-quality-2, 2026-11-23): the follow-ups

Tests first; each new test failed on origin/main before its fix (18 failed, evidence
`docs/evidence/replace_quality_2026-11-23/rq2_mustfail_before_fixes.txt` in the work area).

| ID | Fix | Saved workflows |
|---|---|---|
| R5 | The six `workflows/character_replacement/` graphs migrated **in place** (same node ids, same links): node 15 core `MiniMaxH3ReferenceToVideo` -> `MiniMaxH3_ImageAndReferenceToVideo`, character `ref_image_1`, plate `ref_video_1`, driving clip `ref_video_2`. New `ref_video_fit` choice (appended to the options, default unchanged): *target grid for `<Video 1>`, core canvas for the rest* - the plate on the target grid (zero offset), a driving clip (another shot) on core's canvas, not cover-cropped. | A copy saved before still loads and runs as before (core's node). |
| T5 | Tracker `masks` output = the accepted masks of the window (from `masks/`); a frame the loss rule rejected, or not decided yet, is empty. | Output content changes only on frames that were wrong. |
| T4 | Backend *masks input*: the cursor frame takes the seed (the mask the user confirmed or re-identification took) instead of the precomputed mask, so a bad precomputed frame cannot fail again at its own seed; the track advances. | Same results where the precomputed track was right (the window seed is that same mask). |
| T7 | Face Mask SAM: the report names the frames that fell back and says *plain RECTANGLE*; new appended `sam_fallback` = *hold nearest SAM mask (experimental)* moves the nearest segmented frame's mask with the face rect. | Default `face rect` = old behaviour. |
| T6 | `reid_interval` default 12 for a NEW node (a saved workflow restores its own value). 2-D jump rule as an appended `jump_2d` (off); its setting enters the track key only when on, so tracks under way keep their key. | Unchanged (saved value; option off). |
| R3 | Subject Memory: appended `track_json` (the Tracker's): each approved frame is cut to the subject's box + 15 %; a frame without the subject is left out and the notes say which. Without it: whole frames, as before. | Unchanged without the input. |
| R7 | Identity Fusion under real CFG with cond + uncond in one batch: the positive chunks get the bias, the negative chunks plain attention (each chunk's `cond_or_uncond` passed to an inner override such as NegPiP). Before: off for the whole batch. | Experimental node; behaviour changes only in batched CFG (where it was silently off). |
| T1+ | Multi-Subject Selector: appended `window_json` (H3 Frame Window's): the pick records `clip_frame` / `source_frame`; the Tracker refuses a pick whose clip frame is not 0 (a pick on row 0 of a window starting at frame 7 is frame 7). Wired in task 16 and the three Remake workflows. | Without the input: as before. |
| L8.33 | Anchor Studio / Tracker: frames reach numpy through `.detach().cpu()`. Both flagged paths already received CPU tensors (`_fit(... .cpu())`, `_Source.read(... .cpu())`), so the new GPU-stand-in tests passed before and after: they pin it. | - |

Still open: T4 for the SAM 3 backend is unchanged (it re-tracks from the seed, no loop); G1-G4 (geometry agent).
