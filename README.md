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
| **H3 Strongest Pose** (N8) | ViTPose-H plate detect → crop warp pose hints. The YOLO + ViTPose ONNX code ships in the pack (`mmx_vendor/wap_pose`); WanAnimatePreprocessV2 is not needed. Weights: `models/detection/vitpose_h_wholebody_model.onnx` (+ `yolov10m.onnx`) |
| **H3 Temporal Depth** (N9) | Depth hints (passthrough default) + optional normals. Model backends download weights only with `allow_download` (last input, off by default) |
| **H3 Drift QC** (N11) | Exterior-only RAFT drift gate |
| **H3 Region Mask** (N10) | Depth-horizon or painted region MASK for N2 spine |
| **H3 Family Presets** (N12) | Six-section prompt (#14) + style DNA (#16) |
| **H3 Edit Validator** (N13) | Pre-flight gate (#8) — mask_required, grid, drift |
| **H3 DCC Bridge** (N14) | External EXR sim ingest (no solver) |
| **H3 Pose Puppeteer** (N15) | Driving pose → motion hints; rig_type pose_only = clean pose for Apply MiniMax H3 Fun ControlNet (other rigs blend the pose over the character plate, no retarget) |
| **H3 HDR Roundtrip** (N16) | OCIO view↔scene with HDR tag propagation |
| **H3 Hybrid HDR LoRA** | Apply extracted HDR LoRA to Ref2VA (constant strength) |
| **H3 Hybrid HDR Extract** | Extract LoRA from Ref2VA/FL2VA/Singularity triple on GPU box |
| **H3 Color QC** | Hasler-Süsstrunk + luma DR + texture/flicker for Hybrid A/B |

N1 is a **planner**, not a detector — wire `subject_mask` and/or `bboxes_json`.

## Long video without loading it whole: H3 Media Probe + H3 Frame Window Reader (A20 Phase 1)

The first media nodes of the A20 plan.

- **H3 Media Probe** reads a video's frame count, frame rate (constant or variable), size, rotation, colour tags and
  audio without decoding the pictures, and hands on a lazy handle.
  - Connect core **Load Video**: upload and file choice stay core's, and its trim and crop are honoured.
  - Or connect a `C2C_VIDEO` handle from CustomNodePacks' Load Video (C2C).
- **H3 Frame Window Reader** decodes only a window: `start`, `count`, plus `lookbehind` / `lookahead` handles for a
  tracker.
  - It checks free memory first; a window that does not fit fails in words, with the count that would.
  - `window_json` (contract `h3.frame_window` v1) gives every row's source frame and exact presentation time,
    variable frame rate included, and which rows are the window itself (`core_slice`; `core_start` / `core_count`).
  - `audio` (2026-11-02) is the sound of exactly the frames it read, handles included. A job window's held head
    frames get silence first. With `step` > 1 it is silence (every Nth frame has no single stretch of sound). Feed it
    to Source AV Latent so the generation hears the plate's own sound, or, for lip sync, the new one.

```
Load Video ──VIDEO──► H3 Media Probe ──media──► H3 Frame Window Reader ──images──► …
                                                                    └─window_json─► (frame map for later stages)
```

**H3 Timeline Mapper** builds the frame ledger (contract `h3.frame_ledger` v1): which source frame every output frame
shows after trim, reverse and frame-rate conversion, numbered from `offset`, plus a planner's segments (each with
its own padding, overlap, context and accepted range). It moves no pixels: images, masks and audio read the same
map, so a reverse or a retime applies to every frame-indexed signal at once. Its `report` shows the first and last
rows and any gap or overlap between accepted ranges.

**H3 Shot Boundary Analyzer** finds the shots: hard cuts with a confidence, dips to black / white when asked, a
minimum shot length, manual add / remove, and a contact sheet of every shot's first frame (`shots_json`, contract
`h3.shot_manifest` v1; with `window_json` the shots are also given in source frames). It uses PySceneDetect when
installed and the dependency-free colour-histogram detector otherwise (the manifest says which). Latent context
should not cross a cut: the manifest says so for later stages.

**H3 Sequence Auditor** checks an image sequence before anything renders from it: missing frames, identical
frames, unreadable or empty files and frames of another size, against an expected range and size. It reads only
file headers (OpenImageIO, else PIL), decodes a few frames for a contact sheet, and gives `valid_ranges` in
`manifest_json` (contract `h3.sequence_manifest` v1) plus a plain report. Give it a pattern (`shot.####.exr`,
`%04d`, `@@@@`, `$F4`), a folder, or one frame of the sequence.

**H3 Segment Writer** + **H3 Sequence Assembler**: a long job never holds the whole video.
- Each run writes its segment as one file (`seg_00012.mp4` + a `.json` note) in a job folder inside ComfyUI's
  output folder. Codecs: h264, ProRes 422 HQ 10-bit or FFV1 lossless; BT.709-tagged; written through a
  `.partial` name, so a crash never leaves half a file.
- The assembler joins the segments in number order by remuxing: no decoding, no re-encoding, one packet in
  memory. A missing segment number stops it unless `allow_gaps`.

**H3 Media Conform** conforms the lazy handle and stays lazy (nothing is decoded):
- a source range;
- a constant frame rate, judged on the file's real timestamps so variable-rate clips come out right. An image
  sequence is timed by its frame NUMBERS, so a gap in the numbers is time that passed and later frames do not slip
  early;
- a size: fit / fill (centre crop) / exact, optionally a multiple of N.
Its `ledger_json` names the source frame of every output frame. Colour tags pass through unchanged; colour and
HDR conversion is a later stage.

**A new soundtrack (`audio`, 2026-11-02, for lip sync).** Connect a sound and it replaces the plate's own sound.
- It starts at the first kept frame and lasts exactly as long as the kept frames. A longer sound is cut, a shorter
  one is padded with silence, and the report says which.
- The video is copied packet for packet (no re-encode) into `output/h3_media/`, with the new sound beside it. Every
  frame window and a job's final movie then carry the new sound.
- The same plate and sound reuse the copy.
- It works for video files only; an image sequence must become a video first.

**H3 Proxy Manager** writes a smaller copy of the lazy video and hands it back as a lazy handle.
- Sizes: 1/2, 1/4, 1/8 or custom. Codecs: h264, ProRes 10-bit or FFV1 lossless. The selection's audio can be kept.
- It writes into `<job_dir>/proxies/` inside the output folder.
- Proxies are cached: the same source (file, selection, crop, size, layer) with the same settings is reused, never
  re-encoded. A changed source makes a new proxy.
- Its `source_map` takes proxy frames and coordinates back to the source.

**H3 Sequence Diff** compares two versions frame by frame.
- Methods: pixel difference, block SSIM, or multi-scale SSIM on luma. Multi-scale SSIM is a structural measure, not a
  learned perceptual metric.
- It can compare inside a mask, and can pair frames by their frame maps (the same source frame number).
- It streams both inputs in chunks.
- Outputs: heatmaps of the worst frames, a numeric report, the changed intervals (in frame numbers) and a flicker
  graph.

**H3 Review Package** writes a review folder:
- `review.html`, self-contained: an A/B wipe per frame, difference heatmaps, QC markers and notes. All text is
  escaped;
- a contact sheet with frame numbers and timecode;
- an A | B side-by-side movie;
- a manifest with a sha256 for every file.

**H3 Shot Publisher** publishes rendered media into `publish/<show>/<seq>/<shot>/<task>/v###/`.
- Each version is reserved atomically, so two publishes never share one.
- Stage: wip / review / final. `require_qc = pass` refuses without a passing QC.
- It never overwrites another version, and verifies every copy by sha256.
- Every path stays inside ComfyUI's output folder.

They run on this pack's own copy of the C2C shared engine (`mmx_vendor/c2c_core`, Apache-2.0, kept byte-identical to
CustomNodePacks' by a sync check; see `mmx_vendor/VENDORED.md`). CustomNodePacks is not needed.

With CustomNodePacks installed, its **Load Video (C2C)** also loads stills, EXRs (any layer) and image / EXR
sequences (2026-11-01). Its `video` output plugs straight into H3 Media Probe (`c2c_video`) or H3 Frame Window
Reader (`media`):
- a single still becomes a one-frame source;
- a sequence keeps its own frame numbers (Load Video's `frame_map`).

This was checked live on core 0.39 with frontend 1.53.10.

## One click runs a whole job: H3 Task Spec -> Segment Planner -> Job Controller -> QC Gate -> Job Commit (2026-11-01)

A long H3 job is planned once and then runs by itself. You no longer queue every segment.

- **H3 Task Spec** says what the job is:
  - the task (edit a plate, or generate);
  - the frame range;
  - the segment length on the H3 grid (17k + 5);
  - the context frames each segment continues from, the cross-fade and the tracking handles;
  - seed, retries, the approval policy, auto-continue and the delivery codec.

  It fingerprints the plate, the prompt and the references, so changing one later redoes only the segments that
  used it.
- **H3 Compatibility Validator** checks it before anything runs: the 17k + 5 grid, sizes in multiples of 32, 24 fps,
  the range inside the plate, and the context length. The reasons are given in words.
- **H3 Segment Planner** splits the range into segments. Continuation context, cross-fade, tracking handles and grid
  padding stay separate. The accepted ranges tile the range exactly, and the planner checks this.
- **H3 Job Controller** opens the job (on disk under `output/h3_jobs/<job>/`) and hands each run its next segment:
  the plate window, the seed and the context frames. It also applies your actions: pause, resume, cancel, approve or
  reject a held segment, retry a segment with a new seed, roll back, generate an alternate take of an accepted segment (kept beside the chosen one), choose another take (only the segments after it are redone).
  - Its `context_latent` comes from the previous segment only when that segment generated nothing past its accepted
    end. With a blend tail (or grid padding), the latent's last steps would be out of step with the accepted
    frames. In that case the pixel context is used and the status says so (2026-11-03).
- **H3 Face Track Crop / Face Mask SAM** place the face's paste mask and SAM prompts where the face is in its crop.
  This holds even near a frame edge, where the crop box cannot stay centred on the face (fixed 2026-11-03).
- Fixed 2026-11-05:
  - **H3 Task Spec / Shot Planner**: a 23.976 (29.97, 59.94 ...) shot plan matches a 23.976 plate. The planner
    stored 2997/125 while the plate reports 24000/1001, so the plan was refused.
  - **H3 Face Stitch**: the feather width is worked out per frame (it changed only every 32 frames or so, at
    memory-sized chunk boundaries). A plate of another size than the crops' transform is refused with a message.
  - **Detail Tone Lock**: a refined batch with another number of frames than the source is refused (it silently
    used refined frame 0 everywhere). One refined frame still applies to all.
  - **Prepare Continuation**: an overlap as long as the new segment is refused (nothing new would be generated).
  - **Load Context Latent / H3 Reference Bank**: on a server reachable from other machines (`--listen`), files must
    be inside the C2C media roots, as for C2C Load Video. A local server reads any path, as before.
- **H3 QC Gate** judges the segment:
  - frame count;
  - finite values;
  - no black frames;
  - not frozen;
  - it continues the accepted frames;
  - optionally its difference from the source;
  - optionally its scene's colour (with H3 Subject Memory, below).
- **H3 Job Commit** records the result, by the QC verdict and the approval policy:
  - **accept**: the frames are kept as 16-bit PNGs numbered by global frame;
  - **retry**: with a new seed, up to `max_retries`;
  - **hold for review**.

  When every segment is accepted it assembles the movie from the chosen takes. With auto-continue it queues the next
  run itself, through ComfyUI's own `/prompt`. It stops when the job is paused, cancelled, waiting for review or
  complete, after `max_auto_runs` runs in a chain, or when `H3_AUTO_CONTINUE=0`. The queued runs report to the
  browser tab that clicked, so you see every segment's progress, previews and status there.
- The job file keeps every state change, each segment's attempts (seed, QC, outputs with sha256), the takes and the
  retry history.
  - A run that dies mid-segment is retried on the next click.
  - An assembly that dies is finished on the next click.
  - A rejected segment is never used as continuation context.

## Long generative video: shots, references, subject memory (2026-11-01)

The same job, generated from prompts with no source plate, in shots. It still runs from one click.

- **H3 Shot Planner** turns a prompt timeline into shots, one shot per line:

  ```
  4 | a lighthouse keeper climbs the spiral stairs at dusk | cut
  3 | she reaches the lamp room and looks out to sea | continue
  4 | a storm rolls in over the dark water | dissolve 12
  ```

  - `seconds | prompt | transition` is the format.
  - The transition is the one INTO the shot:
    - **cut** (the default) starts a new scene, and no continuation context crosses it;
    - **continue** keeps the take going with a new prompt;
    - **dissolve N** cross-fades N frames with the end of the previous shot.
  - JSON with the same fields works too.
  - Bad lines are refused, naming the line.
- **H3 Task Spec** takes the plan on `shot_plan`. The shots set the length (leave `count` at 0). Each segment gets
  its shot's prompt, with the Task Spec's `prompt` as a prefix shared by every shot. The Job Controller's new
  `prompt` output carries it to your text encoder.
- **Editing a shot.** Change a shot's prompt and run again: only that shot's segments are generated again, plus
  any `continue` shots that follow it. Shots after a cut or a dissolve are kept.
- **A dissolve needs room.** The last segment of the shot before a dissolve keeps generating past the shot's end,
  and those frames are kept as its tail. The assembly cross-fades them with the new shot's first frames, linearly.
  The extra frames come from that segment's grid padding; if it has too little, the plan says so.
  - The same tail now makes `blend` work. Earlier, blend frames were generated and then dropped.
- **H3 Reference Bank** keeps the job's reference images, each with a role:
  - the roles are identity, face, head, hair, body, outfit, footwear, props, expression, performance, scene,
    lighting, camera and style;
  - exact duplicates (the same pixels) and near duplicates (a 64-bit difference hash within 6 bits) are marked;
  - each image is saved once under `output/h3_refs/<name>/`, and banks chain to add more;
  - `fingerprint` goes to the Task Spec's `references`, `gallery` shows every entry with its marks, and
    `identity` gives the identity / face / head images for the model's reference inputs.
- **H3 Subject Memory** shows what the subject looks like so far. It is rebuilt from the job on every run, so a
  rejected generation can never be in it.
  - Identity comes from the bank.
  - Approved frames come from accepted segments of the current scene, nearest first. After a cut it starts over
    with the bank alone.
  - The scene's colour baseline goes to the QC Gate. The QC Gate's optional `memory` input checks each segment
    against its scene's look (`colour_drift_max`, mean-colour CIE76 delta E, 12 by default, 0 = off). The first
    segment of each scene has no baseline, so it is not checked there. The default of 12 is a first guess, not yet
    measured on real H3 output.
  - With no bank and no accepted frames, `anchors` stops the node that uses it with a sentence saying so. It never
    feeds a black frame.

Live check, one click on the CPU testbed: a 2 s / 3 s / 2 s job (cut, continue, dissolve 6), 168 frames. The movie's
frame numbers run in order, the shot colour changes exactly at the cuts, and the dissolve frames are the expected
mixes. It passes in the classic renderer, in Nodes 2.0 and inside a subgraph. A CPU stand-in replaced the sampler,
so the check shows the job logic, not image quality.

## Attribute transfer on a plate as a job: blood, dirt, wounds, makeup (2026-11-01)

The first benchmark of Revision 3 (the PRB Blood Chunk workflow) is an edit of a whole plate. The job engine now
does that properly. Workflow `workflows/tasks/15_attribute_transfer_job.json` puts it together.

- **Held frames before a fresh segment.** H3 Task Spec `hold_head` (8 in the PRB workflow's AV handles): a segment
  that does not continue from accepted frames first generates that many copies of its first plate frame, then
  drops them. The sound starts that many frames earlier, with silence, so picture and sound stay together.
- **Windows that fit the model.** The Job Controller's `window` is exactly the segment's model length (17k + 5).
  Held frames come first. Grid padding at the end is the plate's own next frames while it has them, then the last
  one repeated. H3 Source AV Latent takes every segment without refusing the last one.
- **A matte cut exactly like the plate.** The Job Controller takes an optional `matte` (a mask clip of the same
  length and size) and gives `matte_window`: the same frames, the same held frames. A matte of another size, or
  shorter than the plate, is refused in words.
- **A driving performance cut the same way (2026-11-02).** `driving` (any size, at least as long as the plate) gives
  `driving_window`, so performance transfer takes the driving clip's matching frames in every segment.
- **Accepted latents handed on.**
  - H3 Job Commit takes the sampler's `latent` and keeps it with the accepted frames.
  - The Job Controller's `context_latent` gives the next segment the latent of the take it continues from, for H3
    Motion Context. A rejected attempt is never handed on.
  - This replaces saving and loading context latents by file path.
- **The plate's sound.** H3 Task Spec records where the plate's sound for the job's frames is, and the assembled
  movie carries it. The assembly checks that the sound lasts as long as the picture (within one frame) and records
  the result in the job.
- **QC for edits.** H3 QC Gate gains `plate` and `mask`.
  - Outside the mask (grown by `protect_px`) the result must stay the plate within `background_max` (2/255). The
    frames that changed are listed by their global numbers.
  - Mask area jumps and dropouts are flagged: a warning, or a failure with `strict_masks`.
  - The black / frozen checks now look only at the frames the job keeps, so held frames and padding no longer count
    as frozen.
- **Workflow 15** is workflow 06's masked-edit chain inside the job. It runs on core Load Video -> H3 Media Probe,
  so it works without other packs.
  - The GPU stage (crop, Image + Reference to Video with the look-dev picture, Source AV Latent, Motion Context
    with `context_latent`, sampler, decode, Subject Uncrop exact outside mask) is one native subgraph. Open it to
    edit.
  - Not run on H3 weights yet.

Live check on the CPU testbed: the shipped workflow with a stand-in in place of the GPU subgraph, on a 70-frame plate
with sound. It passes in the classic renderer, Nodes 2.0, inside a subgraph, after an interrupted run, and on a
second plate of another length and size. See the C2C workspace, evidence L9.76.

### Character swap groundwork: a seam check and the finishing algorithms (2026-11-01, in progress)

- **H3 QC Gate `seam_ratio_max`** (1.5; 0 = off). With a plate and continuation, it compares the step from the
  previous accepted frame into the segment with the steps around it. It also compares that with the plate's own
  ratio at the same place, so real motion or a flash in the plate is not mistaken for a generation seam.
- `mmx_utils/body_double.py` ports the finishing steps of open-source-higgsfield's Body Double (Apache-2.0; see
  CREDITS):
  - follow-camera crop plan;
  - colour-histogram re-identification;
  - mesh-warp alignment;
  - cast matte;
  - mask dropout bridging;
  - background colour match;
  - temporal deflicker.

  The nodes built on them are listed below.
- The limits (seam 1.5x, the 9x5 alignment knots) are the reference's values for another model. They are
  parameters, not yet measured on H3 output.
- **H3 Plate Restore: `register` = `mesh (experimental)`.** Instead of one shift, each frame gets a smooth warp
  fitted on the background around the edit, smoothed over time, for generations that squeeze or drift unevenly.
- **H3 Plate Restore: `cast_matte`** (off by default; `cast_reach`). The composite also covers where the new
  character reaches past the edit mask (bigger hair, a wider costume), and only next to the edit.
- **H3 Track + Crop: `planner_mode` = `follow_camera (smooth zoom)`** (`min_crop`, `follow_radius`). A crop with the
  canvas's aspect follows the subject smoothly. It zooms out when the pose opens up (arms raised, a jump) and never
  cuts the subject: the reference's smoothed centre could lag a fast subject out of frame, so each frame's crop is
  also kept around that frame's box.
- **H3 Temporal Deflicker** (new). Removes frame-to-frame boil in regions that hold still and leaves moving regions
  alone (`motion_low` / `motion_high` set the ramp); an optional mask limits it to the regenerated area. The first
  and last frames pass through unchanged.
- **H3 Temporal Matte Chatter Corrector** (new). When a segmenter loses the hair, a sleeve or the whole subject for a
  few frames, each lost frame would paste the plate back over the new character, which reads as flicker. A pixel
  stays covered while the mask covered it within `window` frames before AND after it; a subject that really leaves
  stays uncovered. With the crop `bboxes` it judges in plate space, where the set holds still while a follow-cam
  crop moves.
  - It adds only the bridged pixels. Frames without a dropout come out unchanged: no soft fringe, no growth.
  - `bridge_take_ends` (off by default) also closes a dropout at the very start or end of the take. Use it only for a
    subject that holds still: a moving subject's next `window` frames of path would be painted into the first frames.
  - It closes a dropout where the subject was both before and after it. For a subject that crosses the frame during
    the dropout, that is the overlap of the two silhouettes, not the whole subject.

### Choosing and following the person to replace (2026-11-02)

Detection and tracking are ComfyUI's own SAM 3 / 3.1 (core 0.39 or newer): load the SAM 3.1 checkpoint with core
**Load Checkpoint**, encode the text `person:20` with its CLIP, and run core **SAM3 Detect** with `individual_masks`
on. The `:20` matters: core's SAM 3 text encoder reads `word:N` as "up to N objects", and plain `person` returns only
the single most likely person, so there would be nothing to choose from.
The three nodes below decide who is who. None of them changes the person silently: when the numbers do not settle
it, they stop and show the candidates.

- **H3 Multi-Subject Selector** (new). Picks the person from SAM3 Detect's candidates and gives them a stable
  subject id.
  - `auto` ranks the candidates by size, by colour similarity to `reference_images`, and by `position_preference`.
  - When the best two are closer than `ambiguity_margin`, it stops. The `gallery` shows the candidates, each with
    its `#` number; set `selection` = `index` and pick one.
  - `subject_lock` (a `subject` from an earlier run) keeps that person and their id.
- **H3 Subject Re-Identification** (new). Finds the person again among new detections: after a cut, an occlusion
  or a lost track.
  - Score = colour similarity to the subject's memory − `motion_weight` × how far they moved (a fraction of the
    width).
  - It stops when the score is under `min_confidence`, or a second candidate is within `margin` (two look-alikes).
    `confirmation` = `pick index` confirms one by hand.
  - Appearance means body and clothing colours. No face model is used.
- **H3 Temporal Subject Tracker** (new). Tracks the person through a clip of any length, as a job.
  - It reads `media` (H3 Media Probe) one `window` at a time, tracks with core SAM 3, and queues the next run
    itself.
  - **A loss is:** the mask vanishes; it shrinks under `area_ratio` (35%) of its recent size; or it jumps sideways
    more than `jump` (15% of the width).
  - **Optional checks:**
    - backward tracking (`validate_backward`);
    - a colour check every `reid_interval` frames (catches a tracker that moved to someone else).
  - **On a loss** it re-identifies the person and goes on. Up to `max_gap` frames may pass with nobody matching
    (occluded, off frame); those frames get an empty mask.
  - **Otherwise the job waits for review.** Its `preview` shows the candidates. Set `review` to:
    - `continue with candidate` and `review_pick`;
    - `end the track here`;
    - `track again`.
  - State and masks are kept in `output/h3_jobs/<job>/track/`. A crash or a restart resumes where it stopped.
  - Changing a setting starts a new track; the old one is kept.
  - `backend` = `masks input` checks a finished track from any other tracker for the same losses.
  - When the track is done, `matte` is the person's matte at the plate's size, as a lazy handle. Connect it to H3
    Job Controller's `matte`; each segment reads only its own frames. While the track runs, the job waits without an
    error. While the track waits for review, it stops with that message.

### The anchor and the props (2026-11-02)

The anchor is the take's first frame repainted with the new character. Every later segment is judged against it.
Make the candidates with your own H3 image path (any number, as one batch); these nodes check and keep them.

- **H3 Anchor Studio** (new). A contact sheet of the source frame and every candidate, numbered. For each candidate:
  - the new character's mask: the person standing where the performer stands, found by core SAM 3 (or given);
  - "NO PERSON" when there is none;
  - colour match to the source subject and to the references;
  - a difference map against the source.

  Colour match compares clothing and body colours; it is not a face comparison.
- **H3 Anchor Approval** (new). `decision` = `approve` or `reject` candidate `#`.
  - The decision is kept in the H3 job (`job`).
  - The approved anchor joins the Reference Bank as identity, so H3 Subject Memory and every segment use it.
  - A rejected candidate is never used.
  - On later runs, `decision` = `review` returns the approved anchor of that source frame. Until one is approved,
    the nodes that need it wait without an error, and `status` says what to do.
- **H3 Prop Consistency Checker** (new). Stops a take when the references, the anchor or the prompt give the new
  character a prop (a hat, a bag, a phone) that the performer does not hold. The video model takes props from the
  source performance, so such a prop pops in and out.
  - A prop the performer already holds is allowed in the anchor.
  - `locked_props` are always allowed.
  - `policy` = `warn` only reports.
  - Props are found by core SAM 3 from their names; it needs the SAM 3 checkpoint's MODEL and CLIP.

**Task workflow 16 (`workflows/tasks/16_character_swap_job.json`)** chains all of this in front of the job engine.
From one click it picks and tracks the person, takes the approved anchor as the new character's picture, checks the
props, then regenerates only the tracked person, segment by segment. The plate's sound is kept. Until a step is
done, the steps after it wait without an error.

Checked on a CPU, with the real SAM 3.1 and a stand-in for the H3 stage:
- 48 frames with a hard cut;
- the person lost at the cut and found again;
- the job completed in 6 automatic runs, one spoiled segment redone;
- outside the person, the plate unchanged.

The H3 stage itself has not been run on H3 weights yet.

**Task workflows 17-26 (Task Mode, 2026-11-02).** The same chain for other targets, each shipped as ONE task node
whose widgets are its controls. Opening the node gives the editable graph, in stage groups from media to QC.

| Workflow | Task | Target word |
|---|---|---|
| 17 | Wardrobe Transfer | `jacket:10` |
| 18 | Prop Replacement | `phone:10` |
| 19 | Object Replacement | `car:10` |
| 20 | Background Replacement | `person:20`, matte inverted so QC protects the person |
| 21 | Body Swap | `person:20` (workflow 16 as one node) |
| 22 | Face Swap | `face:20`, on workflow 11's chain (Identity Fusion, exact uncrop), anchor-approved |
| 23 | Head Swap | `head:20` (face and hair), the same chain |
| 24 | Lip Sync Director | `face:20`. H3 Media Conform puts the NEW soundtrack on the plate; the identity stays (the plate's first frame) |
| 25 | Expression / Performance Transfer | `face:20`. A driving clip (`<Video 2>`), cut like the plate for every segment by the Job Controller's `driving`; the identity stays |
| 26 | Multi-Shot Continuity Director | source-free: a timeline of shots (cuts, dissolves), one character kept by Reference Bank + Subject Memory, colour continuity checked by the QC Gate |

In 22-25 the generation hears each window's own sound (H3 Frame Window Reader's `audio`).

- The target word is just the SAM 3 text: any garment, prop or object can be named.
- The task node's controls: `target`, `prompt`, `selection` / `index`, `review` / `review_pick`, `decision` /
  `candidate`, `props` / `locked_props`.
- Its outputs: job, track, anchor and prop statuses, the gallery, the track preview and the anchor sheet.

### Fixed 2026-11-01: H3 Plate Restore did not paste the edit back

H3 Plate Restore handed one frame's mask to a blend written for a batch, which read the mask's first three columns
as the alpha.

- **Square crops:** the output kept the plate where the regenerated subject should have been (measured: alpha 1.0,
  output = the plate).
- **Other crops:** it stopped with a size error.
- **Affected workflows:** `h3_plate_fidelity_PRB.json` and task workflow 07 were affected. Run them again.

Its colour fit was also made stable. On a low-colour ring (a grey set) the plain fit had many answers, picked
differently on CPU and GPU. It now leans gently toward "no colour change", gives the same result on both, and leaves
a well-coloured ring's fit as it was.

### Convert to Subgraph no longer re-wires nodes with autogrow inputs

ComfyUI's front end has a fault on 1.53.10, 1.55.14 and 1.57.0, with core nodes alone. Converting a node that has
growing input groups (H3 Image + Reference to Video, core MiniMax H3 Reference to Video and others) into a subgraph
puts its links on the wrong inputs. Measured: a reference video arrived on an AUDIO input, and `width`'s link on
`length`.

`web/subgraph_autogrow_fix.js` puts every link back on the input it was on, through the front end's own connect
calls, and removes the empty slots the mistake grew. Nodes that were not affected are not touched.

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
- **H3 Temporal Depth runs real depth models.** Depth Anything V2 / 3, Depth Pro, DepthCrafter and DVD now run
  through ComfyUI-CustomNodePacks' depth runners when that pack is installed. Before, DepthCrafter and DVD always
  stopped with an error and nothing else was offered. Without CustomNodePacks they say what is missing. Any depth node
  works too: wire core's "Run Depth Anything 3" into `passthrough_hints`. With nothing wired, `passthrough` is a
  luminance proxy, and the report now says so. `temporal = align (experimental)` puts the frames on one depth scale
  over the `window` (which did nothing before). Measured: it removes 98 % of per-frame scale / offset flicker on
  synthetic jitter, but it did not help a clean pan, so it is off by default. `normals` carries zeros when off,
  instead of nothing.
- **H3 Context Windows plans what H3 Context Window Sampling does.** Its defaults are now the sampler's (141 / 39). A
  new `overlap_frames` output goes with `window_frames` into the sampler. A test checks that both nodes cut the same
  windows.

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

## Real HDR from H3 (experimental, 2026-11-11)

H3 was trained on SDR video, so it emits SDR. Pushing that through an inverse log curve gives a valid 16-bit file
with the highlights still clipped. The proven route to real HDR is a LoRA that teaches the model to work in a log
code (Lightricks' LumiVid / LTX SDR-to-HDR IC-LoRA). The plan, the research and the owner / developer work split are
in the workspace's `docs/research/l9178_minimax_hdr_rnd.md`. Shipped so far:

- **`tools/h3_vae_log_probe.py`.** Measures which log code (ACEScct, ARRI LogC3, LogC4) the H3 video VAE carries
  best, on your own HDR footage. Reports PU21-PSNR, colour drift down to -8 EV, per-stop error and fake-HDR
  footroom. Run it on the GPU box:

      python tools/h3_vae_log_probe.py --comfy <ComfyUI> --vae <h3 vae .safetensors> --input shot.####.exr --out p.json

- **H3 HDR Roundtrip `code_space`** (appended input, default off = unchanged).
  - `to_view` encodes ACEScg scene-linear into the chosen log code.
  - `to_scene` decodes generated log frames back to ACEScg, and flags fake HDR (pixels below the curve's black code).
  - Save Video (C2C) writes the result as 16-bit LogC3 DPX or half-float EXR.
- **`tools/h3_hdr_dataset.py`** builds the training set from Poly Haven HDRIs (CC0) and your ACEScg EXR plates:
  - Track B targets: log-coded 17n+5 clips, dithered 8-bit FFV1;
  - Track C inputs: degraded SDR renderings, with a held-out tone curve;
  - a parent-disjoint split and `manifest.json`.
  Training config: `docs/research/l9178_h3_hdr_log_lora.yaml` (workspace, ai-toolkit).
- **`tools/h3_hdr_qc.py`** is the checkpoint report. It checks:
  - the fake-HDR footroom;
  - stops above white;
  - exposure and highlight flicker;
  - against a reference: PU21-PSNR, per-stop error and shadow colour drift.
  It prints PASS / CHECK lines against the acceptance targets.
- The curves and camera gamuts are NukeMax's (vendored, `mmx_vendor/hdr_curves`). They match OpenColorIO's ACES
  configs, including CAT02 for ARRI / Sony gamuts.

Not run on H3 weights or a GPU yet: NOT VERIFIED.

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
