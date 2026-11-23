# H3 zero offset on V2V: the plate comes back where it was

Owner order 2026-11-23: "we don't want any pixel offset when doing V2V". Follows `docs/PLAN_h3_plate_fidelity.md`
(faults F1-F5); this note owns F4 (offset) and the geometry half of F1/F2.

## 1. Design note (written before the fixes)

**Contract.** Through every V2V chain of the Suite (Track + Crop -> ... -> Stitch Back; Subject Crop -> ... -> Subject
Uncrop / Plate Restore; Mask Prep / Mask To Latent Space; Detail Reinject; Continuation), with an identity "edit":

1. **Exterior bit-exact**: every pixel outside the paste region is the literal plate tensor (0/255, in float, not
   just after 8-bit rounding).
2. **Interior registered**: the pasted region sits on the plate with < 0.25 px displacement (target), no scale
   drift, and no colour shift beyond resampling.
3. **Masks registered**: a masked plate pixel always lands in a latent cell that is regenerated.
4. **Sound registered**: latent audio tick k is built from plate samples [k*hop, (k+1)*hop).

**Where geometry can change** (audited, every pixel-coordinate or size change on the path):

| Stage | What changes | Rule that must hold |
|---|---|---|
| Track + Crop | float box -> canvas, bilinear `affine_grid` (align_corners False) | inverse theta is the exact inverse |
| Subject Crop | integer box, optional resize (Lanczos / bicubic, half-pixel) | the paste resizes back with the same centres |
| H3 VAE encode (core) | **centre-crops each side to a multiple of 16** (`comfy/sd.py vae_encode_crop_pixels`) | a crop off that grid decodes smaller and shifted by up to 7 px |
| Latent (H/16, W/16), DiT 2x2 tokens | cells cover the CENTRE-cropped frame | masks must be reduced on the same cells |
| Audio VAE encode (core) | **centre-crops the waveform to a multiple of 800 samples** | tick 0 must start at sample 0 |
| Stitch Back / Uncrop / Plate Restore | inverse resize + feathered composite | exterior = `torch.where` literal plate |
| Detail Reinject | frequency separation | an unchanged region comes back unchanged |
| Continuation | latent rows of the previous tail pinned (noise mask 0) | rows start on a (1,4,4,4,4) cycle |
| Frame Handles | frames repeated at head / tail | trimmed back after decode (see open items) |

**Method.** `tools/h3_zero_offset_bench.py` (committed; GPL-3.0): synthetic plates at odd sizes (1922x1081,
1366x770, 1081x1921 = 9:16) carrying independent signals per channel - red: band-limited checkerboard + aperiodic
texture (phase correlation), green: gradients, blue: Gaussian fiducial dots at irrational sub-pixel positions
(centroids: the primary registration number). The Suite's real node classes run end to end with the VAE + model
replaced by a stand-in of known geometry: `exact` (identity) and `vae16` (core's centre crop to a multiple of 16 + a
shift-free 2x softening; its own floor, measured with no chain, is 0.0001 px). Columns: exterior max |diff| x 255 per
channel outside every box (and outside the mask where a mode promises it), dot-centroid displacement inside the fully
pasted region, phase correlation (cross-check), interior mean / max |diff| x 255.

## 2. Findings (before -> after; worst over the three plate sizes unless noted)

| # | Node / chain | Fault | Before | After | Fix |
|---|---|---|---|---|---|
| Z1 | Subject Crop (`divisible_by` not a multiple of 16, or full frame) -> VAE -> Subject Uncrop / Plate Restore | core's VAE cut the off-grid crop and the paste stretched the smaller decode into the box | **5.02 px** (1366x770 full frame, div 8); 4.86 px (1922x1081 tracked box, div 8) | **0.0005 px** | Subject Crop records the crop size it emitted in each box (`metadata.mmx_crop_size`, core's BoundingBoxes key); Subject Uncrop and Plate Restore put a decode that is exactly the VAE-kept size back at core's offset, margins from the plate (`recover_vae_crop`). Report + ledger say so. |
| Z1b | Track + Crop canvas off the 16 grid -> VAE -> Stitch Back | same stretch on the inverse warp (canvas 200x136 -> decode 192x128) | > 1 px (test must-fail twin) | < 0.25 px (test) | Stitch Back rebuilds the canvas: decode at core's offset, margins from `affine_crop_batch(plate)` |
| Z2 | Subject Crop / Uncrop resize (and Source AV / Bridge / Motion Context / AV noise mask Lanczos) | core's `lanczos` runs through 8-bit PIL: clips HDR > 1.0, quantises to 256 levels (floor), even at the same size | HDR x1.5 zoomed: interior max **129.5/255**, mean 4.4/255; SDR 1MP mean 0.48/255 | max 6.5-7.1/255 (resampling of real detail), mean 0.13-0.22/255; SDR 1MP mean 0.14/255 | `mmx_utils/resample.py`: float Lanczos-3 reproducing PIL's coefficients (matches PIL float mode to 5e-7), no clamp, same size = exact copy |
| Z3 | Mask Prep, Mask To Latent Space (masks off the 16 grid) | pooled from the top-left edge (Mask Prep) / adaptive bins over the whole frame (Mask To Latent Space), but core encodes the CENTRE-cropped frame | 1-px strips: up to **182** masked cells marked keep (Mask Prep, 1922x1081), **118** (Mask To Latent Space) | **0** | centre-crop the mask with core's arithmetic before pooling; on-grid masks bit-identical |
| Z4 | Detail Reinject | port error vs NKD split/combine: the whole generated image was multiplied by the plate's detail (detail counted twice) | generated == plate: mean **8.8/255**, max 384/255, dots moved **0.43 px** | mean 0.025/255, 0.0 px | low band of the generated + high band of the plate (divide / subtract), `strength` lerps from the generated |
| Z5 | Source AV Latent, Masked Replace, native audio lock | core's audio VAE centre-cropped the waveform to whole 800-sample ticks: the kept sound sat early | 121 frames (161333 samples): **266 samples = 8.3 ms**; up to 399 samples (12.5 ms) | tick 0 = samples 0-799 | pad the waveform at the END to whole ticks before encoding (`h3_geometry.pad_waveform_to_audio_hop`); trimmed to the frame count as before |
| D1 | `mask_clean._dilate_cv2`, `plate_restore._rect_morph_cv2` (coordinator L8.33) | `.numpy()` without `.cpu()` | guarded by `device.type != "cpu"` already (a CUDA mask never reached `.numpy()`; meta-device test) | `.to(device="cpu")` before `.numpy()`, result back on the input's device | hardening |

**Found by the replace-quality agent (2026-11-23), fixed here:**

| # | Node | Fault | Before | After | Fix |
|---|---|---|---|---|---|
| G1 | Subject Crop in a job (one run per window) | each window planned on its own: the crop moved between segments, the continuation context landed elsewhere | two windows gave different boxes for the same frame (test) | window crops = the whole-clip crops, bit for bit | appended optional `bboxes` + `bboxes_first_frame`: crop exactly at given boxes (Subject Crop on the whole clip, or one box); saved workflows unchanged |
| G2 | Face Stitch (`mmx_utils/face_refine.py`) | one missed detection dipped the composite weight for ~9 frames: the old face ghosted through | min weight < 0.9 over 5+ frames | 1.0 | misses of up to 6 frames with detections on both sides are bridged before smoothing; real dropouts and track ends still fade |
| G3 | Temporal Subject Tracker matte | proxy masks stored 0/255, then enlarged and thresholded: the proxy's pixel steps on the plate | disc edge error **8.0 px** at 4K (256x144 proxy) | **2.6 px** | masks keep their soft 8-bit edge (0.5 decision preserved exactly for every binary reader), thresholded only at plate size; matte cache re-renders once |
| G4 | Matte Chatter (`body_double.steady_masks`) | bool + int64 per frame x cell: ~2 GB at 1080p without bboxes, > 8 GB at 4K | 9 bytes / cell (test peak 5.4 MB) | 1 byte / cell (1.2 MB), bit-exact | capped uint8 distance code |

**Already exact (measured, no change needed):**

- Exterior outside every crop box: **0/0/0** in every chain, both stand-ins, all sizes, before and after.
  `exact outside mask` and Plate Restore with an edit mask: 0/0/0 outside the mask too.
- Track + Crop -> Stitch Back registration: 0.012-0.052 px (inverse theta is exact). Interior loss with the exact
  stand-in, 1.1-1.8/255 mean (max 24-38/255), is the bilinear crop of a 968 px box into a 512 canvas without an
  anti-alias filter - resolution, not offset; the region is regenerated. Left as is (changing the crop filter would
  change every saved Track + Crop output).
- Subject Crop -> Uncrop without a resize: bit-exact (0.0 px, 0/255) with the exact stand-in.
- Plate Restore `register = correct` on an unshifted decode: 0.0 px applied (dead-band 0.01 px), so it does not
  resample a registered crop.
- Continuation tail pinning: the pinned rows are the previous tail's last `rows_for_frames(overlap)` latent rows and
  `tail_slice` refuses a start that is not on a (1,4,4,4,4) cycle, so the pinned frames are frame-exact by
  construction; the re-decoded overlap is compared with the accepted frames (overlap PSNR in the commit report).
  Compared with mo-akkakk's `carry_mode = "prefix"` (third_party/ComfyUI-MiniMaxH3-TemporalWindow-SpatialTiles,
  MIT, `prefix_window`): the same mechanism (carried latent rows, denoise mask 0, audio ticks too). Its window plan
  keeps starts on the cycle only when window and overlap are 17k+5; the Suite refuses otherwise. Nothing to port for
  offset. Viggle-Remake (Apache-2.0) contributed the idea for Z5 (it pads its waveform for the same reason); no code
  was copied.

## 3. Open items (not changed here)

- **Frame Handles (temporal)**: `h3_full_spine_pipeline.json` pads the HEAD to 17k+5 and nothing trims it after the
  decode, so the delivered clip starts with repeated frames (picture and sound late by the pad). Fix needs a trim-back
  path (an appended output carrying the pad, or a Stitch Back option) - a workflow/UX decision; logged.
- **Subject Uncrop default `protect = feathered box`** pastes the whole box (the model's background inside it). The
  default is kept because changing it alters saved workflows; task workflows 04/06 already use `exact outside mask`.
- **Ingest (F1)** not re-measured in this pass (NOT RUN): F1's numbers stand (VHS 26.5/255 8-bit vs core LoadVideo
  0.4/255 float); the Suite's loaders use the C2C 16-bit reader.
- **Real weights**: everything above is UNIT TESTED with stand-ins on CPU. Whether H3's own decoder adds a
  sub-pixel drift is NOT VERIFIED (no weights here); Plate Restore `register = correct` measures and corrects it per
  frame on real output - read its report on the first GPU run.

## 4. Evidence

- Bench: `PY tools/h3_zero_offset_bench.py --json <out>` (before on origin/main 75db635, after on branch
  `zero-offset`). Tables and JSON: workspace `docs/evidence/zero_offset_2026-11-23/bench_{before,after}.{txt,json}`
  (ComfyUI core 0.39.0, CPU). The after table measures Subject Uncrop `feathered box` inside the wired mask (the
  blend follows `cropped_masks`; a dot straddling the mask edge mixes plate and edit, which read as 0.10 px before).
- Tests (local, never pushed): `tests/test_zero_offset.py` (21 tests for Z1-Z5/D1, then 6 more for G1-G4: 27) - every fix has a measured bar and a must-fail
  twin. On this branch 21 passed; the same file against origin/main: 13 failed (every fixed behaviour), 8 passed
  (pure-function checks and the twins). Full suite `run_tests.py`: 3901 passed, 3 skipped, 1 xfailed.
