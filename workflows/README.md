# H3 Masked Face Pipeline — Stage 2 workflow

**Status: untested at runtime on the authoring machine.** This JSON was built without H3 model weights or a GPU. It wires the MiniMaxSuite spine nodes to ComfyUI core H3 nodes using real `node_id` values from `comfy_extras/nodes_minimax_h3.py`.

## Chain

`N7 Frame Handles` → `N1 Track + Crop` → `N3 Control Hints` → `N2 Mask Prep` → `N4 Masked Replace` → **sampling guards** → *(sampler block)* → `N5 Stitch Back` → `N6 Detail Reinject`

### Sampling sub-chain (Slice F)

`N4 Masked Replace` → `P10 Protected Layer Guard` → `P11 Accelerator Conflict` → `P1 Sigma Shift (Ratio Locked)` → `P3 Legal Scheduler` → `BasicScheduler` / `SamplerCustomAdvanced`

Optional: `P6 Per-Frame Denoise` after N4 when face size varies across the clip. `P9 Block Cache T8` patches the MODEL before guards. `P2 Sigma Inspector` is read-only on sigmas (report string).

### P5 — regional denoise (no node)

N2 produces the video-side latent mask. N4 writes `latent["noise_mask"] = NestedTensor((video_mask, audio_mask))`. The DiT applies per-token soft denoise via `noise_mask` during sampling — no SplitSigmas, no extra node.

## Defaults (dancing + lateral pan)

- `planner_mode=tv_lp`
- `lock_size=true`
- `movement_cost=2.0`
- `feather_px=32`

## What you must verify on a GPU box with H3 weights

1. **EmptyMiniMaxH3LatentAV** / **MiniMaxH3ReferenceToVideo** — latent frame count matches N7 output (17n+5).
2. **Canvas size** — N1 `canvas_width`/`canvas_height` match the H3 node width/height (multiples of 32).
3. **N4 Masked Replace** — VAE encode injects crops; `noise_mask` NestedTensor is accepted by the sampler.
4. **Lipsync path** — connect vocal `audio` + `audio_vae`; confirm audio stream locks (mouth follows vocals).
5. **BasicScheduler `denoise`** — strength dial; do **not** use SplitSigmas on H3.
6. **MiniMaxH3_SigmaShiftLocked** — ratio-locked shift pair (default 12 / 3, ratio 4). Core `MiniMaxH3SigmaShift` is fine if you do not need ratio lock.
7. **P10 / P11** — guards pass MODEL through; workflow fails loud on illegal dit patches or cache conflicts.
8. **SamplerCustomAdvanced** + **BasicGuider** — full denoise pass produces video; N5 exterior pixels match plate (`torch.equal` outside mask).
9. **N3 hints** — no consumer yet; geometry-only until Union apply exists.

## Unconnected / placeholder

- Load Image / Load Video inputs — connect your plate clip at node 1.
- Checkpoint / VAE / CLIP loaders — connect your H3 weights.
- Sampler output → VAE decode → link decoded frames into N5 `crops` input (post-sampler path).

## Node index (workflow file)

### `h3_masked_face_pipeline.json` (Stage 2)

| ID | Type |
|----|------|
| 1 | MiniMaxH3_FrameHandles |
| 2 | MiniMaxH3_TrackCrop |
| 3 | MiniMaxH3_ControlHints |
| 4 | MiniMaxH3_MaskPrep |
| 5 | EmptyMiniMaxH3LatentAV |
| 6 | MiniMaxH3ReferenceToVideo |
| 7 | MiniMaxH3_MaskedReplace |
| 8 | MiniMaxH3SigmaShift *(replace with MiniMaxH3_SigmaShiftLocked for ratio lock)* |
| 9 | BasicScheduler |
| 10 | BasicGuider |
| 11 | SamplerCustomAdvanced |
| 12 | MiniMaxH3_StitchBack |
| 13 | MiniMaxH3_DetailReinject |

### `h3_full_spine_pipeline.json` (Phase C1 — full spine)

`N7 Frame Handles` → `N1 Track + Crop` → `N8 Strongest Pose` / `N9 Temporal Depth` → `N10 Region Mask` → `N2 Mask Prep` → `N4 Masked Replace` → `P1 Sigma Shift Locked` + `P3 Legal Scheduler` → *(sampler)* → `N5 Stitch Back` → `N6 Detail Reinject` → `N11 Drift QC`

| ID | Type |
|----|------|
| 1 | MiniMaxH3_FrameHandles |
| 2 | MiniMaxH3_TrackCrop |
| 3 | MiniMaxH3_StrongestPose |
| 4 | MiniMaxH3_TemporalDepth |
| 5 | MiniMaxH3_RegionMask |
| 6 | MiniMaxH3_MaskPrep |
| 7 | EmptyMiniMaxH3LatentAV |
| 8 | MiniMaxH3ReferenceToVideo |
| 9 | MiniMaxH3_MaskedReplace |
| 10 | MiniMaxH3_SigmaShiftLocked |
| 11 | MiniMaxH3_LegalScheduler |
| 12 | BasicGuider |
| 13 | SamplerCustomAdvanced |
| 14 | MiniMaxH3_StitchBack |
| 15 | MiniMaxH3_DetailReinject |
| 16 | MiniMaxH3_DriftQC |

## Deferred / cut (this slice)

- **P4 dual-clock shim** — unnecessary on core with `ModelSamplingAV`; see pack README.
- **P7** — needs LoRA asset.
- **P8** — LongMedia-sized; deferred.
- **P12 audio QC** — needs decoded H3 output; skipped.
