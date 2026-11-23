# Remake replacement — `replace/remake_*_replace.json` (2026-11-23)

Character (whole window), head / face (crop window) and item / prop (crop window) replacement through **Viggle
Remake** and **H3 Remake Replace**, run as an H3 job: pick on frame 0, track, approve an anchor, then one click per
segment (auto-continue). Each graph's note is its guide; design `docs/PLAN_h3_replacement_quality.md`. **Status:
wiring checked against the registered node definitions (tests/test_workflow_wiring.py); loaded on the testbed (see
the branch report); NOT run on H3 weights or the Remake LoRA. The LoRA file name and the 6-step / euler sampling are
placeholders to confirm from Viggle's model card.**

---

# H3 Open Regen (standard) — `h3_open_regen_std_*.json`

Ported from rdh9616/ComfyUI-Open-H3-Regenerate (Apache-2.0) onto the Suite's `MiniMaxH3_OpenRegen*` nodes; guide in
the main README, section "Regenerate STD (Open-H3)". `h3_open_regen_std_draft_video_to_2k.json` (a draft video to
2K), `h3_open_regen_std_prompt_to_2k.json` (prompt -> 768p draft -> 2K) and `h3_open_regen_std_decode_saved_latent.json`
(resume after a failed decode). Core nodes only besides the Suite. **Status: wiring checked against the registered
node definitions (tests/test_workflow_wiring.py) and the graph contracts (tests/test_open_regen_workflows.py); NOT
run on H3 weights or a GPU.** Model widgets hold file names only.

---

# Character replacement — `character_replacement/1_…6_*.json`

Six task workflows (replace keeping composition, change pose, change look, expression transfer, masked subject,
across a take). Guide: `docs/character_replacement.md`. **Status: built in the real front end, validated by the
server (only missing model / media files reported), not run on H3 weights.**

---

# H3 Plate Fidelity — `h3_plate_fidelity_PRB.json`

**Status: generated from `test_minimax_PRB_Chandhan.json` by a script; socket names and
types checked against every node's real schema (0 mismatches) and link bookkeeping validated.
Not run end to end here — no H3 weights on the authoring machine.** Diagnosis and design:
`docs/PLAN_h3_plate_fidelity.md`.

What changed from the original:

| Where | Before | After | Why |
|---|---|---|---|
| Ingest | `VHS_LoadVideo` | core `Load Video` → `Get Video Components` (+ `Get Image Size` for frame count) | cv2 decode measured 26.5/255 colour error and 8-bit on a 10-bit BT.709 ProRes plate; core measured 0.4/255, float |
| Sampling | model → guider | + `H3 Latent Colour Anchor` | removes the ref2va red drift each step, measured on the region the model keeps |
| Finish | `MVEx Subject Uncrop` (whole rectangle pasted) | `Detail Match` → `Plate Restore` → `Reference Colour Match` (bypassed until wired) → `Pixel Repair` | puts the original plate back outside the edit, corrects offset and colour, restores sharpness and grain |

The old VHS loader is left in the graph, muted, so the original filename stays visible.
`Reference Colour Match` is bypassed on purpose: wire a tight crop of the subject's skin
from `<Picture 1>` — the stitched references have a flat replaced background that would
skew the statistics — then enable it.

---

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
9. **N3 hints** — consumed by core `MiniMaxH3FunControlNetApply` (control_video) in `h3_masked_face_pipeline.json`
   (core 0.39 ships it; it needs an H3 Fun ControlNet model patch).

## Wired end to end (2026-11-18, L8.52)

`h3_full_spine_pipeline.json` and `h3_masked_face_pipeline.json` are complete graphs now (they were wiring skeletons):

- **Source:** Load Video → Get Video Components → N7 Frame Handles (images and audio).
- **Models:** UNET Loader (H3 Ref2VA), CLIP Loader (type `minimax`), and video and audio VAE Loaders, wired to
  Reference to Video, Masked Replace and the decoders. Pick your own files; the names are the Suite's defaults.
- **Sampling:** Random Noise and KSampler Select feed Sampler Custom Advanced.
- **Back to the plate:** the sampler output is decoded and the **generated** crops go into N5 Stitch Back `crops`
  (the skeleton had the original crops there). The plate goes into N5 and N6, and N6 gets N5's composite mask.
- **Output:** Create Video (24 fps, decoded audio) → Save Video.
- **Full spine only:** N8 and N9 read the plate; N10 Region Mask reads the crops (crop space). N11 Drift QC compares
  the plate with the result inside N5's composite mask.
- **Masked face only:** a Load Image (as Mask) node, titled "Face mask - crop space", feeds N2. Paint the mask or
  replace the file. N3 Control Hints → `MiniMaxH3FunControlNetApply` → Sigma Shift → guider and scheduler.

Built and serialised by the frontend itself (core 0.39.0 / frontend 1.53.10). The C2C workspace build script is
`docs/evidence/L8.52/wire_workflows_live.py`. **Not run end to end:** there are no H3 weights on the authoring
machine. The CPU wiring tests (graph loads, types match, validators pass) do run on both.

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

---

# H3 Hybrid HDR A/B — `h3_hybrid_hdr_ab.json`

**Status: layout + node placeholders only — not run end-to-end on the authoring machine (no H3 weights).** Design: `docs/PLAN_h3_hybrid_hdr.md`.

## Branches (wire on GPU box)

| Branch | MODEL | HybridHDR strength | ColorQC |
|--------|-------|-------------------|---------|
| A | Ref2VA base | — | yes |
| B | Singularity | — | yes |
| C | Ref2VA | 0.5 | yes |
| D | Ref2VA | 1.0 | yes |

1. Run **MiniMaxH3_HybridExtract** once (or `tools/h3_hybrid_extract.py`) with matched Ref2VA / FL2VA / Singularity variants.
2. Decode each branch to IMAGE → **MiniMaxH3_ColorQC** (optional shared reference frame for Oklab distance).

## Two-stage late HDR (not hook keyframes)

HybridHDR applies **constant** LoRA strength. For “Ref2VA early, HDR late”:

1. Sample steps `0 … N` with plain Ref2VA MODEL.
2. Continue steps `N … end` with **MiniMaxH3_HybridHDR** patched MODEL.

Use **KSamplerAdvanced** (`start_at_step` / `end_at_step`) or **SamplerCustomAdvanced** — **never SplitSigmas** on H3.

## h3_hdr_trackA_ltx_to_logc3_dpx16.json (experimental, 2026-11-11)

Track A of the HDR plan (workspace docs/research/l9178_minimax_hdr_rnd.md): an H3 render, saved losslessly, goes through
the LTX-2.5 SDR-to-HDR IC-LoRA (LTX-Enhanced's SDR_to_HDR_Pro graph). Save Video (C2C) then writes the ACEScg delivery as
ARRI LogC3 (EI800) 16-bit DPX, beside the ACEScg half EXR.

Needs LTX-Enhanced, ComfyUI-LTXVideo, CustomNodePacks, and the gated LTX-2.5 HDR IC-LoRA (LTX-2.x Community License).
Built and saved in the real frontend (core 0.39.0 + frontend 1.53.10). Not run on a GPU yet.

## h3_hdr_trackB_log_lora.json (experimental, 2026-11-11)

Track B of the HDR plan: H3 with the log-native HDR LoRA (`h3_hdr_log_acescct_v1`) generates ACEScct log video. H3 HDR
Roundtrip (to_scene, code_space ACEScct) decodes it to ACEScg and checks for fake HDR. Save Video (C2C) writes ARRI LogC3
(EI800) 16-bit DPX. The H3 part is core 0.39's official "Image to Video (MiniMax H3)" blueprint graph; leave
first_frame empty for text-to-video. Not run on a GPU yet.

## h3_hdr_trackC_ic_lora.json (experimental, 2026-11-11)

Track C: any SDR clip to real HDR with the side-by-side In-Context LoRA (`h3_hdr_trackc_icl_acescct_v1`):
1. Load Video (C2C) and Image Stitch put the SDR clip side by side.
2. Solid Mask + Mask Composite make a right-half mask, which H3 Mask Prep and H3 Masked Replace use so H3
   regenerates only the right half.
3. Image Crop keeps that half; H3 HDR Roundtrip decodes it; Save Video (C2C) writes LogC3 16-bit DPX.

Both graphs were built and serialized in the real frontend (core 0.39.0 + frontend 1.53.10): every widget was set by
name and read back, and every required input is linked. Not run on a GPU yet.
