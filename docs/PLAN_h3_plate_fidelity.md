# H3 plate fidelity: stop the red drift, the offset, and the softening

Status: design, 2026-09-26. Driven by `test_minimax_PRB_Chandhan.json` (drozbay's
"LatentMaskInpainting w Reference v3.1" on MaskVidExperiments), where a baby is
replaced inside a VFX plate with `minimax_h3_ref2va_bf16`.

## What is actually wrong — measured or sourced, not guessed

| # | Fault | Evidence | Where it enters |
|---|---|---|---|
| F1 | **Ingest mis-decodes the plate.** `VHS_LoadVideo` uses `cv2.VideoCapture` with no colour-matrix control. On a 10-bit ProRes 4444 plate tagged BT.709: worst error **26.5/255** (green +26.5, red −15, skin R −4.6 / B +2), neutrals untouched — the BT.601-matrix signature. And it is **8-bit**: 151 distinct levels on a 256-px ramp. | Measured here, `scratchpad/ingest_test.py`. Core's native `LoadVideo` on the same file: **0.4/255, 256 levels, float32**. | Before H3. Whole frame. |
| F2 | **Background inside the crop is regenerated.** `MVEx_SubjectUncrop` pastes the **entire crop rectangle** (alpha = 1 except a 32 px border ramp) because `cropped_masks` is not wired. Everything inside the box went bicubic-up to 2 MP → H3 VAE encode → **diffusion** decoder (Kijai, Comfy-Org #23: "it's a diffusion VAE") → lanczos-down. H3 VAE reconstruction floor is ~38 dB (comfyui-wiki, 2026-09-19). | Code: `nodes_subject_crop.py` 522–583. | After decode. A visible soft, regrained, re-coloured box. |
| F3 | **Model colour drift, red, per segment.** Comfy-Org #55: red/blush patches on the plain `ref2va` checkpoint with no LoRA, behaving as **temporal colour drift that intensifies frame by frame**, segment-shaped (resets at guide frames). Realism-People LoRA at 1.0 suppresses it (this graph runs it at 0.5). Released H3 is CFG-distilled (MiniMaxAI #75), so the oversaturation of the distillation CFG is baked in; LF-CFG (arXiv 2506.21452) locates oversaturation in the **low-frequency** band. | Sourced. Not reproducible here (no weights on this box). | During sampling. Compounds. |
| F4 | **Offset.** Three distinct kinds are possible and this design measures all three rather than guessing: spatial (sub-pixel shift of kept content), temporal (frame lag — `length % 17 != 5` "silently comes back shorter", panghea/ComfyUI-MiniMax-H3-Inpaint-Tools), and colour offset (a DC shift between the regenerated box and the plate). `MiniMaxH3_DriftQC` already *measures* spatial displacement; nothing *corrects* it. | Mixed. | After decode. |
| F5 | **Texture/sharpness of generated content.** The diffusion decoder regenerates texture; generated regions are smoother and lack the plate's grain. `MiniMaxH3_DetailReinject` copies plate texture from the SAME location, which is wrong for a **replaced** subject (the plate under the new baby is the old figure). | Code + sourced. | After decode. |
| F6 | **Damaged pixels.** Ref2VA residual noise/"snow" (MiniMaxAI #50, sgl-project/sglang #34110) and local chroma blotches (#55). | Sourced. | After decode. |

## What does NOT need a new node

- **F1 is a node swap.** Replace `VHS_LoadVideo` with core `LoadVideo` → `GetVideoComponents`. Measured correct. Building a second loader would duplicate core.
- **F2 is partly one wire.** Connect `SubjectCrop.cropped_masks` → `SubjectUncrop.cropped_masks`. `PlateRestore` below does this properly (colour + offset + feather), but the wire alone removes most of the box.

Ship a corrected copy of the workflow in `workflows/` with both changes and the new nodes wired in. Never edit `third_party/`.

## New nodes (MiniMax H3 Suite)

Maths in `mmx_utils/<name>.py` (pure torch, no comfy imports at module level), node in
`mmx_nodes/<name>.py`, registered in `__init__.py` `_NODE_SPECS` **and** the elif chain —
mind the `endswith` collision rule: a more specific suffix must be tested first.

All colour work in **linear light** (sRGB decode → work → encode). No clamp until the
final encode; values above 1.0 survive. Every node returns a `report` STRING with the
numbers it measured, because "it looks better" is not an acceptance criterion.

### N1 `MiniMaxH3_LatentColorAnchor` — stop the drift where it starts (F3)
Category `MiniMax H3/Sampling`. Inputs: `model`, `latent` (the LATENT fed to the
sampler; uses its `noise_mask`), optional `mask` (overrides), `strength` 0–1 (0.8),
`mode` [`mean`, `mean_std`] (`mean`), `per_frame` BOOL (True), `start_percent` (0),
`end_percent` (0.85), `min_preserved` (0.05). Output: MODEL.

Mechanism: `sampler_post_cfg_function`. `comfy.samplers.KSamplerX0Inpaint` calls the
model — post-CFG hooks included — **before** it overwrites the preserved region with the
source latent, so `args["denoised"]` in the preserved region IS the model's own
prediction of content whose truth we hold. The difference there, per latent frame and
per channel, is the drift. Remove it from the whole frame, every step, before it
compounds.

- Inside the sampler everything is the flat pack `[B, 1, N_video + N_audio]`
  (`comfy.utils.pack_latents`). The guider sets `inner_model.latent_shapes`; the hook
  receives that model as `args["model"]`. `unpack_latents` → correct stream 0 (video,
  `[B, 24, T, h, w]`) → `pack_latents` back. Audio untouched.
- Non-nested latents (plain `[B, C, T, h, w]`) also supported.
- Compare in the sampler's space: `model.process_latent_in(source)`. For H3 this is the
  identity (`MiniMaxH3Video.scale_factor = 1.0`, no shift) — assert it in a test so a
  core change is caught.
- Mask: latent-space preserved = `1 - noise_mask`, prepared to the video latent shape
  (`comfy.sampler_helpers.prepare_mask`); NestedTensor masks → stream 0.
- Per (frame t, channel c): `d = mean(src[pres]) - mean(den[pres])`; `mean_std` also
  matches spread with the ratio clamped to [0.5, 2]. Apply `den += strength * d` to the
  whole frame. Skip frames with preserved fraction < `min_preserved`.
- Only between `start_percent` and `end_percent` (via `model_sampling.percent_to_sigma`).
  Late steps are left alone — they are texture, not colour.
- Cheap: a handful of reductions per step. No extra model calls.

### N2 `MiniMaxH3_PlateRestore` — the finishing composite (F2, F4, colour half of F3)
Category `MiniMax H3/Finish`. A superset of `SubjectUncrop`: same `cropped_images`,
`original_images`, `bboxes` contract (accepts the MVEx/MMX box format), plus
`edit_mask` (crop-space MASK, 1 = regenerated), `grow_px` (8), `feather_px` (24),
`register` [`off`, `measure`, `correct`] (`correct`), `max_shift_px` (4.0),
`colour` [`off`, `mean`, `affine`] (`affine`), `temporal_lag_check` BOOL (True).
Outputs: `images`, `restore_mask` (the alpha actually used), `report`.

Per frame, inside the crop, on the PRESERVED ring (`edit_mask` grown by `grow_px`,
inverted, eroded 2 px so the feather band is not used as evidence):
1. **Spatial offset**: sub-pixel phase correlation (Hann-windowed, luma, parabolic peak
   refinement) between resized-back generated crop and plate crop, restricted to the
   ring by zeroing elsewhere. `correct` warps the generated crop by the negative shift
   (bilinear `grid_sample`, border padding); refuses shifts above `max_shift_px` and
   says so. Calibration gate (test): injected 0 / 0.25 / 0.5 / 1 / 2 px recovered within
   ±0.05 px on a textured synthetic.
2. **Colour**: `affine` = per-frame 3×4 linear-light least squares plate ← generated on
   ring pixels, trimmed (drop the 10% worst residuals, refit once); degenerate/ill-
   conditioned fits fall back to `mean`. Temporally smoothed over ±2 frames so a bad
   frame cannot flicker. Applied to the whole generated crop.
3. **Composite**: alpha = feathered (grown) `edit_mask` × the existing border ramp.
   Outside alpha the output is the **literal plate tensor** — bit-exact, which is what
   gives back the plate's sharpness, grain and colour everywhere the model was not
   asked to change anything.
4. **Temporal lag**: correlate the per-frame ring-mean difference curve at lags −3…+3;
   report the best lag and warn if ≠ 0. Report only — a lag means a frame-count bug
   upstream and silently re-timing would hide it.

Report: per-frame shift (dx, dy), colour correction magnitude (ΔE-ish in linear),
ring pixel count, lag, and which frames fell back.

### N3 `MiniMaxH3_DetailMatch` — give generated content the plate's sharpness and grain (F5)
Category `MiniMax H3/Finish`. Inputs: `images` (after PlateRestore), `original_images`
(the plate at the same geometry), `restore_mask` (from N2), `sharpen` 0–1 (0.7),
`max_gain` (2.5), `grain` 0–1 (1.0), `seed`. Outputs: `images`, `report`.

- **MTF compensation measured, not dialled**: Laplacian pyramid (5 levels) of plate and
  of the pre-restore generated frame over a band just OUTSIDE the edit where both exist
  (the generated crop still covers it before compositing — pass the uncomposited
  generated crop as optional `generated_crop` + `bboxes`; if absent, fall back to
  measuring against a plate-derived reference and say so). Per level k:
  `g_k = clamp(sqrt(E_plate_k / E_gen_k), 1, max_gain)`, smoothed across frames.
  Apply `g_k` (lerped by `sharpen`) to the generated region's bands inside the mask.
  This restores exactly the frequency response the VAE/model lost in THIS shot, and
  nothing more — no generic unsharp mask.
- **Grain match**: noise level function of the plate (residual after a 3×3 median, binned
  by luma, per channel, flat areas only) vs the generated region's; add zero-mean
  Gaussian grain, per-channel correlated like the plate's, scaled so the generated
  region's noise level function reaches the plate's. Temporally independent per frame,
  deterministic from `seed`.

### N4 `MiniMaxH3_ReferenceColorMatch` — hold the subject to its reference (F3, identity side)
Category `MiniMax H3/Finish`. Inputs: `images`, `edit_mask`, `reference` IMAGE (the
`<Picture 1>` image, optionally with its own `reference_mask`), `strength` (0.6),
`mode` [`mean_std`, `histogram`] (`mean_std`), `temporal_smooth` frames (5). Outputs:
`images`, `report`.

Match the edit region's colour distribution to the reference's in **Oklab** chroma (a, b)
with lightness left alone by default (the plate's lighting must win on L). `histogram`
uses 1-D CDF matching per chroma channel. The correction is estimated per frame,
smoothed over time, applied only inside the feathered mask. Report the mean chroma
shift removed, with the red component (+a) called out explicitly — that is the number
that says whether the blush drift was there and how much was taken out.

### N5 `MiniMaxH3_PixelRepair` — clean damaged pixels (F6)
Category `MiniMax H3/Finish`. Inputs: `images`, optional `mask` (restrict), `impulse`
0–1 (0.5), `blotch` 0–1 (0.5), `debug` BOOL. Outputs: `images`, `damage_mask`, `report`.

- **Impulse / snow**: a pixel is damaged when it deviates from BOTH its 3×3 spatial median
  and the median of its temporal neighbours (t−1, t, t+1 at the same position) by more
  than k·MAD (k from `impulse`). Replace with the spatiotemporal median. Static content
  is never touched because the temporal test fails for real detail that persists.
- **Chroma blotches**: in Oklab, chroma residual = a,b minus a large-radius (≈ 1/40 of the
  short edge) robust local estimate; connected regions whose residual exceeds a
  threshold AND whose luma residual is small (a blotch changes colour, not shape) are
  pulled back toward the local estimate, feathered. That is the red-cheek pattern.
- **Must be a no-op on clean input**: a test asserts max change < 1e-4 on a clean textured
  synthetic.

## Explicitly out of scope
- A new video loader (F1 is solved by core).
- Retraining, a new VAE decoder, or a conditioned decoder (ASUKA arXiv 2312.04831 /
  Asymmetric VQGAN arXiv 2306.04632 do this with training; N2 is the training-free
  equivalent: known pixels are restored, not re-decoded).
- Touching `third_party/`.

## Acceptance
- Unit tests per node on synthetics with KNOWN faults injected: shift, per-frame colour
  drift, box regrain, blur (known MTF), grain deficit, chroma blotch, impulse noise,
  latent drift in a fake sampler call. Each fault is recovered to a stated tolerance, and
  each node is (near) a no-op on clean input.
- N1 tested against the real `pack_latents`/`unpack_latents` and a NestedTensor noise_mask.
- `run_tests.py` green; node registration test green; import test with comfy stubbed.
- Corrected workflow JSON loads (all node ids exist) — static check.

## Sources
- Comfy-Org/MiniMax-H3 discussions #55 (red/blush drift), #23 (diffusion VAE, fp32),
  #30 (faces on wide shots); MiniMaxAI/MiniMax-H3 #50 (ref2va noise), #75 (CFG-distilled),
  #78 (VAE colour distortion claim).
- comfyui-wiki 2026-09-19, H3 video VAE optimisation (38 dB reconstruction floor).
- panghea/ComfyUI-MiniMax-H3-Inpaint-Tools (17k+5 silently shortens).
- arXiv 2312.06640 Upscale-A-Video (training-free wavelet colour correction),
  2312.04831 ASUKA and 2601.15368 Aligned Stable Inpainting (decode as local
  harmonisation), 2306.04632 Asymmetric VQGAN (unmasked-region loss in latent
  inpainting), 2506.21452 LF-CFG (oversaturation lives in low frequency).

---

## Review corrections — FINAL (Cursor adversarial review, 2026-09-26)

Verified against ComfyUI source: post-CFG hooks run before `KSamplerX0Inpaint`'s
overwrite; `args["model"]` is the BaseModel carrying `.latent_shapes`; `denoised` is
the packed `[B, 1, N_video + N_audio]`; CFG=1 skips the uncond pass but still runs
post-CFG hooks. The central N1 bet holds. These corrections are binding:

**N1**
1. `unpack_latents` with ONE stream returns `[combined]` still shaped `[B,1,N]`
   (`comfy/utils.py` 1428). Reshape to `latent_shapes[0]` in that case. Test it.
2. `process_latent_in` on the AV pack is NOT identity — `MiniMaxH3` scales the audio
   tail (`model_base.py` `_scale_audio_slice`). Measure and correct on the unpacked
   VIDEO stream only; assert the audio slice is bit-identical after the hook.
3. `denoise_mask` is not in the hook's args. Build the preserved mask in the node's
   closure exactly as `CFGGuider.sample` does (`prepare_mask` per stream, pack), and
   re-read `args["model"].latent_shapes` every call — context windows change shapes.
4. The hook RETURNS the edited `denoised`.
5. Flow sigmas decrease: active while `sigma >= percent_to_sigma(end_percent)` and
   `sigma <= percent_to_sigma(start_percent)`.
6. `per_frame=False` = one correction vector over all T.
7. `m = model.clone(); m.set_model_sampler_post_cfg_function(hook)`. Never mutate input.

**N2**
8. Zeroing outside the ring before phase correlation biases the peak toward (0,0).
   Use masked normalised cross-correlation — Padfield 2012, "Masked Object
   Registration in the Fourier Domain" — PORTED to torch from scikit-image's
   `registration/_masked_phase_cross_correlation.py` (BSD-3, attributed in CREDITS.md),
   integer peak + parabolic sub-pixel refinement. No new pip dependency.
9. Warp with `drift_qc.inject_pixel_shift`; composite with `mask_confined_blend` +
   `gaussian_blur_mask`; extract SubjectUncrop's paste loop into
   `mmx_utils/plate_restore.py` rather than duplicating it.
10. Colour fit in linear light with NO clamp (`colour_match_region` clamps — not
    reused for this). `tone_compensate.fit_affine` is the degenerate fallback.
    Temporal guard = median-of-5, not a ±2 mean.

**N3**
11. After N2 the exterior band is literal plate, so MTF cannot be measured on
    composited frames. N3 runs on the RAW decoded crops, BEFORE N2, with the same
    contract as N2 (`cropped_images`, `original_images`, `bboxes`, `edit_mask`).
    Band ratios are normalised by the coarsest level so a contrast difference (which
    N2's affine owns) is not double-corrected as sharpness.
12. Laplacian pyramid goes into `mmx_utils/frequency.py`; not DetailReinject.

**N4**
13. Oklab is new: `mmx_utils/color_space.py`, ported from Björn Ottosson's reference
    (public domain / MIT), with sRGB↔linear alongside. Runs after N2.

**N5**
14. Motion guard: flag only if spatial outlier AND temporal outlier AND the
    neighbours agree (`|I(t-1) - I(t+1)| < k·MAD`) — an isolated spike, not motion.
    Repair with `median(I(t-1), I(t+1), spatial_median)`. Test on a moving clip: no
    change > 1e-4 outside the damage mask.

**Registration**
15. `reference_color_match` ends with `color_match` — never add a generic
    `color_match` branch. Never use `endswith("detail")` or `endswith("latent")`.
    Give each node its own module and branch.

**Pipeline order**
`[core LoadVideo]` → SubjectCrop → encode → sampler with **N1**-patched model → decode
→ **N3** DetailMatch (crop space) → **N2** PlateRestore → **N4** ReferenceColorMatch
→ **N5** PixelRepair → write.
