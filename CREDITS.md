# Credits

## Comfyui_Minimax_h3_latent_Upscaler (split / tiled upscaler)

- **Source:** [Comfyui_Minimax_h3_latent_Upscaler](https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler) by LBH-123-AI
- **Licence:** MIT
- **Ported into this pack:** `mmx_utils/split_upscale.py`, `mmx_nodes/split_upscale.py` — direct port of the split/tiled upsampler. The learned 3D latent upscaler came later through the MMH3 port (H3 Latent Upscale (Model), see below); its weights are not shipped.

## ComfyUI-MiniMax-H3-Image-Studio (still-image generation)

- **Source:** [ComfyUI-MiniMax-H3-Image-Studio](https://github.com/) (upstream vendored at `third_party/ComfyUI-MiniMax-H3-Image-Studio`)
- **Licence:** Unlicense (public domain)
- **Ported into this pack:** `mmx_utils/image_studio.py`, `mmx_nodes/image_studio.py` — H3 video model used as a still-image generator via short frame packets, decode, and frame selection. `H3SamplingSettings` and `H3WorkflowNote` were not ported (see overlap notes in pack docs). Temporal grid helpers `decoded_frames_for_latent_t` / `latent_t_for_frame_count` live in `mmx_utils/h3_grid.py`.

## MiniMax-H3-NativeAudio-MusicVideo-Workflow (native audio lock + multishot)

- **Source:** vendored at `third_party/MiniMax-H3-NativeAudio-MusicVideo-Workflow` — sub-packs `ComfyUI-H3-NativeAudioLock` and `ComfyUI-H3-Multishot` (not `ComfyUI-Spectrum-MiniMax-H3`, already ported as `MiniMaxH3_SpectrumApply`)
- **Licence:** licence not stated by upstream (no LICENSE file, no SPDX header, no README statement) — ported at the repository owner's direction, 2026-09-21
- **Ported into this pack:** `mmx_utils/native_audio.py`, `mmx_utils/multishot.py`, `mmx_nodes/native_audio.py`, `mmx_nodes/multishot.py` — exact-audio lock (video-only denoise via nested noise mask), reference-audio stereo guard, multishot script split and chained samplers. `apply_gguf_arch_patch.py`, `h3_interior_patch.py`, loader-any wrappers, and `H3Keyframes` were not ported (see overlap notes in pack docs).


## ComfyUI-MiniMaxH3-Director (storyboard Director)

- **Source:** ComfyUI-MiniMaxH3-Director by CGlide (vendored at `third_party/ComfyUI-MiniMaxH3-Director`)
- **Licence:** **GPL-3.0**, declared in its LICENSE and README badge. This pack has been GPL-3.0 since its initial commit, so the licences agree and no relicensing was needed.
- **Ported into this pack:** `mmx_nodes/director/` — `MiniMaxH3_Director`, `MiniMaxH3_PreviewOverride`, `MiniMaxH3_RetakeStitch`, `MiniMaxH3_EnhancePrompt`, `MiniMaxH3_SaveLastFrame`. Their offline suites came too, as `tests/test_director_plan.py` (305 checks) and `tests/test_director_nodes.py` (50).
- **Not ported:** the canvas timeline front-end in `js/`. It is itself a fork of the LTX Director's canvas editor, and this workspace already has a better one — WanDirector's seven-track DOM timeline. A second canvas implementation would be the weaker of the two and would have to be maintained beside it. The Director works without it: an empty `timeline_data` falls back to `global_prompt`.
- **Also not registered:** `MiniMaxH3_DirectorChain`, exactly as upstream leaves it — the backend works, there is no usable way to give it a timeline.
- **Changed here:** node ids namespaced from the upstream `*CS` names, so installing the original alongside is a visible duplicate rather than a silent collision; and the eight HTTP routes now register through a guard instead of a bare decorator. Upstream's decorator is evaluated at import time against `PromptServer.instance`, which only exists once the server is up — so importing the module before that raised AttributeError and took all five nodes out of `/object_info`. Upstream's own test harness documents working around this by faking a server; that workaround is no longer needed.

## MiniMaxH3-Director / "Muse Minimax Director" (all-in-one chunked Director)

- **Source:** MiniMaxH3-Director (vendored at `third_party/MiniMaxH3-Director`)
- **Licence:** MIT, per the upstream README badge (no LICENSE file in the repository).
- **Ported into this pack:** `mmx_nodes/director/allinone.py` — `MiniMaxH3_DirectorAllInOne`. Kept alongside the other Director rather than instead of it: that one returns conditioning and a latent to wire into a sampler, this one runs the whole pipeline internally and **chunks**, so a video longer than H3's reliable ~15s per call is split with each continuation chunk seeded from the previous chunk's own last frames and last seconds of audio.
- **Changed here:** a `prompt` input was **added**. Upstream has no prompt socket at all — every word comes from `timeline_data`, written by its own canvas timeline, which is not ported. Without it the node would generate from an empty prompt. PyAV and the HTTP route are now imported and registered defensively for the same reason as above.
- **Registration:** it is a classic-API node, and this pack registers through the V3 `comfy_entrypoint`. It is adapted by `mmx_nodes/director/v1_adapter.py` rather than registered alongside, because ComfyUI's loader takes the V1 branch and **returns** if a pack exports `NODE_CLASS_MAPPINGS` — so exporting one mapping to register one node would have silently unregistered the other ninety-seven. `tests/test_v1_adapter.py` pins that, including a check that the pack root never grows such an export.


## comfyui-deno-custom-nodes (targeted text-encoder unload)

- **Source:** Deno Custom Nodes (vendored at `third_party/comfyui-deno-custom-nodes`)
- **Licence:** **GPL-3.0-only**, declared in its README and pyproject classifiers. This pack is GPL-3.0, so the licences agree.
- **Ported into this pack:** `mmx_utils/encoder_unload.py`, `mmx_nodes/encoder_unload.py` — `MiniMaxH3_TextEncoderUnload`. Taken because it matters more on H3 than anywhere else: the encoder is Qwen3-VL-32B, used once at the start and then resident for the whole sample, and a general "free memory" would unload the diffusion model that is about to run.
- **Not ported from that pack:** the LTX-specific nodes (this is not an LTX pack), the service-API nodes, the two RTX nodes (they need NVIDIA's Video Effects SDK, a separate Windows-only install), and its Image/Video Compare nodes — those have four modes against the sixteen in `ComfyUI-CustomNodePacks`' own `video_comparer` (scopes, false colour, bit-depth crush, audio waveform/spectrogram/loudness, synced player), so porting them would have been a downgrade. Recorded here so it is not re-audited.

## ComfyUI-MiniMaxH3-FirstBlockCache + ComfyUI_H3FBC (first-block residual caching)

- **Sources:** both vendored at `third_party/` — `ComfyUI-MiniMaxH3-FirstBlockCache` and `ComfyUI_H3FBC`. Two independent implementations of the same idea.
- **Licence:** neither states one (no LICENSE file, no SPDX header, no README statement) — ported at the repository owner's direction, 2026-09-21.
- **Ported into this pack:** `mmx_utils/first_block_cache.py`, `mmx_nodes/first_block_cache.py` — `MiniMaxH3_FirstBlockCache`. **Merged, not chosen between**: each had something the other lacked and neither alone is safe for H3 video. FirstBlockCache contributed the temporal guard, the sigma window, per-uuid branch contexts, accelerator-conflict detection and the run statistics; H3FBC contributed warmup steps, metric stride and OOM tolerance.
- **Why the merge mattered:** the plain metric averages over the whole latent, so on a 124-frame clip one frame can change completely while the mean stays under threshold — that step caches and the frame freezes while the rest of the shot moves. The temporal guard decides on the worst frame instead. Neither pack's defaults caught this alone.

## ComfyUI-MiniMaxH3DualClockSampler / ComfyUI-MiniMaxH3-Cache (assessed, not ported)

- **Sources:** vendored at `third_party/`.
- **Not ported, with reasons** so this is not re-audited: the dual-clock sampler is already covered by `MiniMaxH3_TurboSampler` (legacy dual-clock Euler) plus `MiniMaxH3_DualClockShim`, which detects ComfyUI's native `ModelSamplingAV` and passes MODEL through when it is present. `ComfyUI-MiniMaxH3-Cache` is superseded by `MiniMaxH3_BlockCacheT8` and `MiniMaxH3_FirstBlockCache`, which between them cover both caching strategies with conflict detection neither original had.

## scikit-image — masked normalised cross-correlation (Padfield 2012)

- **Source:** scikit-image 0.26, `skimage/registration/_masked_phase_cross_correlation.py`.
  Algorithm: D. Padfield, "Masked Object Registration in the Fourier Domain", IEEE
  Transactions on Image Processing 21(5), 2012.
- **Licence:** BSD-3-Clause.
- **Ported into this pack:** `mmx_utils/registration.py` (torch, no scikit-image import at
  runtime — it is not a dependency of this pack). Used by `MiniMaxH3_PlateRestore` for the
  INTEGER part of the offset; the sub-pixel part is a masked Lucas–Kanade refinement
  written here, using the same bilinear warp that later applies the correction so the
  estimate and the correction agree.
- **Why a port and not the plain phase correlation:** the evidence is a ring around the
  edit, not the whole frame. Zeroing outside the ring and phase-correlating biases the peak
  toward zero shift; Padfield's formulation normalises by the actual overlap at each lag.

## Oklab — Björn Ottosson

- **Source:** https://bottosson.github.io/posts/oklab/ (matrices and pipeline).
- **Licence:** public domain / MIT, as published by the author.
- **Ported into this pack:** `mmx_utils/color_space.py`, alongside the exact sRGB transfer
  function. No clamp on the linear side: values above 1.0 survive.

## MaskVidExperiments (MVEx) — drozbay (ported code)

- **Source:** https://github.com/drozbay/MaskVidExperiments (rev c32ed8c, 2026-09-11, the upstream HEAD on
  2026-10-21).
- **Licence:** GPL-3.0, the same as this suite. Copied with attribution, as the owner authorised.
- **Ported:** `mmx_nodes/subject_crop.py` and `mmx_utils/subject_crop_planner.py` (Subject Crop / Uncrop and the crop
  planner), `mmx_nodes/mask_cleanup.py`, `mmx_nodes/mask_to_latent.py` and `mmx_nodes/audio_mask.py`. Each carries a
  `# PORTED FROM:` header. `mmx_utils/crop_planner_tv.py` is a textbook formulation, not derived from MVEx.

## NKD Basic Tools — Nekodificador (behaviour reference)

- **Source:** https://github.com/Nekodificador/ComfyUI-NKD-Basic-Tools (`nkd_av_latent.py`, rev 227b9c2).
- **Licence:** MIT.
- **Used for:** behaviour only - how a masked joint AV latent is assembled for H3 (both VAE encodes, the audio length
  rule, keep / regenerate / follow-mask audio). `MiniMaxH3_SourceAVLatent` is an original implementation; no code
  was copied. It differs on purpose: it refuses or records off-grid sizes instead of letting the VAE centre-crop
  them, and reduces the mask on H3's own frame grouping.

## H3 Composer scaffolds — 1038lab (Comfyui-Minimax-H3-Promptor)

- **Source:** https://github.com/1038lab/Comfyui-Minimax-H3-Promptor `web/js/prompt_composer.js` TEMPLATES (rev bdbde80).
- **Licence:** GPL-3.0 (same as this pack).
- **Ported into this pack:** the eight mode scaffolds in `mmx_nodes/promptor.py` (`COMPOSER_TEMPLATES`).

## Ports of 2026-10-22 (owner: "port all relevant nodes of MiniMax H3")

The decisions for every pack in both reference folders are in the work area's
`docs/research/minimax_port_plan.md`. The licence texts of the MIT sources are in `THIRD_PARTY_LICENSES/`.

### ComfyUI-H3-Context-Noise: beijinren

- **Source:** https://github.com/beijinren/ComfyUI-H3-Context-Noise (rev 7e55312).
- **Licence:** MIT, Copyright (c) 2026 beijinren.
- **Upstream sources:** the palette, the 36x64 grid and `alpha_for` come from MacroSony's
  https://github.com/MacroSony/minimax-h3-chained-character-swap (MIT, Copyright (c) 2026 MacroSony).
- **Ported:** `mmx_utils/context_taper.py` and `mmx_nodes/context_taper_noise.py`, as H3 Context Taper Noise
  (frames, latent).
- **Changed:** V3 nodes. Behaviour unchanged.

### ComfyUI-H3-Motion-Context-MultiRef: seitanism (keyframe nodes by Reithan)

- **Source:** https://github.com/seitanism/ComfyUI-H3-Motion-Context-MultiRef (rev 361624f).
- **Licence:** GPL-3.0.
- **Ported:** `mmx_nodes/av_noise_mask.py`, as H3 Set / Clear AV Noise Mask and H3 Hard Keyframes (from
  `MiniMaxH3CustomKeyframesMasked`).
- **Changed:**
  - The frame mask is reduced on H3's own grid, not by trilinear resampling.
  - Keyframes come as one image batch plus a list of positions, so no custom front end is needed.
  - Video and audio lock switches were added.
  - The pre-#15375 shim was dropped; the nodes need native nested AV masks.
- **Not ported:** the soft Custom Keyframes node, because core's Add Guide does the same.
- **Also ported (L8.47):** `mmx_utils/h3_bridge.py` and `mmx_nodes/av_bridge.py`, as H3 AV Bridge and H3 AV Bridge
  Assemble. Their sources are `h3_masked_bridge.py`, `h3_audio_grid.py`, `MiniMaxH3AssembleBridgeAudio` and the audio
  helpers. Here the frames and the sound are assembled in one node, and the clips are resized to the decoded canvas
  when they differ.

### H3-LongTakeNoCuts: xyzDist

- **Source:** https://github.com/xyzDist/H3-LongTakeNoCuts (rev 556085d).
- **Licence:** MIT, Copyright (c) 2026 xyzDist.
- **Ported:** `mmx_nodes/latent_blend_frames.py`, as H3 Blend Latents By Frames, H3 Keyframe Mask and H3 Mask Levels.
- **Changed:**
  - Keyframes map onto H3's real latent steps, and the frame count comes from the latent.
  - Masks are reduced on H3's grid.
- **Not ported:** H3 Video Lock. Set AV Noise Mask's `video_lock` does the same thing with core's nested mask
  layout.

### Herrgotts-H3-Infinite-Continuation-Suite: HerrgottMargott

- **Source:** https://github.com/HerrgottMargott/Herrgotts-H3-Infinite-Continuation-Suite (rev 4b1edd6).
- **Licence:** GPL-3.0.
- **Ported:** `mmx_utils/freeze_tail.py`. This is the stable-tail consensus freeze detector (`analyze_freeze_tail`)
  with its calibrated defaults.
- **Used by:** H3 Continuation Commit's `freeze_tail` option. It reports a frozen FL2VA ending, or cuts the segment
  back to the longest legal 5 + 17k length before the freeze.
- **Not ported:** the upstream continuation nodes and their phase snapping. The suite's H3 Continuation already does
  that part.

### ComfyUI_MinimaxH3_AutoContext: supElement

- **Source:** https://github.com/supElement/ComfyUI_MinimaxH3_AutoContext (rev bfa7ac3).
- **Licence:** Apache-2.0. The text is in `THIRD_PARTY_LICENSES/ComfyUI_MinimaxH3_AutoContext-LICENSE.txt`.
- **Ported:** `mmx_utils/seam_correction.py` (the engine, kept as it is) and `mmx_nodes/seam_correction.py`, as H3 Seam
  Correction.
- **Changed:**
  - PySceneDetect is optional and imported lazily.
  - The report is in English.
  - The `cut_detection` switch is exposed (upstream reads it but never offers it).
- **Not yet ported:** the TST attention patch (planned as experimental, L8.50), the face-fix chain, and the
  auto-context sampler.

### Comfyui-MMH3-UltimateUpscale: bbaudio-2025 (L8.37)

- **Source:** https://github.com/bbaudio-2025/Comfyui-MMH3-UltimateUpscale (rev 0a8d5ee).
- **Licence:** MIT, Copyright (c) 2026 bbaudio-2025. The text is in
  `THIRD_PARTY_LICENSES/Comfyui-MMH3-UltimateUpscale-LICENSE.txt`.
- **Upstream sources:** MMH3 Tiled Diffusion follows shiimizu/ComfyUI-TiledDiffusion (MultiDiffusion / Mixture of
  Diffusers). Its H3 3D latent upscaler is the inference code of LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler (MIT,
  Copyright (c) 2026 LBH-123-AI; text in `THIRD_PARTY_LICENSES/Comfyui_Minimax_h3_latent_Upscaler-LICENSE.txt`). The
  checkpoints are huggingface.co/LBH-123-AI/Minimax_h3_latent_Upscaler (Apache-2.0); they are not shipped.
- **Ported:** `mmx_utils/tiled_diffusion.py` + `mmx_nodes/tiled_diffusion.py`, as H3 Tiled Diffusion (experimental);
  `mmx_utils/h3_latent_upscaler.py` + `mmx_nodes/latent_upscale_model.py`, as H3 Latent Upscale (Model).
- **Changed** (design: `docs/h3_tiled_sampling_design.md`):
  - Tile starts are always even. Upstream put the last tile on an odd latent row for odd sizes (720p), off the DiT
    patch grid.
  - Tile predictions are summed in fp32; positive and negative keep their own tile plan.
  - Video denoise masks are cropped per tile (upstream refused them).
  - Core's H3 Fun ControlNet (a model patch) gets its own crop of the hint per tile. Upstream only checked the old
    ControlNet field, so on core the whole control video was squashed into every tile.
  - Block / step caches and the Mask-Aware ControlNet are refused with a sentence; a single tile calls the model
    untouched; kill switch `MMX_TILED_DIFFUSION=off`; trapezoid weights (the LTX-Enhanced rule) next to gaussian and
    uniform.
  - The latent upscaler is one node that returns the upscaled latent, with optional cross-faded temporal chunks;
    the normalisation values are the weights repo's `config.json`.
- **Not ported:** MMH3 Ultimate Upscale itself and its temporal / spatial / Fun ControlNet / inpaint param nodes. H3
  Split Upscale (the LBH port above) already does the temporal + spatial split; Tiled Diffusion replaces the spatial
  split; the Fun ControlNet param nodes target a ControlNet API core did not merge (core loads the H3 Fun ControlNet
  as a model patch). The LTX 2.5 nodes belong to the LTX-Enhanced pack.
- **Also ported (L8.51), into H3 Split Upscale / H3 Spatial Split Params:** the even-size sampling of odd edge tiles
  (here the pad repeats the edge instead of zeros, and tile starts are kept even too), the per-step dynamic fade, the
  per-frame brightness match, equal tiles (on the 2-cell grid), the overlap blend shape and masked-area noise.

### ComfyUI-MiniMaxH3-Context-Loop: ethanfel (NikoDemon80) (L8.48)

- **Source:** https://github.com/ethanfel/ComfyUI-MiniMaxH3-Context-Loop (rev 43afeff).
- **Licence:** GPL-3.0, Copyright (C) 2026 NikoDemon80.
- **Ported (backend only):** `mmx_utils/tagged_refs.py` + `mmx_nodes/tagged_refs.py`, as H3 Tagged Picture / Video /
  Audio and H3 Tagged Reference to Video: the tag grammar, strict / soft / disabled compliance, the native numbering
  and H3's 9 / 3 / 3 capacity.
- **Changed:**
  - The prompt alone activates a reference; there are no scenes or Plan (the suite's H3 Continuation runs segments).
  - Numbering follows the order of the chain. Upstream sorts each kind by tag name, although its tooltips promise
    chain order.
  - Tagged Reference to Video calls core's MiniMax H3 Reference to Video; the labels are tested against core's own
    presentation order.
- **Not ported:** the Scene Prompt Editor @ tray, Project Asset Carousel and Plan Studio front end (their UI is never
  ported), Qwen-only `#anchors`, the reference cache, the external RefMod backend, sequential / source-timeline modes
  that need Context-Loop's scene state, and the fingerprint lineage.
