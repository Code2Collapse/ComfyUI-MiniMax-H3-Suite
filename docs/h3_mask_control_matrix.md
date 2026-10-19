# MiniMax H3: masks, references and ControlNet together (A16 Part III)

What can run in ONE H3 sampling pass, what needs stages, and what is not supported - read from ComfyUI's code
(0.39, `comfy/ldm/minimax/model.py`, `comfy/model_base.py`, `comfy_extras/nodes_minimax_h3.py`,
`comfy/ldm/minimax/controlnet.py`). Visual quality of each combination is **not** verified here (no H3 weights on the
build machine); "supported" means the tensors, positions and masks line up and the tests prove it.

## What one H3 pass can carry

| Input | How H3 uses it | Where it lands in space |
|---|---|---|
| Text (+ tags) | prompt rows | - |
| Keyframes / guides (`MiniMaxH3AddGuide`, first / last frame) | conditioning rows, never denoised | the TARGET's token grid - must have the target's size |
| References (`<Picture i>`, `<Video k>`, `<Audio j>`) | conditioning rows packed before the target | their OWN grid, centred and area-normalised; a token's coordinate is its top-left edge |
| Source latent + denoise mask | the latent the sampler starts from; the mask becomes a per-token timestep (`rows_t = 1 - m * sigma`) | the target |
| Fun ControlNet (control video, inpaint mask + source video) | a model patch: a residual added after layers 0/10/20/30/40 | control / mask / source are centre cover-cropped to the target |

Facts that decide the combinations:

1. Core's Fun ControlNet adds its residual to **every** video row - preserved ones too - and to the text rows; only audio
   rows are zeroed. **H3 Mask-Aware ControlNet** gates it per row by the sampler's denoise mask (`preserved_strength`
   0 = preserved rows get nothing), and by `control_region` (where this control may steer at all).
2. A plate-sized control video on a cropped target is cover-cropped by core, so it lands offset (measured: ~10 px in
   the test case). Wire **Subject Crop's `bboxes`** into Mask-Aware ControlNet `crop_bboxes`: the control / source
   / mask / region get the source's own crop and stretch.
3. A reference video of the same shot on core's 768 canvas is fed half a token up-left (-14 px for a 1440 crop):
   **H3 Image + Reference to Video** `ref_video_fit = target grid`.
4. The H3 VAE centre-crops off-16 frame sizes and drops frames off the 17k+5 grid: **H3 Source AV Latent** refuses or
   records them.
5. Core turns a pixel-frame noise mask into latent rows with a linear-in-time resample (measured: one row early at
   half strength): **H3 Source AV Latent** reduces it on H3's own (1,4,4,4,4) grouping.
6. A denoise mask never guarantees exact pixels (the VAE decodes everything): **H3 Subject Uncrop**
   `protect = exact outside mask` does, and reports it.

## The matrix

| Combination | Runs as | How (MiniMaxSuite) | Caveat |
|---|---|---|---|
| Pose guidance + masked character replacement | one pass | Source AV Latent (mask) -> sampler; Mask-Aware ControlNet (pose video, `control_region` = edit region, `crop_bboxes`); Subject Uncrop exact | the Fun ControlNet was released for the t2v weights; on ref2va weights it runs structurally, quality unverified |
| Depth + masked regeneration | one pass | same, depth video as control | same |
| Canny / edges inside an edit region | one pass | same, edges video; `control_region` keeps edges off the background | same |
| HED / MLSD | one pass if the Union model was trained on them | control video of that kind | which control types the Union model knows comes from its model card - not verified here |
| Reference identity + regional edit | one pass | Image + Reference (refs) + Source AV Latent (mask) | references sit on their own grid: a reference showing the subject elsewhere / at another scale can pull placement (open question, offset experiment V7) |
| Source video, partial regeneration | one pass | Source AV Latent with the mask | exact pixels only through Subject Uncrop |
| Several guide frames + regional mask | one pass | Add Guide (target grid) + Source AV Latent | guide frames inside the masked time range are pinned; outside, they and the source agree only if they are the source |
| Time-varying mask | one pass | Source AV Latent (per-frame mask, H3 grouping) | the mask is coarse in time: a latent row covers 1 or 4 frames |
| Soft / feathered mask | one pass | soft mask = per-token timestep (quantised to 1/256 per 2x2 token, 32 px) | fine feathering belongs in the composite (Subject Uncrop `soft mask edge`), not the latent mask |
| Stable crop | stages: crop -> one pass -> uncrop | Subject Crop -> ... -> Subject Uncrop | crop size on the 32 px grid; control / refs on the same grid |
| Long-form + masked edits | stages per segment | (Motion Context rebuild, L8.43) | - |
| AV continuation, time-restricted edit | one pass per segment | Source AV Latent `audio_mode = follow mask` | audio ticks are 1/40 s; a 17-frame stride is 28.3 ticks |
| Protected-source composite | after the pass | Subject Uncrop `exact outside mask` | reports any change where the original was kept |
| Fun ControlNet inpaint (mask + source) AND latent denoise mask | one pass | Mask-Aware ControlNet `inpaint` on, same pixel mask; latent mask from the same mask | the two masks must agree; the ControlNet mask is binarised at 0.5 |

## Constraint policy (what wins when they disagree)

| Constraint | Hard or soft | Mechanism |
|---|---|---|
| "These pixels must stay" | hard | Subject Uncrop `exact outside mask` (bit-exact, measured) |
| "These tokens may change" | hard in the latent, soft at the pixel edge | denoise mask (per-token timestep) |
| "This control applies only here" | hard (residual multiplied by 0 elsewhere) | Mask-Aware ControlNet `control_region` x denoise mask |
| "Control strength over the steps" | soft | `strength` x `schedule` inside the sigma window |
| Identity / appearance | soft | references, LoRA |
| Composition / placement | soft unless anchored | keyframes on the target grid; spatial-anchor references on the target grid |

A control never reaches a preserved token (preserved_strength 0), never reaches tokens outside its region, and the
composite never lets a protected pixel change - so the controls cannot undo the mask. What none of them can force is
the model's choice INSIDE the edit region: that stays soft guidance.
