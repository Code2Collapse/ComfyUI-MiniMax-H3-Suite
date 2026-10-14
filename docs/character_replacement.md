# Character replacement with H3 — six workflows

Six ready-to-load workflows in `workflows/character_replacement/`, one per task. They are built from the pack's
existing nodes and ComfyUI core nodes only, and every one has a note on the canvas with its own short guide.

**Status.** Each graph was built in the real ComfyUI front end (core 0.39, front end 1.53.10). It was then reloaded in
the classic and Nodes 2.0 renderers: no unknown node types, no unconnected required inputs, and the prompt
serialises. The server validated each queued graph. The only errors were the files that do not exist on the test
machine (models, plate, mask, character). **None of them has been run on H3 weights yet.** The settings below are
starting values taken from ComfyUI's official H3 reference-to-video template, not tuned results. See
[Before trusting the output](#before-trusting-the-output).

| # | Workflow file | Use it when | Head Size Mode | Extra inputs |
|---|---|---|---|---|
| 1 | `1_replace_keep_composition.json` | A different person plays the shot exactly as filmed | keep original composition | — |
| 2 | `2_keep_identity_change_pose.json` | The new character moves like another clip | match character proportions | driving video (`<Video 2>`), pose render of the driving clip |
| 3 | `3_keep_pose_change_look.json` | Same pose and motion, new face / hair / wardrobe | match character proportions | pose render of the plate |
| 4 | `4_transfer_expression.json` | Same person, expression / eyes / lips from another clip | keep original composition | driving face video (`<Video 2>`) |
| 5 | `5_replace_masked_subject.json` | Replace whatever the mask covers (a person, a prop) | keep original composition | — |
| 6 | `6_replace_across_video.json` | One constant identity over a whole take with movement | match character proportions | — (tracked crop) |

## What every workflow does

```
plate video ─┬─────────────────────────────────────────────────────────┐
mask video ──┴─> H3 Subject Crop ──> cropped plate ─┐                   │
                    │  head_room_masks ─────────────┼─> latent mask ─┐   │
                    │  bboxes ──────────────────────┼───────────────┼─> H3 Subject Uncrop ─> video
character image ────┼─> Reference to Video <────────┘               │        ^
H3 Family Presets ──> Prompt Preview & Edit ──> (prompt)            │        │
                                       │                            v        │
                                       └──> Masked Replace ──> guards ──> sampler ──> decode
```

1. **H3 Subject Crop** cuts a steady window around the masked subject. It leaves room for the new head, as set by
   Head Size Mode, and outputs `head_room_masks`: the region the new head may occupy.
2. **H3 Family Presets** writes the prompt from a fixed template for the task. `<Video 1>` is always the cropped
   plate (framing, motion, timing). `<Picture 1>` is the character. The template also names the expression
   source, the head size, the pose and what stays untouched. The same settings always give the same prompt.
3. **MiniMax H3 Prompt Preview & Edit** shows that prompt after a run. Typing in its box turns `use_edited_text`
   on, and from then on your edit is sent instead. Turn it off to take the template again, for example after
   changing the family. For a free-form prompt, the Promptor LLM director still works in its place. **MiniMax H3 Prompt Writer v2** can stand in for Family Presets: the same template, refined from a plain-language brief by a local or cloud model, with the template as its fallback.
4. **Reference To Video** (core) takes the character as `ref_image_0` and the cropped plate as `ref_video_0`. Tasks 2
   and 4 also take the driving clip as `ref_video_1`.
5. **H3 Mask To Latent Space**: `head_room_masks` becomes the latent mask. **H3 Masked Replace** then keeps
   everything outside it.
6. The sampling guards (**Protected Layer Guard**, **Accelerator Conflict**, **Sigma Shift (Ratio Locked)**,
   **Legal Scheduler**) sit between the model and the sampler. Tasks 2 and 3 add **H3 Mask-Aware ControlNet**
   with the pose render, which steers only the regenerated region.
7. **H3 Subject Uncrop** pastes the decoded crop back into the full plate through the same `head_room_masks`. The
   regenerated region and the composited region are then the same region, and outside it the plate is unchanged.

## Step by step

1. **Models.** The workflows name these files. Pick your own copies in the loaders if the names differ.

   | Loader | File | Folder |
   |---|---|---|
   | Load Diffusion Model | `minimax/minimax_h3_ref2va_bf16.safetensors` | `models/diffusion_models` |
   | Load CLIP (type `minimax`) | `qwen3vl_32b_minimax_h3_bf16.safetensors` | `models/text_encoders` |
   | Load VAE "video VAE" | `MiniMax/minimax_h3_video_vae_fp16.safetensors` | `models/vae` |
   | Load VAE "audio VAE" | `MiniMax/minimax_h3_audio_vae_fp32.safetensors` | `models/vae` |
   | Load Model Patch (tasks 2, 3) | `minimax_h3_fun_controlnet_union_pruned_bf16.safetensors` | `models/model_patches` |

2. **Plate.** Load the shot in `plate`. H3 needs a frame count of 17·n + 5 (22, 39, … 124, 141). Trim the clip first.
   Otherwise the run stops with the pack's frame-grid message, which lists the legal lengths.
3. **Mask.** Load a mask video in `subject mask`. White means replace, and the red channel is read. It needs one
   mask frame per plate frame. Any segmenter works. Save its mask as a video, for example SAM 2/3 or the C2C
   mask nodes. What to cover depends on the task:
   - tasks 1, 3, 6: the whole performer;
   - task 2: the performer, grown to cover where the new pose will reach. Only the masked region is regenerated, so
     an arm that moves outside it gets cut off;
   - task 4: the **face only**, so the head, hair and body stay the plate's;
   - task 5: exactly the thing to replace.
4. **Character.** Load the new character in `character`, sharp and evenly lit. For **match character proportions**,
   use a picture with the shoulders visible.
5. **Head Size Mode.** Set the **same** mode on **H3 Subject Crop** and **H3 Family Presets** (the workflow already
   does this):
   - **match character proportions**: the new head keeps the character's own head-to-body ratio and sits on the
     performer's neck. Connect `original_pose` (pose of the plate) and `reference_pose` (pose of the character
     image) from any pose estimator (DWPose, SDPose, controlnet_aux) to measure the size difference. Or type it
     into `head_scale`, for example 1.15. With neither, the scale is 1, no extra room is made, and the Subject Crop
     report says so;
   - **keep original composition**: the new head takes the original head's size and screen position.

   The head is always scaled evenly about the chin, never stretched.
6. **Driving clip** (tasks 2 and 4). Load it in `driving (<Video 2>)`. Use the same frame count as the plate.
7. **Pose render** (tasks 2 and 3). Load a drawn pose video in `pose render`, for example a DWPose or SDPose render,
   or **H3 Swap Control** for face-only landmarks. Task 2 uses a render of the **driving** clip, task 3 a render of
   the **plate**. **H3 Mask-Aware ControlNet** keeps it off the plate you keep (`preserved_strength` 0).
8. **Optional, experimental: face close-up.** Lynx gives its identity path a tight, aligned face. The H3 equivalent
   is a second reference. Run **H3 Face Track + Crop** on the character image (`crop_factor` about 1.6). It needs
   an ultralytics face detector such as `face_yolov8m.pt` in `models/ultralytics/bbox`. Wire its `crops` output into
   Reference to Video `ref_image_1`. Then turn on `face_closeup_reference` on Family Presets, which names
   it `<Picture 2>`, the authority for facial identity. Expression transfer ignores it. This is not A/B-tested yet;
   see run E below.
9. **Prompt.** Set `subject_name` on Family Presets, for example "a woman in a red coat". Add details in
   `task_notes`. Queue once. Read the prompt in Prompt Preview & Edit and change it there if needed.
10. **Run.** The output is saved under `output/h3_replace/<task>`.

## Starting settings

| Node | Setting | Value | Why |
|---|---|---|---|
| H3 Subject Crop | `mode` | `combined` (task 6: `tracked`) | combined holds one window still. tracked follows a subject that crosses the frame |
| | `crop_scale` | 1.5 | leaves context around the subject |
| | `divisible_by` | 32 | H3's latent token is 32 px |
| | `upscale_megapixels` | 1.0 | small subjects get enough pixels |
| H3 Mask To Latent Space | `spatial_method` | `max` | a token that touches the mask is regenerated, so there is no gap at the edge |
| H3 Mask-Aware ControlNet | `strength` / `preserved_strength` | 1.0 / 0.0 | control acts only inside the regenerated region |
| BasicScheduler | `scheduler`, `steps`, `denoise` | simple, 20, 1.0 | the official H3 r2v template |
| KSamplerSelect | `sampler_name` | `res_multistep` | the official H3 r2v template |
| Guider | | BasicGuider (no CFG) | the official H3 r2v template |
| RandomNoise | `noise_seed` | 123 | fixed, so A/B runs are comparable |
| H3 Subject Uncrop | `feather` | 16 | soft seam. `cropped_masks` = `head_room_masks` keeps the paste to the regenerated region |

## Before trusting the output

These workflows encode what the model is *told*. What it *does* with that has not been measured yet. Run this A/B
on a GPU with H3 weights before relying on any of the following: the head size, the expression transfer or the
identity match. Use a fixed seed (123) and the settings above unless a row says otherwise.

| Run | Change | Measure |
|---|---|---|
| A | your previous workflow, unchanged | head height output/plate per frame, landmark offset from the plate's face, mask-edge deviation, ArcFace similarity to the character, outside-mask pixels equal to the plate |
| B | this pack's crop with Head Size Mode | same |
| C | B + the cropped plate as `<Video 1>` and the template prompt | same + mouth / eye openness against the plate |
| D | C + pose control through H3 Mask-Aware ControlNet | same |
| E | C + the aligned face crop as `<Picture 2>` (`face_closeup_reference`) | ArcFace similarity to the character, against C |
| T5 | best of A–E + **H3 Identity Fusion Controller** (model after the loader / Mask-Aware ControlNet, `positive` from Reference to Video, `replacement_mask` = the latent head-room mask; then add a mouth + eyes `expression_mask`) | everything above; identity inside vs leakage outside the region |

`tools/h3_replace_bench.py` measures every column of this table. `score` scores one output, `run` drives the whole matrix on a ComfyUI server, and `report` prints the table. See the README.

What is already pinned by tests on CPU: the crop window holds the target head on every frame. The head-room mask
covers that head plus its padding. Outside the mask the result equals the plate exactly. Crop then uncrop maps a
point back within 1 px.

## Things to know

- H3 has no face or expression input. Identity comes from `<Picture N>` and structure and expression only from
  pixels: the kept plate, a reference video, or the ControlNet's control video. Expression transfer (task 4) relies
  on the model reading `<Video 2>` as the expression source, and that is untested.
- Workflows saved before this release could load **H3 Subject Uncrop** with an empty `feather`, and four other
  nodes lost values the same way. ComfyUI's front end drops saved values at force-input positions when the counts
  happen to line up. The pack now corrects this on load. Open and re-save an older workflow once.
- The validation step on a machine without the models reports "not in list" for every model. That only means the
  files are missing.
