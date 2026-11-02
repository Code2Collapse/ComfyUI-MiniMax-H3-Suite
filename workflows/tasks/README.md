# MiniMax H3 task workflows

Twenty-four ready-to-load workflows, one per task. Drag a file onto ComfyUI. Every workflow has:

- a guide note on the left: the models it needs, the checkpoint variants, the inputs, every control, the output and
  the limits;
- titled groups from left to right (Models -> inputs -> conditioning -> sampling -> decode -> composite / output);
- download links on the model loaders: ComfyUI's missing-models dialog offers the official files.

**Status.** Every graph was built in ComfyUI's own front end (core 0.39, frontend 1.53.10) and checked three ways:
- it reloads in both renderers;
- it converts to a prompt;
- the server validates it, reporting only the model and media files absent on the build machine.

None of them has been run on H3 weights: the build machine has no GPU weights. The starting values follow core's own
H3 templates (res_multistep, simple, 20 steps, guider cfg 1). Treat quality settings as starting points.

| # | File | Task | Model |
|---|------|------|-------|
| 1 | `01_text_or_image_to_video` | text to video; first frame / first + last frame to video; optional Turbo LoRA | fl2va |
| 2 | `02_reference_to_video_guides` | reference pictures + guide images anchored at chosen frames; Geometry Report | ref2va |
| 3 | `03_reference_modification` | edit one image with an instruction and up to 8 references (frame selector picks the still) | ref2va |
| 4 | `04_masked_character_replacement` | replace a masked subject with a character - zero-offset geometry, plate sound kept, pixels outside the mask bit-exact | ref2va |
| 5 | `05_masked_generation_controlnet` | 4 + a pose / depth / edges control, applied only where regenerated, on the plate's own crop | ref2va + Fun ControlNet Union |
| 6 | `06_masked_edit_exact_composite` | prompt-only regional edit, sound regenerated only on masked frames, transform ledger shown | ref2va |
| 7 | `07_hires_crop_regenerate_restore` | one stable high-resolution crop, re-sampled at low denoise, detail + grain matched, restored into the plate | ref2va |
| 8 | `08_long_form_continuation` | long video one segment per queue, resumable (H3 Continuation) | fl2va |
| 9 | `09_long_form_reference_mask` | edit a LONG plate segment by segment with a reference and its mask video (+ optional pose control) | ref2va (+ ControlNet) |
| 10 | `10_audio_video_continuation` | long video whose picture follows a fixed long soundtrack | fl2va |
| 11 | `11_face_head_performance` | new face / head, the plate's performance kept (Identity Fusion, experimental); optional driving performance | ref2va |
| 12 | `12_full_frame_character_object_background` | whole-frame edit of a character / object; Invert Mask to edit the background instead | ref2va |
| 13 | `13_hires_enhancement` | upscale + re-sample in tiles (H3 Split Upscale) at low denoise, source as <Video 1> | ref2va |
| 14 | `14_latent_persistence_sampler` | locked sigma shift, schedule inspector, optional First-Block Cache; save the AV latent, then (stage B) give the saved clip a new sound with its picture kept exactly | ref2va |
| 15 | `15_attribute_transfer_job` | blood / dirt / wounds / makeup on a whole plate as a JOB: one click, segment by segment, plate + matte decoded per window, outside the mask kept and checked, failed segments redone alone, resumable, the plate's sound restored; the GPU stage is one native subgraph | ref2va |
| 16 | `16_character_swap_job` | replace ONE person in a whole plate as a JOB (body double): SAM 3.1 picks and tracks the person (losses re-identified or sent to review), an approved anchor is the new character's picture, props checked, then workflow 15's job engine regenerates only the tracked person; the plate's sound kept | ref2va + SAM 3.1 |
| 17 | `17_wardrobe_transfer_job` | TASK MODE: one garment (`jacket:10`, any word) changed through a whole plate as a job - tracked, anchor-approved, only the garment regenerated | ref2va + SAM 3.1 |
| 18 | `18_prop_replacement_job` | TASK MODE: a hand-held prop (`phone:10`) replaced, held and moved as in the plate; the prop check guards it | ref2va + SAM 3.1 |
| 19 | `19_object_replacement_job` | TASK MODE: an object in the scene (`car:10`) replaced with the same position and motion | ref2va + SAM 3.1 |
| 20 | `20_background_replacement_job` | TASK MODE: everything BEHIND the tracked person replaced (matte inverted), the person protected by QC | ref2va + SAM 3.1 |
| 21 | `21_body_swap_job` | TASK MODE: workflow 16 (one person replaced) as one task node | ref2va + SAM 3.1 |
| 22 | `22_face_swap_job` | TASK MODE: one face (`face:20`) replaced through a whole plate, the performance kept (workflow 11's chain: Identity Fusion, exact uncrop), anchor-approved; the generation hears each window's sound | ref2va + SAM 3.1 |
| 23 | `23_head_swap_job` | TASK MODE: one head and its hair (`head:20`) replaced, the performance kept | ref2va + SAM 3.1 |
| 24 | `24_lip_sync_job` | TASK MODE: one person (`face:20`) speaks a NEW soundtrack - H3 Media Conform puts the sound on the plate, every window and the movie carry it, the identity stays (the plate's first frame) | ref2va + SAM 3.1 |

Workflows 17-24 are in TASK MODE (Rev 3 Part 16): the whole pipeline is ONE node whose widgets are the task's controls (target word, prompt, selection, review, anchor decision, props); open it for the editable graph with every stage. They share workflow 16's chain and the job engine; built and server-validated like the others, not run on H3 weights. Workflow 16 puts subject tracking and an approved anchor in front of workflow 15's job: its subject, track and anchor nodes were run on the real SAM 3.1 (CPU), and the whole job end to end with a CPU stand-in for the GPU stage (evidence L9.78). Workflow 15 is the H3 job engine around workflow 06's chain; its job logic was also run end to end on the CPU with a
stand-in in place of the GPU subgraph (evidence L9.76 in the C2C workspace).

## Inputs the workflows expect

The loaders carry placeholder file names; pick your own files in them:
- plates and mask videos: `plate.mp4`, `subject_mask.mp4`, `long_plate.mp4`;
- images: `character.png`, `first_frame.png`, `attribute_lookdev.png`;
- sound: `soundtrack.wav`.

Mask videos are white where the picture changes. Any segmenter works; save its mask as a video of the same length
and size as the plate.

- H3 clip lengths are 17k + 5 frames (124, 141, 158, ...). H3 Source AV Latent refuses other lengths and names the
  nearest legal ones.
- Sizes go on the 32 px token grid (1920x1088, not 1080).
- H3 renders 24 fps. The long-form workflows take a 23.976 fps plate frame for frame. To keep a plate's own sound
  in sync, the plate must be exactly 24 fps.

## Bypassed and muted nodes

Optional parts ship switched off and say so in their titles:
- Turbo LoRA, last frame, second reference, driving performance, pose control, Invert Mask, First-Block Cache;
- stage B of workflow 14.

Enable a node with Ctrl+B (bypass) or Ctrl+M (mute).

## Rebuilding

The files are generated by `docs/evidence/A16/workflows/build_task_workflows.py` in the C2C workspace (15:
`docs/evidence/L9.76/build_task_workflow.py`). It drives
ComfyUI's front end, so a saved file is exactly what ComfyUI writes. `tests/test_task_workflows.py` keeps them
consistent with the nodes:
- node types and link tables are intact, and pack outputs match the live schema;
- each guide note has all its sections;
- every model loader carries a download link;
- nothing is switched off without saying it is optional.
