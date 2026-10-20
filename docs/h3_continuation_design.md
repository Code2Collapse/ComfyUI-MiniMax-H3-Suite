# H3 Continuation: design (A16 Part IV)

Replaces the H3 Motion Context chain for long, resumable generation. Written before the code (A7).

## What the current Motion Context does wrong (from its source)

`mmx_utils/motion_context.py` (port of NikoDemon80/ComfyUI-H3-Motion-Context, GPL-3.0):

1. **Negative anchors by default.** The audio context keyframe sits at `end_frame - ref_audio_t / (5/3)`: with the
   defaults (22 video frames, 24 audio frames = 40 ticks) that is frame -2, before the target's time origin.
2. **Audio and video contexts end at different times.** 22 frames = 36.67 audio ticks, rounded to 37: the audio context
   ends 8.3 ms after the video context in every segment.
3. **The context is paid for twice.** The previous tail is added as conditioning rows AND the target regenerates the same
   frames, which are then trimmed: extra tokens in every attention step for frames that are thrown away.
4. **No chain.** `MiniMaxH3_MotionContextChain` is a placeholder; the user sets `clip_index` by hand on Save / Load each
   run. No record of which segment is which, no acceptance, no resume, no integrity check, no drift measurement.
5. **Pixel-path stretch.** A previous clip of another aspect is stretched (`crop = "disabled"`).

## The new mechanism: the tail lives IN the target, pinned by the denoise mask

Segment k is an ordinary H3 clip of L legal frames (17n+5). Its first O frames are the last O frames of the previous
ACCEPTED segment, carried as their exact latent tokens (video rows and audio ticks), with denoise mask 0 there and 1
elsewhere. Core's sampler resets masked rows to the given latent every step and at the end (KSamplerX0Inpaint), so the
pinned tokens come out unchanged (checked on every commit), and H3's per-row timestep tells the model those rows are
finished. The model sees the context as the start of its own clip - continuous positions, no RoPE patch, no extra
tokens.

- O must be a whole number of H3 latent steps from a cycle start: 5, 22, 39, 56, ... (17m + 5), and the tail of a
  17k+5 latent always starts on a cycle (proved in code). **Default O = 39**: 39 frames = exactly 65 audio ticks, so
  audio and video contexts end together. 22 and 56 are allowed and their seam offset (8.3 ms, 6.7 ms) is reported.
- Accepted output never changes again: only frames after the head of each segment are appended.

## Global time accounting (no lost, duplicated or shifted frames; no audio drift)

- Segment 0 covers global frames [0, L). Segment k (k > 0) covers [S_k, S_k + L) with S_k = E_{k-1} - O, and adds
  frames [E_{k-1}, S_k + L). E is always the number of accepted frames.
- Audio is cut by GLOBAL time, never by per-segment rounding: accepted frames [a, b) own samples
  [round(a / 24 * sr), round(b / 24 * sr)). The cumulative error is at most one sample, whatever the segment count.
- Audio phase: a segment's audio starts on the previous segment's 25 ms tick grid, so its origin sits a few ms off its
  first frame. The tick nearest the frame's GLOBAL time is taken (within 12.5 ms, never accumulating) unless that would
  start the sound so early that it ends before the segment's last frame (the clip's audio is only `ticks / 40 - frames
  / 24` longer than its picture: +8.3 ms for 124 frames / 207 ticks); then the next tick. Bound: `max(12.5, 25 - slack)`
  ms = 16.7 ms for 124 frames. Found by the 30-segment test: the first version left 4 ms of silence at seams.

## Two nodes and an on-disk session

`H3 Continuation Next` (before the sampler):
- reads the session folder (created on first use), refuses a corrupt one with a plain message;
- a segment left `running` by a crash becomes `failed` and its interval is generated again (`rerunning`);
- a segment `awaiting approval` blocks the next one until the user decides (`decision` = accept / reject);
- builds the segment latent: the template latent (Empty H3 AV latent / Reference to Video's latent, which fixes size and
  length) with the head replaced by the stored tail and the noise mask set; keyframes inside the pinned head are dropped
  (reported) because the head is fixed;
- writes the segment record (`running`) atomically; outputs latent, conditioning, segment info, report.

`H3 Continuation Commit` (after the sampler + decoders):
- checks it belongs to the running segment; checks the pinned head came back bit-exact in the latent;
- drops the head frames / samples, appends the new frames to the session output (numbered PNG sequence, readable by any
  sequence loader) and the audio (WAV per segment + global sample bookkeeping);
- stores the new tail latent (safetensors, written to a temp file then renamed, SHA-256 in the record);
- measures drift against segment 0 and the seam: colour (Lab mean / spread), sharpness (Laplacian variance ratio), seam
  jump (first new frame vs last accepted frame, relative to the clip's median frame-to-frame change), frozen /
  duplicated frames, audio level at the seam; with the approval policy decides `accepted`, `awaiting approval` or
  `rejected` (a rejected segment's interval is generated again);
- outputs the new frames, audio, report, session state JSON.

Session record per segment: id, parent, global interval, overlap, config fingerprint (sampler-relevant inputs), status,
tail file + SHA-256, output frame range, metrics, timestamps. `session.json` is rewritten atomically after every change.

Bounded memory: only one segment's latent and frames are ever in memory; everything accepted is on disk.

## Slice 2 (L8.45): per-segment sources - long plates, masks, control videos, soundtracks

Masked long-form editing, long control videos and sound-driven long video all need each segment's OWN slice of a long
input. That slice cannot be cut upstream of Next (Next decides the interval; a node feeding Next's latent cannot depend
on Next's output - a cycle). So Next takes the long inputs and cuts them itself.

- New optional inputs on Next (appended, saved workflows unchanged): `video_vae`, `audio_vae`, `source_video` (VIDEO),
  `source_mask` (VIDEO, white = regenerate), `control_video` (VIDEO), `soundtrack` (AUDIO) and the `sound` choice
  (generate / keep the source video's sound / keep the soundtrack / regenerate where masked).
- VIDEO inputs stay lazy: only the segment's frames are decoded (core's VideoFromFile.as_trimmed + get_components,
  float decode, the same path as core's Load Video). Core trims by absolute presentation time, so the stream's own start
  time is added (a file whose first frame is not at t = 0 would otherwise shift every segment by that many frames) and
  the count of every slice is checked. Frames are taken by INDEX: a 23.976 fps plate is conformed frame for frame to 24
  fps (reported); its own sound can only be kept at exactly 24 fps.
- With `source_video` the segment latent is the plate's frames encoded (H3 Source AV Latent's code: refuses off-grid
  sizes, mask reduced on H3's grouping), its head replaced by the previous tail; noise mask = the source mask with the
  head rows / ticks at 0. The template latent still sets the segment length and must have the plate's size.
- Sound: kept sound (source or soundtrack) is cut on GLOBAL time from the segment's audio origin for the segment's tick
  count and pinned (mask 0); "regenerate where masked" pins the ticks of unmasked frames (H3 Source AV Latent's rule).
- Outputs appended: `segment_source` (the plate frames), `segment_mask` (pixel mask), `segment_control` (control frames)
  - for H3 Mask-Aware ControlNet and for an exact composite before Commit (core Image Composite Masked).
- The end: a slice shorter than the segment marks the source end. The missing frames are padded (last frame repeated,
  mask kept) and Commit keeps only frames before the end; once everything is accepted, Next says the session is
  complete. A soundtrack sets the end at its last sample. No reliance on container frame-count metadata.
- Not possible in one graph: cropping the plate (H3 Subject Crop) between the slice and the encode. For high-resolution
  crops over a long plate: crop the whole plate first (stage 1, save the crop video + boxes), run the continuation on
  the crop video, uncrop afterwards.

## Temporal positions and the offset policy (A17 s4.5)

Read from the installed core (owner stack, v0.39.0-43, `comfy/ldm/minimax/model.py` `PackedLayout`):

- H3 uses a 3-D RoPE (t, h, w) over one packed sequence. Text takes t = 0..text_len-1. References are packed next,
  and the target's audio and video start at `cursor` = text_len + the references' time spans. Video rows advance by
  `FRAME_RESCALE * (1,4,4,4,4)[k % 5]` = 5/3 per pixel frame (40 Hz against 24 fps). The cycle restarts with every
  target.
- There is no offset input. Every pass, including every continuation segment, builds its positions from its own
  origin.
- Attention under RoPE depends only on position differences. Shifting a pass's target rows would not change how its
  frames relate to each other. It would only move the target away from the text and image references, to distances
  that do not occur when a clip is generated in one pass. `ComfyUI-MiniMax-H3-LongMedia` does exactly this shift
  (an `APPLY_MODEL` wrapper, `temporal_positioning.py`). Its benefit is unmeasured.
- Core's context windows do not support H3 audio-video. `MiniMaxH3` implements neither
  `map_context_window_to_modalities` nor `resize_cond_for_context_window` (`comfy/model_base.py:2152`), so a
  multimodal window raises `NotImplementedError` (`comfy/context_windows.py:244-248`).

Policy: **no positional offset**. Continuity comes from content. The previous segment's tail is pinned inside the new
target, where the positions between it and the new frames are exactly the trained ones. `H3 Context Windows` still
prints each window's start on the 40 Hz timeline, but only as a reference value; nothing applies it. A shifted-RoPE
variant can be tried only as an opt-in A/B experiment on the owner's GPU, never as a default.

## Measuring continuity (A17 s4.4)

`H3 Continuation Commit` measures every segment deterministically, with no model involved. The measurements go in the
report and in `session.json` (`metrics`):

| Measure | What it catches |
|---|---|
| pinned head, largest latent change | a sampler or node that dropped the noise mask (refused above 1e-3) |
| overlap re-decode PSNR (dB, mean and worst frame) | the pinned overlap decoded again in the new clip's context, compared with the frames already accepted: VAE context effects at the seam |
| seam jump (x the clip's median frame change) | a visible jump between the last accepted frame and the first new one |
| colour drift (CIE Lab dE of the mean vs segment 0) | slow colour drift over many segments |
| sharpness (Laplacian variance vs segment 0) | detail loss over many segments |
| frozen frames | duplicated or stalled frames |

A summary line over all accepted segments gives the worst value of each measure so far. Seam jump, colour and
sharpness can hold a segment for approval. The overlap PSNR is reported, not enforced, until real outputs show where
a limit belongs.

## Not in this slice (stated, not hidden)

- Identity memory beyond what the user's conditioning carries (references / LoRA stay in the positive conditioning of
  every segment). A reference bank of accepted anchor frames needs the text encoder per segment - next slice.
- FIFO-Diffusion (arXiv 2405.11473): diagonal denoising needs a different noise level per frame inside one pass. H3 has
  per-row timesteps (`rows_t = 1 - m * sigma`) driven by the denoise mask, so a queue of rows at increasing noise is
  expressible; but a latent row covers 1 or 4 frames on H3's (1,4,4,4,4) grid, the audio stream rides on its own
  shifted schedule, and FIFO's lookahead assumes per-frame independence of the noise level from the model's training.
  Training-free in principle; not implemented here; to be assessed with the tiny CPU model and then on the GPU.
- Visual quality and long-run drift on real outputs: owner's GPU.
