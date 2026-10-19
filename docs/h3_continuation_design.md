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

## Not in this slice (stated, not hidden)

- Identity memory beyond what the user's conditioning carries (references / LoRA stay in the positive conditioning of
  every segment). A reference bank of accepted anchor frames needs the text encoder per segment - next slice.
- FIFO-Diffusion (arXiv 2405.11473): diagonal denoising needs a different noise level per frame inside one pass. H3 has
  per-row timesteps (`rows_t = 1 - m * sigma`) driven by the denoise mask, so a queue of rows at increasing noise is
  expressible; but a latent row covers 1 or 4 frames on H3's (1,4,4,4,4) grid, the audio stream rides on its own
  shifted schedule, and FIFO's lookahead assumes per-frame independence of the noise level from the model's training.
  Training-free in principle; not implemented here; to be assessed with the tiny CPU model and then on the GPU.
- Visual quality and long-run drift on real outputs: owner's GPU.
