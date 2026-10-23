# H3 context-window executor: design (L8.36, A15 Phase 1.4)

Written before the code (A7). `H3 Context Windows` only plans; nothing consumed its schedule, and core's own context
windows raise NotImplementedError for H3 audio-video (D0.26). This adds **H3 Context Window Sampling**, a model patch
that runs every model call as overlapping time windows inside ONE sampling pass and fuses their predictions.

It complements H3 Continuation (L8.43), which renders accepted segments one after the other with the previous tail
pinned. Context windows keep the whole clip in one pass: every step denoises all frames, but attention runs per
window, so its cost grows with the window, not with the clip.

## Facts it rests on (core 0.39.0-43, comfy/ldm/minimax/model.py)

- Latent steps follow FRAME_PER_TOKEN (1, 4, 4, 4, 4): a 5-step cycle is 17 frames, a legal clip is 2 + 5k steps
  (17k + 5 frames). A window that starts on a cycle (step s, s % 5 == 0) and is 2 + 5k steps long has the same
  internal structure as a whole clip.
- Time positions: video step k sits at cursor + F(k) * 5/3 (F = frames before step k); audio tick a at cursor + a
  (40 ticks/s, 5/3 ticks per frame). Audio rows are channel-major: channel 0's ticks, then channel 1's.
- The target video and audio are the last two segments of the packed layout; text, references and keyframe
  conditioning come first. `_forward` uses the payload's layout when its signature matches.

## Window plan

- `window_frames` (legal, snapped like the planner) -> Lw = 2 + 5 (W - 5) / 17 latent steps; `overlap_frames` is
  snapped up to a legal length too (17m + 5 frames = 2 + 5m steps): the stride Lw - Ow must be whole cycles, and
  2 + 5k - (2 + 5m) is.
- Starts 0, stride, 2 stride, ... until a window reaches the end; every start on a cycle. The last window is cut at
  the clip's end (shorter, never shifted), so it is legal on a legal clip and longer than the overlap. T <= Lw: one
  window = the model is called untouched (bit-identical).
- Audio per window: ticks [round(F(s) 5/3), round(F(e) 5/3)); the last window ends at the clip's last tick. Exact when
  the start frame is a multiple of 3, i.e. on 51-frame (15-step) strides; otherwise the edge is off by at most half a
  tick (12.5 ms) and the report says so.

## Positions

- `global` (default): each window's rows are the full clip's layout rows for those steps and ticks, so a window
  predicts what the full clip would predict there. This is not an offset; it is the clip's own timeline (D0.26).
- `local (experimental)`: each window is laid out as a clip of its own (positions from the cursor, keyframes inside it
  re-indexed, the others left out). This is the LongMedia-style temporal shift D0.26 allows only as an opt-in A/B.

## What each window gets

| Input | Per window |
|---|---|
| video latent, video denoise mask | steps [s, e) |
| audio latent, audio denoise mask | ticks [a0, a1), both channels |
| text, references | whole |
| keyframe conditioning (FL2VA) | global: all, at their own positions; local: those inside the window, re-indexed |
| core H3 Fun ControlNet hint | encoded once for the clip, cut to [s, e) per window |
| old-style ControlNet in the conditioning, block / step caches | refused with a sentence (as Tiled Diffusion) |

## Fusion

Trapezoid weights over the overlap (latent steps for video, ticks for audio; ramps exclude their end points, so two
windows sum to 1), summed in fp32 and divided by the summed weight. Frame-local predictions therefore come back
exactly; nothing is lost, duplicated, reordered or shifted (tested).

## Budgets, compatibility, kill switch

- Attention cost per call ~ sum over windows of (rows in window)^2 instead of (rows in clip)^2; compute grows by the
  overlap share. Peak memory: one window plus the fp32 accumulators.
- Composes with H3 Tiled Diffusion: whichever wrapper was set first runs inside the other (temporal windows, each
  tiled spatially, or the reverse).
- New node; nothing existing changes. `MMX_CONTEXT_WINDOWS=off` calls the model untouched.

## Tests

- Plan: starts on cycles, legal lengths, coverage, overlap >= requested, single window for short clips; audio ticks
  cover the clip, exact on 15-step strides.
- Wrapper with stand-in models: a frame-local model -> windowed == full (video and audio); a position-reading model ->
  global positions per window, local positions restart; masks and the control hint cut along time; refusals; the
  single-window and kill-switch pass-through.
- On the tiny H3 (tests/h3_tiny.py): one window bit-identical to the plain model; several windows through core's
  sampler, finite, preserved rows kept; together with Tiled Diffusion.

Image quality (seams over time, drift) is the owner's GPU measurement; the node ships experimental until then.
