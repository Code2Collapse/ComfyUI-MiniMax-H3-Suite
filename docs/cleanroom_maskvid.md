# Clean-room Phase A — MaskVidExperiments → MiniMaxSuite

**Targets:** `MiniMaxH3_DifferentialDenoise`, `MiniMaxH3_FrameRangeMask`
**Source read for understanding:** `third_party/MaskVidExperiments` (GPL-3.0, © 2026 drozbay)
**Output licence:** GPL-3.0 (ComfyUI-MiniMaxSuite)

This document is the behavioural specification. Phase B is implemented **from this
document**, not from the source files.

---

## Two honest caveats before anything else

**1. Clean-room is not legally required for these two targets.** MaskVidExperiments is
GPL-3.0 and ComfyUI-MiniMaxSuite is GPL-3.0, so by the project's own licence matrix a
direct port with attribution would be permitted. It is being done as clean-room anyway
for two practical reasons: it is the only route by which these behaviours could ever
move to one of the five **Apache-2.0** packs, and a direct GPL port would carry an
attribution and provenance burden on every future edit. Where clean-room *is* legally
load-bearing is the Apache targets later in this program.

**2. This is independent reimplementation, not a two-team clean room.** A strict clean
room separates the person who reads the original from the person who writes the
replacement. Here the same agent does both, so the guarantee offered is weaker and
should be described accurately: the algorithm is reimplemented from this written
specification, and the mechanical check is the no-identical-lines grep recorded in
Phase B. Do not describe the result as "clean-room" to a third party without that
qualification.

**Licence-hygiene finding:** MaskVidExperiments declares `license = {file = "LICENSE"}`
in `pyproject.toml` and states GPLv3 in `README.md:161`, but **no LICENSE file exists in
the working tree or in git**. The licence is not in doubt (the README is explicit), but
anyone redistributing that clone should be aware the file is absent.

---

## A1 — Differential denoise with a soft per-step edge

### The problem being solved

Differential Diffusion drives a *per-step* denoise mask from a single greyscale mask:
as sampling proceeds, progressively more of the image is allowed to change. Stock
implementations do this with a **hard threshold** — at each step the greyscale mask is
compared against a falling scalar and turned into a binary mask.

The consequence is that **every intermediate mask has a razor-sharp edge**, no matter
how carefully the input mask was feathered. On video this reads as a crawling, aliased
boundary, because the sharp edge lands on a slightly different pixel each frame.

### The behaviour to reproduce

Replace the hard threshold with a **linear ramp in mask-value space**, so an
intermediate mask keeps a soft edge whose *spatial* width follows the blur already
present in the input mask.

The key property, and the reason this works: for a mask with local gradient `|∇m|`, a
ramp of width `w` in mask VALUE space produces a band of width `w / |∇m|` in PIXEL
space. A heavily feathered mask has a small gradient and therefore stays wide and soft;
a sharp mask has a large gradient and stays sharp. No spatial blur is applied, so the
operation is per-pixel and temporally stable — which is what makes it usable on video.

### Schedule

The threshold is computed in **timestep space**, not sigma space, and normalised so it
runs from 1 at the first sampling step to 0 at the last:

```
t_now  = timestep(sigma_current)
t_from = timestep(sigma_first)
t_to   = timestep(max(sigma_last, sigma_min))
threshold = (t_now - t_to) / (t_from - t_to)
```

`sigma_min` is the model's own floor; the last scheduled sigma is used instead when it
is higher, so a partial-denoise schedule still spans the full 1→0 range.

### The ramp

With softness `s`:

* `s = 0` → hard comparison `m >= threshold`, i.e. the stock behaviour exactly.
* `s > 0` → a linear ramp of width `s` centred on the sweeping threshold.

The ramp's **centre traverses `[s/2, 1 - s/2]` rather than `[0, 1]`**. This detail is
what preserves the stock endpoints: pixels at mask value 1 must denoise fully from the
very first step, and pixels at 0 must never denoise. A ramp centred on `[0, 1]` would
violate both ends — value-1 pixels would ramp in late and value-0 pixels would leak.

* `s = 1` degenerates to blending by the raw mask, i.e. no schedule at all.

### Strength

A final blend between the scheduled mask and the raw input mask, so the effect can be
dialled back without changing the schedule: `out = strength*scheduled + (1-strength)*raw`.

### Interface

MODEL in, MODEL out. The function is installed via the host's
`set_model_denoise_mask_function` (`comfy/model_patcher.py:658`) on a cloned patcher.

### What Phase B does differently (deliberate divergence)

* The ramp is expressed as explicit **band edges** (`lo`, `hi`) rather than as an
  algebraic rearrangement. Same result, and it makes the endpoint reasoning legible.
* The mask maths is a **pure function** taking `(mask, threshold, softness, strength)`,
  with no model or `extra_options` involved. The original couples it to the sampler
  callback, which cannot be unit-tested without standing up a model. Ours is testable
  directly, and the schedule maths is separated into its own pure function too.
* Input validation and a `report` string (project rule R8), which the original has not.

---

## A2 — Frame-range mask

### The problem being solved

Choosing which frames of a clip to inpaint normally means building a mask batch
elsewhere and hoping the frame count matches. Direct authoring of "frames 0–24 and 40"
is not otherwise available.

### The behaviour to reproduce

Parse a comma/newline separated list of frame numbers and Python-style slices into a
set of frame indices, then emit a MASK batch where selected frames are fully masked
(1.0) and the rest empty (0.0), at a requested resolution.

**Accepted syntax**

| Form | Meaning |
|---|---|
| `12` | single frame |
| `0:24` | Python slice semantics — **stop excluded** |
| `0:24:2` | with step |
| `:24`, `40:` | omitted bound |
| `100:end` | `end` as an alias for an omitted stop |
| `-5` | negative indices count back from the end |

**Semantics that matter**

* Slice endpoints **clamp** to the batch, exactly as Python slicing does — `0:9999` on a
  24-frame batch selects all 24 rather than erroring.
* A **single frame** outside the batch **raises**, because that is what list indexing
  does, and silently dropping it would hide a real authoring mistake.
* A step of zero is an error.
* `end` is only meaningful as a stop bound; using it as a start is an error.
* Whitespace inside a segment is insignificant.
* Duplicate/overlapping ranges collapse — the result is a sorted set.

### Interface

Text + frame count + width/height in, MASK batch out `[frames, H, W]`, plus a `report`.

### What Phase B does differently (deliberate divergence)

* Parsing returns a **sorted list plus a structured diagnostic** rather than a bare
  list, so the node can report what it selected without re-parsing.
* Errors name the offending segment *and* show the accepted syntax, and are raised as a
  dedicated exception type rather than bare `ValueError`, so callers can distinguish an
  authoring error from an internal one.
* Frame count is validated as positive before parsing, so `0` frames gives a clear
  message rather than a confusing modulo error.

---

## Phase B acceptance

1. Original module in `mmx_nodes/` + `mmx_utils/`, GPL-3.0 header naming this document.
2. Tests, CPU-only, one INVARIANT comment per test, including:
   * `softness=0` reproduces the hard threshold exactly;
   * ramp endpoints hold — mask 1 denoises at step 0, mask 0 never does;
   * soft edge width scales inversely with mask gradient;
   * slice semantics: stop excluded, clamping, negatives, `end`, step, dedupe;
   * single out-of-range frame raises; out-of-range slice clamps.
3. Equivalence spot-check against the original **in a throwaway environment**, results
   recorded here, original never imported by shipped code.
4. `>3 consecutive identical lines` grep against the source: recorded, must be empty.

---

## Phase B results (recorded 2026-09-06)

**Modules shipped** — `mmx_utils/differential_denoise.py`, `mmx_utils/frame_ranges.py`,
`mmx_nodes/differential_denoise.py`, `mmx_nodes/frame_range_mask.py`,
`tests/test_cleanroom_maskvid.py`. Registration 29 → **31**.

**Tests:** 20 passed, CPU-only.

**No-identical-lines check** (blank lines, comments and docstrings excluded, since
those are prose rather than expression):

```
longest identical run: 3 line(s)   limit: 3   violations: 0   RESULT: PASS
```

The first run **failed** at 4 lines — an `io.Float.Input(...)` schema block where the
API shape plus identical parameter values forced convergence. The parameter *values*
are functional facts and were kept; the *arrangement* was changed by hoisting the input
list into a `_inputs()` classmethod, which also reads better than a nested literal.

**Throwaway equivalence** (original imported only by the harness; no shipped module
imports MaskVidExperiments):

```
A1 differential denoise: 72 configs, max |ours - original| = 0.000e+00
A2 frame ranges:         52 cases,   0 mismatch(es)
```

### One intentional divergence — a bug in the original

At **`strength = 0.0`** the two disagree, and ours is correct.

The original guards its blend with `if strength and strength < 1:`. Python treats `0.0`
as falsy, so at `strength=0` the blend is **skipped entirely** and the node returns the
fully thresholded mask — i.e. `strength=0` behaves identically to `strength=1`. Both
implementations' tooltips document 0 as "the raw input mask".

Ours returns the raw mask. Pinned by `test_strength_blends_toward_the_raw_mask`.

This is the clearest argument for writing from a specification rather than transcribing
code: the defect is invisible when you copy the line, and shows up immediately when you
implement what the documentation says. The equivalence figures above exclude
`strength=0` for that reason; every other configuration matches to 0.0.
