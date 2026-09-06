# MiniMax H3 Suite — Work Log

**REGENERATED 2026-08-28 from a live inventory, not copied from a prior doc.**
Every number below came from a command run against the tree on that date. Per the standing rule,
any future brief's "what exists" section must be regenerated the same way — never inherited.

Regeneration commands:
```
pytest tests/ -q | tail -1
ls mmx_nodes/*.py mmx_utils/*.py tests/test_*.py | wc -l
<registration smoke test, production conditions — see §7>
```

---

## 1. Live inventory

| | |
|---|---|
| **Nodes registered** | **25 / 25** (verified under production conditions, §7) |
| **Tests** | **247 passed**, 0 failed |
| Node modules | 25 (`mmx_nodes/`) |
| Util modules | 36 (`mmx_utils/`) |
| Test files | 55 |
| Frontend widgets | 5 JS files (`web/`), 991 LOC |
| Frontend pin | `comfyui_frontend_package` **1.45.21**; `WEB_DIRECTORY = "web"` |
| Source LOC | 8,460 (nodes + utils) |
| Test LOC | 3,047 |
| Reference repos | 42 in `third_party/` (read-only) |
| **Runtime-verified against H3 weights** | **0 — no checkpoints on this box** |
| **Verified in a live browser** | **0 — see §8** |

### Nodes by category

| Category | Count | Nodes |
|---|---|---|
| `MiniMax H3/Spine` | 7 | TrackCrop · MaskPrep · StitchBack · ControlHints · MaskedReplace · DetailReinject · FrameHandles |
| `MiniMax H3/Sampling` | 7 | ProtectedLayerGuard · AcceleratorConflict · LegalScheduler · SigmaShiftLocked · PerFrameDenoise · BlockCacheT8 · SigmaInspector |
| `MiniMax H3/Control` | 4 | StrongestPose · TemporalDepth · RegionMask · PosePuppeteer |
| `MiniMax H3/Color` | 3 | OCIOBridge · ColorRoundTripQC · HDRRoundtrip |
| `MiniMax H3/Authoring` | 2 | FamilyPresets · EditValidator |
| `MiniMax H3/Bridge` | 1 | DCCBridge |
| `MiniMax H3/QC` | 1 | DriftQC |

---

## 2. Slices shipped

| Slice | Content | Tests at close |
|---|---|---|
| Stage 1 | N1 TrackCrop, N2 MaskPrep, N5 StitchBack (geometric spine) | 16 |
| Stage 2 | N3 ControlHints, N4 MaskedReplace, N6 DetailReinject, N7 FrameHandles, workflow, OOM/CPU fallback | 102 |
| Slice F | 7 sampling-stack nodes (P1/P2/P3/P6/P9/P10/P11); P4 cut, P5 = no node | 143 → 148 |
| Slice I | I1 OCIOBridge + W4 ColorRoundTripQC ("port and de-clamp") | 165 |
| Stage 3a | N8 StrongestPose, N9 TemporalDepth, N11 DriftQC | 197 → 201 |
| Stage 3b–3d | N10 RegionMask, N12 FamilyPresets, N13 EditValidator, N14 PosePuppeteer, HDRRoundtrip, DCCBridge | 233 |
| Phase C1 | Defect fixes (D6 official prompt format, D1 clip-global horizon, …) | 233 |
| **Stage 4** | **4 DOM widgets (W1–W4), `ui=` payloads, workflow repair** | **247** |

---

## 3. ⚠️ The bug that mattered most — pack registered ZERO nodes in production

Found 2026-08-26 while regenerating this file. **Every prior "19 nodes register" claim was true only
of the test harness, never of a real ComfyUI.**

Our package directories were named `utils/` and `nodes/`. Both collide with ComfyUI top-level names:
- `ComfyUI/utils/` — a real package (`__init__.py`, `extra_config.py`, `json_util.py`, …)
- `ComfyUI/nodes.py` — a real module

ComfyUI imports its own `utils` at startup, **before** custom nodes load. Once `sys.modules["utils"]`
holds ComfyUI's package, `sys.path` is irrelevant — Python returns the cached module and every
`from utils.X import` raises `ModuleNotFoundError`.

Proven by simulating a real process:
```
import utils.json_util                       # what ComfyUI does at startup
>>> NODES REGISTERED: 0            (all 19 failed: No module named 'utils.affine_transform', ...)
```
The test suite stayed green throughout, because the harness never imported ComfyUI's `utils` first.

**Fix:** `utils/` → `mmx_utils/`, `nodes/` → `mmx_nodes/`; 115 import lines across 64 files.
**Now:**
```
import utils.json_util
>>> NODES REGISTERED: 25
```
A regression test (`test_registers_even_when_comfyui_utils_already_imported`) reproduces production
conditions and asserts the occupant of `utils`/`nodes` is *not* ours, so the simulation cannot
silently become a no-op.

Two follow-on defects fixed in the same pass:
- The rename missed **string-literal** module paths (`monkeypatch.setattr("utils.ocio_config…")`) —
  invisible to an import-statement search.
- Cursor added a helper that **renamed directories as a side effect of running the tests**. Removed;
  a test run must never mutate the source tree.

---

## 4. Bugs found by running the code (16)

Cursor drafted most code but **could not run a shell in any session**, so every "done" it reported
was unverified. These are what running it caught.

| # | Where | Bug |
|---|---|---|
| 1 | `h3_constants.py` | Fallback caught `ImportError`; `comfy_kitchen` skew raises **`AttributeError`** → whole suite uncollectable |
| 2 | `crop_planner_tv.py` | **Inverted containment sign** → LP infeasible → silent fallback to an *unsmoothed* path |
| 3 | `canny_torch.py` | Sobel `conv2d(padding=1)` = zero-padding → 124/1024 spurious border-ring edges on a flat field |
| 4 | `test_h3_constants.py` | Asserted `video_latent_t(22)==3`; core says **7** |
| 5 | 4 heavy nodes | try/except wrapped only the device *query*, not the compute → OOM hard-fail on 8 GB |
| 6 | `ocio_bridge` | **Invented OCIO view name** `"ACES 2.0 - SDR Video"` — exists in no display |
| 7 | `ocio_config.py` | `getColorSpace()` **returns None** for unknown names, doesn't raise → bogus names accepted |
| 8 | `depth_hints.py` | `io.Image.Output(..., optional=True)` — **`optional` doesn't exist on Output** → broke all registration |
| 9 | `optical_flow.py` | No guard for RAFT's **128 px floor** → raw torchvision error on small crops |
| 10 | `drift_qc.py` | Hardcoded `device("cpu")` while mask stayed on cuda → device mismatch; also made `run_with_cpu_fallback` a no-op |
| 11 | `vitpose_plate.py` | Hardcoded `D:/PROJECT/...` path — dead on the Linux box |
| 12 | `vitpose_plate.py` | Pack root on `sys.path` made `models` top-level → `attempted relative import beyond top-level package`; ViTPose was **never reachable** |
| 13 | package layout | **`utils`/`nodes` shadowing — 0 nodes in production** (§3) |
| 14 | **all 4 Stage 4 widgets** | **Read socket values out of `onExecuted` — a browser-only failure** (§4a) |
| 15 | both workflow JSONs | **Mis-wired sampling chain and an out-of-range link slot** (§4b) |
| 16 | `test_js_python_parity.py` | Pasted a *copy* of the JS into Python — could never fail when the widget changed (§4c) |

**Three tests were tightened, never loosened:** canny uniform `mean<0.05` → **exactly 0**; affine
round-trip white-noise `mean<1e-2` → band-limited `max<1e-3`; sigma parity `1e-12` against the
**shipped** JS, with a negative control.

### 4a. Bug 14 — "green suite, dead in production", browser edition

The most dangerous of the three registration-class bugs, because it is invisible to **every**
server-side check: the suite was green, 25 nodes registered, and all five JS files parsed as clean
ES modules. Every widget would still have sat on a spinner forever.

All four widgets read their data from the `onExecuted(output)` argument. **That argument is the
node's UI payload, not its return sockets.**

- `execution.py:563` — the `executed` message is gated on UI output existing at all:
  `if len(output_ui) > 0: ... server.send_sync("executed", {..., "output": output_ui, ...})`
- `execution.py:377-381` — `output_ui` comes **only** from `NodeOutput.ui`
- `comfy_api/latest/_io.py:2285` — `NodeOutput(*args, ui=None, ...)`; `*args` are the **sockets**

`ui=` appeared in **zero of the 25 nodes**, so no `executed` message was ever emitted. Adding a
STRING *socket* named `boxes_json` put nothing in front of the browser.

**Fix:** real `ui=` payloads on all four nodes (canonical V3 pattern, `comfy_extras/nodes_train.py:1481`).
Sockets and `execute()` math untouched. Two compounding defects fixed alongside:
- the loading state had no exit on error/interrupt → permanent spinner. Now `bindExecutionLifecycle`
  covers `executing`, `execution_error`, `execution_interrupted`, and prompt-finish (`node === null`).
- `drawLoadingSpinner` derived its arc from `Date.now()` while `rafThrottle` scheduled exactly one
  frame — a *frozen* fake spinner. Now a real rAF loop that stops when the state leaves `loading`.

**Perf defect found in the fix:** `ui.PreviewImage` writes one PNG per **batch frame**. All three
image-emitting nodes were passing the full clip — ~97 temp files per run for a widget that draws one
backdrop. Now `[:1]`; the full trajectory comes from `mmx_boxes`, and the IMAGE sockets keep the full
batch.

| Node | UI keys | Widget |
|---|---|---|
| TrackCrop | `images` + `mmx_boxes` | W1 box trajectory |
| MaskPrep | `mmx_pixel_mask` + `mmx_token_preview` | W2 token grid |
| SigmaInspector | `mmx_sigma` | W3 sigma plot (live client-side recompute) |
| DriftQC | `images` (heatmap) + `mmx_drift` | W4 drift curve |

DriftQC's heatmap is a **native node preview only** — `PreviewImage` writes to `FolderType.temp`
(`_ui.py:395-400`) and `cleanup_temp()` clears it at startup *and* shutdown (`main.py:531`, `:614`),
so nothing accumulates. W4 deliberately ignores it and plots from `mmx_drift`; wiring the widget to
it would duplicate a preview ComfyUI already renders for free.

### 4b. Bug 15 — both shipped workflows were mis-wired

Found by a new test (`test_workflow_links_are_consistent`, `test_workflow_slots_match_live_schema`)
written *after* the SigmaInspector wiring, precisely because a hand-edited graph is easy to get wrong.
It failed immediately on code that had already shipped:

- **`h3_full_spine_pipeline.json` — the sampling chain could not run.** `LegalScheduler` takes
  `SIGMAS` (`legal_scheduler.py:33`) and its own description says *"Wire BasicScheduler sigmas → this
  → SamplerCustomAdvanced"*. The graph fed it a **MODEL** from `SigmaShiftLocked`, and **no
  `BasicScheduler` existed at all**. `SigmaShiftLocked` was also missing its required `model` input
  and its `report` output, `BasicGuider.model` was never connected, and both nodes carried stale
  `widgets_values` counts.
- **Duplicate edge** — links 2 and 7 both carried `TrackCrop.crops → MaskedReplace.crops`.
- **`h3_masked_face_pipeline.json`** — link 4 targeted **slot 3 of a 3-input node** (out of range);
  the correct slot is 2 (`video_latent_mask`).

**Fix:** inserted `BasicScheduler` between `SigmaShiftLocked` and `LegalScheduler`, corrected every
slot list to match the live schema, connected the guider, dropped the duplicate edge, fixed the
out-of-range slot. `SigmaInspector` is wired as a **read-only tap** on `LegalScheduler.sigmas` — it
cannot sit "between" the shift node and the scheduler, because it consumes `SIGMAS` and emits only
strings.

### 4c. Bug 16 — a parity test that could never fail

`web/w3_sigma_plot.js` reimplements `mmx_utils.sigma_schedule.time_shift_sigma` in JS so the plot can
recompute client-side without re-queueing. Two implementations of one formula drift. The first
parity test pasted a **copy** of the JS into the Python file — so it measured the copy, and editing
the shipped widget left it green.

Rewritten to extract `timeShiftSigma` out of the real widget file by brace balancing and run it under
node. **Negative control:** perturbing the shipped formula by 0.1% fails at `1e-12`; restoring it
passes. If the function is renamed the extractor **fails loudly** rather than skipping.

---

## 5. N11 DriftQC calibration — known bias, documented

Measured on this box (RAFT-small, `supersample=2`):

| frame | injected | reported | error |
|---|---|---|---|
| 16 px | 1.0 | 1.0114 | 0.011 |
| 32 px | 1.0 | 1.0019 | 0.002 |
| 48 px | 1.0 | 0.8670 | 0.133 |
| 192 px | 0.0 | 0.0231 | 0.023 |
| 192 px | 0.5 | 0.3770 | 0.123 |
| 192 px | 1.0 | 0.8209 | 0.179 |
| 192 px | 2.0 | 2.0207 | 0.021 |
| 256 px | 1.0 | 0.8213 | 0.179 |

**Known bias, not a bug:** RAFT-small under-reports ~1 px displacements by roughly 0.18 px at native
crop sizes; noise floor at zero shift is 0.02–0.04 px. **Keep `threshold_px` ≥ 0.25** so the gate sits
above noise + bias.

**Why `supersample` is explicit:** the calibration test originally passed *by accident*. At its 64 px
frame size `_prep_for_raft` upscales to the 128 px RAFT floor — a free 2× supersample. Real crops
(128–512 px) got none, where the numbers were 1.0 → 0.7285 (err **0.272**) and 2.0 → 1.6848
(err **0.315**) — outside the gate, while CI stayed green. Tests now pin 16/32/48/192/256 px.

**Exterior-only proven:** randomising the entire masked interior moves the number to 0.17 px.

---

## 6. Verified facts (cite, don't re-derive)

- Soft masks: `ceil(m*256)/256`, max-pool to the 2×2 DiT patch, **no 0.5 threshold**
  (`model_base.py:2224-2232`). Granularity = 32 source px → fine feather is **pixel-space only**.
- Per-row timesteps already in core (`model.py:593-608`) — a soft noise mask **is** the sigma curve.
- Audio = VAE over raw 32 kHz waveform, **not mel**.
- Union: blocks `[0,10,20,30,40]`, `control_in_dim 49`, `control_apply_audio false` — **no ComfyUI
  loader path exists** (D1).
- **Only `NodeOutput.ui` reaches the browser** (`execution.py:377-381`, `:563`). Socket values never
  appear in `onExecuted`. §4a.
- `addDOMWidget` renders on Vue and legacy frontends alike; `node.imgs` / canvas draws do not on Vue
  (`third_party/ComfyUI-OCIO/web/ocio_io.js:201-202`, MIT).
- **Never call `addWidget` after `addDOMWidget`** — `Comfy-Org/ComfyUI_frontend#7942` (open,
  untriaged, reported against 1.37.8). Schema widgets exist before JS runs, so appending the DOM
  widget last is the safe order.
- **ViTPose is reachable here**: `onnxruntime 1.27.0` + `vitpose_h_wholebody_model.onnx` on disk.
  Real detection: **133 keypoints** (COCO-WholeBody), mean confidence 0.5451, 77/133 above 0.3.
- **DVD backend unavailable** — `third_party/DVD` not cloned. **DepthCrafter weight-gated** — UNet not
  local (SVD base is). `normals_from_depth` needs no model. N9 defaults to `passthrough`.
- `comfy_kitchen` skew raises `AttributeError`, **not** `ImportError` → fallbacks use `except Exception`.
- Test python: `D:/PROJECT/ComfyUI_windows_portable/comfy_env/python.exe` (shell python has no torch).
- `node` v24.15.0 is on this box — usable for JS/Python parity tests.

---

## 7. Registration smoke test — production conditions

Anything less than this does not prove registration:

```python
import sys, os, asyncio, importlib.util
sys.path.insert(0, ".../third_party/ComfyUI")
import utils.json_util                      # poison sys.modules as ComfyUI does
mp = ".../ComfyUI-MiniMaxSuite"; n = mp.replace(".", "_x_")   # nodes.py:2250
spec = importlib.util.spec_from_file_location(n, os.path.join(mp, "__init__.py"))
mod = importlib.util.module_from_spec(spec)
sys.modules[n] = mod                        # nodes.py:2262 — BEFORE exec_module
spec.loader.exec_module(mod)
nl = asyncio.run((await mod.comfy_entrypoint()).get_node_list())
```

---

## 8. NOT done — open and blocked

| Item | Status |
|---|---|
| **Live browser verification of W1–W4** | **NOT DONE.** Everything in Stage 4 is source-level and headless. No widget has been seen to render. Three things source checks cannot prove, in order: (1) the four states (empty/loading/error/success) on one widget, (2) the #7942 DOM-vs-built-in layout gap on a resized node, (3) W3's live recompute on a slider drag without re-queue. Closure: `tools/playwright_widget_screenshot.py` on the GPU box. |
| **Runtime validation vs real H3 output** | **Blocked** — no checkpoints. Everything is code-reading or synthetic-tensor. |
| **Acceptance clips** | **Blocked** — need a real dancing + lateral-pan plate. Deliberately not synthesised. |
| **ControlNet-Union apply** | **Blocked by design (D1)** — no ComfyUI path exists. N3 ships hint extractors only. |
| Delegation C phases C2–C6 | C2 is 3 nodes (TurboLoRA / TurboSampler / AudioQualityGate); 6 of 9 already exist |
| Deferred widgets | mask painter · waveform / false-colour (#66) · viseme overlay (#37) |
| ViTPose in CI | Reachable on this box, but mocked in tests (needs onnx + weights) |
| Open questions to user | Director path (a/b/c) · scope confirmation · Linux-box weights · H3 licence jurisdiction (excludes US/EU) |
