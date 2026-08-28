# ComfyUI-MiniMaxSuite — H3 geometric spine + sampling guards

MIT/Apache node pack for MiniMax H3 production workflows (V3 `io.ComfyNode` API).

## Nodes

| Node | Role |
|------|------|
| **H3 Frame Handles** (N7) | Editorial head/tail handles + H3/WAN/LTX2 grid padding; audio resync |
| **H3 Track + Crop** (N1) | TV/L1 crop planner; `crops` + `H3_TRANSFORM` |
| **H3 Control Hints** (N3) | Crop-space canny / passthrough hints (no Union consumer yet) |
| **H3 Mask Prep** (N2) | Video-side latent mask |
| **H3 Masked Replace** (N4) | Inject crops + `noise_mask` NestedTensor (lipsync via audio lock) |
| **H3 Stitch Back** (N5) | Inverse affine paste; exterior = literal plate |
| **H3 Detail Reinject** (N6) | Plate texture transfer inside mask |
| **H3 Protected Layer Guard** (P10) | Fail if protected DiT blocks / FinalLayer paths are patched |
| **H3 Accelerator Conflict** (P11) | Fail on incompatible cache stacks; passes MODEL through |
| **H3 Legal Scheduler** (P3) | Validate sigma schedule; rejects SplitSigmas |
| **H3 Sigma Shift (Ratio Locked)** (P1) | ModelSamplingAV with video/audio ratio lock (default 4 = 12/3) |
| **H3 Per-Frame Denoise** (P6) | Face-size-driven per-frame noise_mask strength |
| **H3 Block Cache (T8)** (P9) | Dual-metric F1B0 block cache; `cache_device` default `cpu` |
| **H3 Sigma Inspector** (P2) | Read-only σ table report (no JS plot in this slice) |
| **H3 OCIO Bridge** (I1) | Scene-linear ↔ display via OCIO (HDR-safe, no clamp) |
| **H3 Color Round-Trip QC** (W4) | Display round-trip error probe with HDR gate |
| **H3 Strongest Pose** (N8) | ViTPose-H plate detect → crop warp pose hints |
| **H3 Temporal Depth** (N9) | Depth hints (passthrough default) + optional normals |
| **H3 Drift QC** (N11) | Exterior-only RAFT drift gate |
| **H3 Region Mask** (N10) | Depth-horizon or painted region MASK for N2 spine |
| **H3 Family Presets** (N12) | Six-section prompt (#14) + style DNA (#16) |
| **H3 Edit Validator** (N13) | Pre-flight gate (#8) — mask_required, grid, drift |
| **H3 DCC Bridge** (N14) | External EXR sim ingest (no solver) |
| **H3 Pose Puppeteer** (N15) | Driving pose → motion hints (Union blocked) |
| **H3 HDR Roundtrip** (N16) | OCIO view↔scene with HDR tag propagation |

N1 is a **planner**, not a detector — wire `subject_mask` and/or `bboxes_json`.

## Dual-clock shim (P4 — not shipped)

On ComfyUI core `180060c` and later, `ModelSamplingAV` is native (`comfy/model_sampling.py`). A dual-clock sampler shim is a no-op — use core **MiniMaxH3SigmaShift** or our **MiniMaxH3_SigmaShiftLocked** for ratio-locked shifts.

## Workflow

See `workflows/h3_masked_face_pipeline.json` and `workflows/README.md` (runtime untested without H3 weights).

Recommended sampling chain after N4:

`N4` → `P10 ProtectedLayerGuard` → `P11 AcceleratorConflict` → `P1 SigmaShiftLocked` → `P3 LegalScheduler` → sampler

## Tests

```bash
cd ComfyUI-MiniMaxSuite
python run_tests.py
```

One test file per `mmx_nodes/` and `mmx_utils/` module. CPU-only; no weights required.

## Licence

Ported utilities carry upstream licence headers and `# PORTED FROM:` provenance lines.
