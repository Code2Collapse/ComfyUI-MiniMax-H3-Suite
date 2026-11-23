# Changelog

## Unreleased

### Fixed
- **Zero offset on V2V** (owner order 2026-11-23; `docs/PLAN_h3_zero_offset.md`, bench `tools/h3_zero_offset_bench.py`):
  - Subject Uncrop / Plate Restore / Stitch Back: a decode that the H3 VAE cut to a multiple of 16 is put back at
    core's offset with the margins from the plate instead of being stretched into the box (5.02 px -> 0.0005 px).
    Subject Crop records the crop size in each box (`metadata.mmx_crop_size`) and says so in its report.
  - Lanczos resizes (Subject Crop / Uncrop, Source AV bridge, Motion Context, AV noise mask) run in float: core's went
    through 8-bit PIL (HDR clipped at 1.0, 256 levels; 129.5/255 -> 7/255 on an HDR plate).
  - Mask Prep and Mask To Latent Space reduce masks that are off the 16 px grid on the cells the VAE actually encodes
    (182 / 118 masked cells left "keep" -> 0). On-grid masks are unchanged.
  - Detail Reinject combines the generated low band with the plate's high band (the detail was counted twice: an
    unchanged region came back 8.8/255 different and 0.43 px moved -> 0.025/255, 0 px).
  - Source AV Latent, Masked Replace and the native audio lock pad the soundtrack to whole 1/40 s ticks before the
    audio VAE: core cut up to 12.5 ms off the head (8.3 ms for 121 frames).
  - The OpenCV fast paths of mask dilate / grow move a mask to the CPU before `.numpy()`.
  - Subject Crop: appended optional `bboxes` / `bboxes_first_frame` - crop at given boxes, one plan for a whole job
    (each window planned on its own moved the crop between segments).
  - Face Stitch no longer fades the face out for one missed detection (the old face ghosted through ~9 frames).
  - Temporal Subject Tracker keeps the soft mask edge until the plate-size matte (4K edge error 8.0 -> 2.6 px).
  - Matte Chatter's dropout bridge uses 1 byte per cell instead of 9 (bit-exact).
- Compact-node layout engine: a node removed right after it was created (paste then undo, a script that adds and
  removes) is no longer kept alive by the engine (about 185 KB each, measured in CNP). Same fix as CNP's engine (L9.01).

## 1.1.0 - 2026-11-17

### Added
- **HDR** (owner order A29; category `MiniMax H3/HDR`, 18 nodes; how-to `docs/HDR_HOWTO.md`, workflows
  `workflows/hdr/`):
  - `mmx_utils/hdr_math.py`: the single source of truth for every HDR encode / decode (ported from
    ComfyUI-H3HDR-Test; default `diffhdr_log_16384`), plus sRGB, PQ, gamut (Rec.709 / Rec.2020 / ACEScg), the
    editable VAE bias table, soft-shoulder preview, clip masks and the guided gain-map join.
  - Inference and I/O: HDR Log Encode / Decode, Bias Correct, Gamut Convert, Clip Mask, Gain Map Join, HDR Prompt,
    Tonemap Preview, Load / Save EXR Sequence (half or float, ZIP, never clamped), Save Log 16-bit (PNG / TIFF),
    Save HDR10 (HEVC 10-bit, PQ, BT.2020, mastering metadata), Eval Report.
  - Dataset and training: Fetch Poly Haven, Fetch Hugging Face (size cap), HDRI -> Clips, Make Pairs, Train IC-LoRA
    (ai-toolkit `minimax_h3_ref2va`, SDR as reference video, loss on the target only). They run as background jobs
    with progress, cancel and skip-what-is-done; a jobs panel shows the log and the loss curve.
  - Workflows: (a) SDR -> HDR EXR, (b) + Regenerate 4K gain-map join, (c) dataset build, (d) training, (e) eval.
- **Open-H3 Regenerate STD** (9 nodes) and **Open-H3 Regenerate 2K IC** (5 nodes), two distinct pipelines.
- `__version__`, `pyproject.toml`, this changelog.

### Changed
- H3 Latent Upscale (Model) accepts a plain H3 video latent (from VAE Encode).

### Tested / not tested
- Tested here on synthetic data (CPU): log round trip < 1e-4 relative from 2^-10 to 16384, EXR keeps values > 1 (cv2
  and OpenImageIO), bias table, gamut, PQ / HDR10 tags, gain-map join, job cancel / resume / skip, workflow wiring.
- Not tested here: the real H3 model, the HDR LoRA, a GPU, real downloads, a real training run; picture quality is
  not verified.

### Compatibility
- No existing node was renamed or removed; saved workflows load unchanged. 196 nodes registered.

## 1.0.0 - before 2026-11-16

The Suite as it stood before versioning (164 nodes), unversioned.
