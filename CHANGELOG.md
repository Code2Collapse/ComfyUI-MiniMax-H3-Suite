# Changelog

## Unreleased

### Added
- **Viggle Remake on H3** (category `MiniMax H3/Replace`; design `docs/PLAN_h3_replacement_quality.md`): H3 Remake
  Conditioning (Viggle), H3 Remake Keyframe Until, H3 Remake Noise, H3 Remake Prefix - ported from the reference
  nodes (Apache-2.0, CREDITS.md) - and **H3 Remake Replace**, the adapter from an approved anchor + a plate window to
  Remake conditioning in the window's own framing (zero offset), chunked by the Job Controller (anchor pinned on the
  first window, the accepted context held clean on later ones), with character, face / head and item / prop modes.
  Experimental: not run on H3 weights or the Remake LoRA.
- Workflows `workflows/replace/remake_character_replace.json`, `remake_head_face_replace.json`,
  `remake_item_replace.json` (job-chunked; Plate Restore puts the literal plate back outside the mask).
- `tools/h3_replace_bench.py`: re-entry identity drop and multi-shot identity spread (`--shots`, manifest
  `"cases"`), with the measure named (ArcFace when installed, otherwise masked colour histogram).
- H3 Image + Reference to Video: `image_batches` (appended, default `first image only` = unchanged) - `every image`
  makes each image of a reference batch its own `<Picture N>`; the report now says when images were left out.

### Fixed
- H3 Subject Memory: the approved anchor (H3 Anchor Approval) is now in its `anchors` output, as Approval promised,
  without routing Approval's bank (R2).
- H3 Identity Fusion Controller: a pixel-frame mask is reduced with H3's own frame groups, so a frame where the head
  enters or leaves is no longer dropped (R6).
- H3 Temporal Subject Tracker: refuses a subject picked on a frame other than 0 instead of putting that mask on frame
  0 and tracking the wrong spot (T1).
- H3 Face Track + Crop: with an identity anchor, a lone face that is not the performer (an extra while the performer
  is out of frame) and a first-frame face below `identity_threshold` are no longer taken; a connected
  `identity_reference` that cannot run (insightface missing) is an error instead of silently ignored; the report
  says whether identity was on (T2, T3).
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
