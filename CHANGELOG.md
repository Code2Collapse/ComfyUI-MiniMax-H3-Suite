# Changelog

## Unreleased

### Added
- **H3 Prompt Writer** node (`MiniMaxH3_H3PromptWriter`, category `MiniMax H3/Prompt`; design
  `docs/PLAN_h3_prompt_writer_node.md`): the H3 Prompt Writer extension's prompt logic (MIT, ported) as a workflow node
  with image / video / image-sequence / audio sockets, eight modes, and any provider - offline template, local Hugging
  Face transformers (any repo id), local GGUF (child process), Ollama, OpenAI-compatible, OpenAI, Anthropic, Gemini.
  Official-format audit plus one repair pass. Prompt, negative and report outputs. Example workflow
  `workflows/h3_prompt_writer_reference.json`.
- Prompt Writer v2: its description points to the new node (no input change; saved workflows load unchanged).

### Fixed
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
