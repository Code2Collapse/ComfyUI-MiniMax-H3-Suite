# Vendored code

| Folder | Canonical source | Licence | Kept equal by |
|---|---|---|---|
| `c2c_core/` | [ComfyUI-CustomNodePacks](https://github.com/Code2Collapse/ComfyUI-CustomNodePacks) `c2c_core/` (same author, Code2Collapse) | Apache-2.0 (`THIRD_PARTY_LICENSES/c2c_core-LICENSE.txt`); Apache-2.0 code may be included in this GPL-3.0 pack | `tools/sync_vendored.py` group `py-core` (workspace) + the pre-push gate's `vendored_sync` check |
| `hdr_curves/_curves.py`, `hdr_curves/_primaries.py` (L9.179, HDR) | [ComfyUI-NukeMaxNodes](https://github.com/Code2Collapse/ComfyUI-NukeNodePack) `nukemax/nodes/ocio_color/` same names (Code2Collapse; `_curves.py` ported from ComfyUI-OCIO, Apache-2.0) | GPL-3.0 (same as this pack); the ComfyUI-OCIO part Apache-2.0, compatible | `tools/sync_vendored.py` groups `py-hdr-curves` / `py-hdr-primaries` (`hdr_curves/__init__.py` is this pack's own) |
| `wap_pose/models/onnx_models.py`, `wap_pose/pose_utils/pose2d_utils.py`, `wap_pose/pose_utils/human_visualization.py` | [ComfyUI-WanAnimatePreprocessV2](https://github.com/Code2Collapse/ComfyUI-WanAnimatePreprocessV2) same paths (Alibaba Wan team, kijai, Code2Collapse) | Apache-2.0 (`THIRD_PARTY_LICENSES/ComfyUI-WanAnimatePreprocessV2-LICENSE.txt`) | `tools/sync_vendored.py` groups `py-wap-onnx-models` / `py-wap-pose2d` / `py-wap-humanvis` (the `wap_pose` `__init__` files are this pack's own) |

Why: owner directive A20 (2026-10-31). The MiniMax H3 Suite must finish its main workflows without CustomNodePacks or
NukeMax installed, and shared engine code must not become two drifting implementations.

Rules:
- Never edit `c2c_core/` here. Fix the canonical tree in CustomNodePacks, then run `tools/sync_vendored.py --write`.
- When both packs are installed, the two copies load under different package names. Process-wide caches go through
  `c2c_core.shared_state()`, and objects that cross packs (such as the `C2C_VIDEO` handle) are checked by their
  fields or contract name, never with `isinstance`.

Contents (2026-10-31):
- `c2c_core/video`: the lazy video handle, the probe / index table with exact variable-frame-rate timestamps, the
  window reader, the memory budget, the audio descriptor, source metadata and test media.
- `c2c_core/video` (L9.89, owner A21 Rev 3 Part 3): the media entry point CustomNodePacks' unified Load Video (C2C)
  uses. It has:
  - `sequence.py`: frame numbers, padding classes, steps, missing frames, Nuke frame ranges, timelines;
  - `exr.py`: EXR parts, layers, channels and header;
  - `media.py`: `probe_media` for a movie, a sequence or one still, and `apply_timeline`;
  - `passes.py`: the lazy `C2C_EXR_PASSES` bundle.
  The reader gained black frames, named EXR layers, a bounded frame cache, and random access for reversed selections.
- `c2c_core/timeline`: the frame ledger (contract `h3.frame_ledger` v1.1: `source.numbers` holds an image sequence's
  own frame numbers, and -1 marks a black frame).
