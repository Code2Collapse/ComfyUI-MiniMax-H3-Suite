# Vendored code

| Folder | Canonical source | Licence | Kept equal by |
|---|---|---|---|
| `c2c_core/` | [ComfyUI-CustomNodePacks](https://github.com/Code2Collapse/ComfyUI-CustomNodePacks) `c2c_core/` (same author, Code2Collapse) | Apache-2.0 (`THIRD_PARTY_LICENSES/c2c_core-LICENSE.txt`); Apache-2.0 code may be included in this GPL-3.0 pack | `tools/sync_vendored.py` group `py-core` (workspace) + the pre-push gate's `vendored_sync` check |

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
- `c2c_core/timeline`: the frame ledger (contract `h3.frame_ledger` v1).
