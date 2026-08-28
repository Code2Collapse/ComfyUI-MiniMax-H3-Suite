# MIT License — nuke-nodes-comfyui (EXR I/O pattern)
# PORTED FROM: nuke-nodes-comfyui :: io_nodes.py (OIIO read/write) @ HEAD

"""DCC sim frame ingest — real EXR only, no solver (R21)."""

from __future__ import annotations

import glob
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

SIM_TYPES = ("fire", "smoke", "fluid", "cloth", "rigid_body", "crowd")
INTERACTION_MODES = ("fully_preserved", "attribute_transfer", "reference")


class DCCSequenceError(RuntimeError):
    """Human-facing error when EXR sequence cannot be loaded."""


def _require_oiio():
    try:
        import OpenImageIO as oiio
    except ImportError as exc:
        raise DCCSequenceError(
            "OpenImageIO is not installed — cannot read EXR sim frames. "
            "Install OpenImageIO bindings or render to another format externally."
        ) from exc
    return oiio


def resolve_exr_paths(exr_sequence: str) -> list[str]:
    raw = (exr_sequence or "").strip()
    if not raw:
        raise DCCSequenceError(
            "exr_sequence is empty — point this input at a folder or glob of rendered EXR "
            "frames. This node ingests external sim output only; it does not run a solver."
        )
    if any(ch in raw for ch in "*?[]"):
        paths = sorted(glob.glob(raw))
    elif os.path.isdir(raw):
        paths = sorted(
            str(p) for p in Path(raw).iterdir() if p.suffix.lower() in (".exr", ".sxr")
        )
    elif os.path.isfile(raw):
        paths = [raw]
    else:
        parent = Path(raw).parent
        pattern = Path(raw).name
        if parent.is_dir():
            paths = sorted(glob.glob(str(parent / pattern)))
        else:
            paths = []
    paths = [p for p in paths if os.path.isfile(p)]
    if not paths:
        raise DCCSequenceError(
            f"No EXR frames matched {exr_sequence!r} — render the sim externally and wire "
            "the sequence path here. This node does not synthesise or solve simulations."
        )
    return paths


def read_exr_plane(path: str) -> np.ndarray:
    """Read one EXR subimage 0 to float32 HxWx3 — values above 1.0 preserved (no clamp).

  Limitation: only subimage 0 is read; channel-name routing and multi-subimage
  selection are not implemented. EXR with more than three channels must be
  pre-packed as RGB or exported as separate passes.
    """
    oiio = _require_oiio()
    inp = oiio.ImageInput.open(path)
    if not inp:
        raise DCCSequenceError(f"OpenImageIO could not open {path!r}.")
    try:
        spec = inp.spec()
        w, h, channels = spec.width, spec.height, spec.nchannels
        if channels > 3:
            raise DCCSequenceError(
                f"{path!r} has {channels} channels — this bridge reads subimage 0 as flat RGB only. "
                "Pre-pack to 3 channels or export a separate RGB pass; channel-name routing is not supported."
            )
        buf = inp.read_image(0, 0, 0, channels, oiio.FLOAT)
        if buf is None:
            raise DCCSequenceError(f"OpenImageIO read failed for {path!r}.")
        arr = np.array(buf, dtype=np.float32).reshape(h, w, channels)
        if channels == 1:
            arr = np.repeat(arr, 3, axis=-1)
        return arr
    finally:
        inp.close()


def load_exr_sequence(exr_sequence: str) -> torch.Tensor:
    """Load EXR sequence to IMAGE tensor [T,H,W,3] float32 HDR."""
    paths = resolve_exr_paths(exr_sequence)
    frames = [read_exr_plane(p) for p in paths]
    stacked = np.stack(frames, axis=0)
    return torch.from_numpy(stacked)


def build_retention_payload(
    *,
    sim_type: str,
    interaction_mode: str,
    video_index: int = 1,
    fps: float = 24.0,
) -> dict[str, Any]:
    sim = (sim_type or "fluid").lower()
    mode = (interaction_mode or "fully_preserved").lower()
    if sim not in SIM_TYPES:
        sim = "fluid"
    if mode not in INTERACTION_MODES:
        mode = "fully_preserved"
    label = f"<Video {int(video_index)}>"
    lines = [
        f"{label}: {sim} sim reference",
        f"camera: {mode}",
        f"environment: attribute_transfer ({sim})",
        f"timing: {float(fps):g} fps external sim",
    ]
    return {
        "video_label": label,
        "sim_type": sim,
        "interaction_mode": mode,
        "fps": float(fps),
        "retention_analysis_lines": lines,
    }


def retention_json_text(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True)
