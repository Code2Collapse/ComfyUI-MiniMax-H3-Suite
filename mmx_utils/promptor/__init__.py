# PORTED FROM: 1038lab/ComfyUI-Minimax-H3-Promptor @ upstream
# Comfyui-Minimax-H3-Promptor — author 1038lab, GPL-3.0
"""MiniMax H3 promptor utilities (ported from Comfyui-Minimax-H3-Promptor)."""

from __future__ import annotations

from pathlib import Path

PROMPTOR_ROOT = Path(__file__).resolve().parent
TEMPLATES_DIR = PROMPTOR_ROOT / "templates"

__all__ = ["PROMPTOR_ROOT", "TEMPLATES_DIR", "snap_duration_frames"]


def snap_duration_frames(duration: float, fps: float = 24.0) -> int:
    """Snap duration to the H3 17n+5 frame grid (same logic as pipeline_engine)."""
    raw_frames = max(5, round(float(duration) * fps))
    rem = (raw_frames - 5) % 17
    return raw_frames if rem == 0 else raw_frames + (17 - rem)
