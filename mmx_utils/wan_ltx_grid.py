# MIT License — ComfyUI-AV-Handles
# Copyright (c) 2025 Adam Pizurny
# PORTED FROM: ComfyUI-AV-Handles :: utils/wan_utils.py @ HEAD
"""WAN / LTX2 frame-grid helpers (padding modes for N7)."""

from __future__ import annotations

import math


def calculate_wan_frames(target_frames: int) -> int:
    """Round UP to next WAN-compatible frame count (4n+1)."""
    if target_frames <= 1:
        return 1
    n = math.ceil((target_frames - 1) / 4)
    return max(1, 4 * n + 1)


def calculate_next_wan_frames(current_frames: int) -> int:
    if current_frames < 1:
        return 1
    if is_wan_compatible(current_frames):
        return current_frames
    n = math.ceil((current_frames - 1) / 4)
    return 4 * n + 1


def is_wan_compatible(frames: int) -> bool:
    if frames < 1:
        return False
    return (frames - 1) % 4 == 0


def is_ltx2_compatible(frames: int) -> bool:
    if frames < 1:
        return False
    return (frames - 1) % 8 == 0


def calculate_ltx2_frames(target_frames: int) -> int:
    if target_frames <= 1:
        return 1
    n = math.ceil((target_frames - 1) / 8)
    return max(1, 8 * n + 1)


def calculate_next_ltx2_frames(current_frames: int) -> int:
    if current_frames < 1:
        return 1
    if is_ltx2_compatible(current_frames):
        return current_frames
    n = math.ceil((current_frames - 1) / 8)
    return 8 * n + 1
