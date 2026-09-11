"""CPU-only tests for Comfyui-Minimax-H3-Promptor port (mmx_utils/promptor)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PACK = Path(__file__).resolve().parents[1]
if str(PACK) not in sys.path:
    sys.path.insert(0, str(PACK))

from mmx_utils.promptor import snap_duration_frames
from mmx_utils.promptor.prompt_builder import PromptBuilder
from mmx_utils.promptor.task_detector import TASK_DESCRIPTIONS, TaskDetector
from mmx_utils.h3_grid import calculate_h3_frames, is_h3_compatible


def test_task_detector_auto_t2v():
    # INVARIANT: no media inputs -> T2V
    assert TaskDetector.detect() == "T2V"


def test_task_detector_respects_explicit_override():
    # INVARIANT: UI description maps back to short code
    assert TaskDetector.detect(user_override=TASK_DESCRIPTIONS["FL2VA"]) == "FL2VA"


def test_task_detector_auto_fl2va_two_images():
    # INVARIANT: exactly two images -> FL2VA
    assert TaskDetector.detect(image_count=2) == "FL2VA"


def test_task_detector_auto_ref2va_omni():
    # INVARIANT: image + video -> Ref2VA
    assert TaskDetector.detect(image_count=1, has_video=True) == "Ref2VA"


def test_prompt_builder_loads_system_template():
    # INVARIANT: system_base.txt is discoverable under mmx_utils/promptor/templates
    pb = PromptBuilder()
    sys_prompt = pb.build_system_prompt("T2V", duration=5.0, output_language="English")
    assert "MiniMax H3" in sys_prompt
    assert "Video Duration: 5" in sys_prompt


def test_prompt_builder_fl2va_alignment_instruction():
    # INVARIANT: FL2VA alignment references both picture endpoints
    pb = PromptBuilder()
    alignment = pb.generate_alignment_instruction("FL2VA", duration=5.0, image_count=2)
    assert "<Picture 1>" in alignment
    assert "<Picture 2>" in alignment
    assert "0.00-second" in alignment


def test_snap_duration_frames_matches_h3_grid():
    # INVARIANT: 5.0s @ 24fps -> 124 frames on 17n+5 grid
    assert snap_duration_frames(5.0) == 124
    assert is_h3_compatible(snap_duration_frames(5.0))
    assert snap_duration_frames(5.0) == calculate_h3_frames(round(5.0 * 24))


@pytest.mark.parametrize("duration", [4.0, 5.0, 10.0, 15.0])
def test_snap_duration_frames_param(duration):
    # INVARIANT: duration snap always lands on 17n+5 and matches suite grid helper
    frames = snap_duration_frames(duration)
    assert frames == calculate_h3_frames(max(5, round(duration * 24)))
    assert is_h3_compatible(frames)
