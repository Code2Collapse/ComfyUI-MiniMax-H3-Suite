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


# -- Prompt Preview & Edit (C2C L8.21: the generated prompt stays editable) ----

def _editor():
    from mmx_nodes.promptor import MiniMaxH3_PromptEditor
    return MiniMaxH3_PromptEditor


def _ran(**kw):
    out = _editor().execute(**kw)
    return out.args[0] if hasattr(out, "args") else out[0]


def test_prompt_editor_wired_prompt_wins_by_default():
    # INVARIANT: with use_edited_text off (the default, and every saved workflow) a wired prompt wins over the box,
    # exactly as before the toggle existed.
    assert _ran(prompt="generated", _stored_prompt="old draft") == "generated"
    assert _ran(prompt="", _stored_prompt="typed") == "typed"


def test_prompt_editor_edited_text_wins_when_turned_on():
    # INVARIANT: use_edited_text on -> the box wins; an empty box never blanks the prompt (falls back to the wire).
    assert _ran(prompt="generated", _stored_prompt="my edit", use_edited_text=True) == "my edit"
    assert _ran(prompt="generated", _stored_prompt="  ", use_edited_text=True) == "generated"


def test_prompt_editor_fingerprint_follows_the_text_that_runs():
    # INVARIANT: the cache key changes exactly when the prompt that reaches H3 changes.
    E = _editor()
    a = E.fingerprint_inputs(prompt="generated", _stored_prompt="draft")
    assert a == E.fingerprint_inputs(prompt="generated", _stored_prompt="other draft")      # ignored text
    assert a != E.fingerprint_inputs(prompt="generated", _stored_prompt="draft", use_edited_text=True)


def test_prompt_editor_toggle_is_appended_and_optional():
    # INVARIANT: the toggle comes after the existing inputs and is optional, so saved widgets_values keep their order.
    schema = _editor().define_schema()
    ids = [i.id for i in schema.inputs]
    assert ids[-1] == "use_edited_text" and ids.index("_stored_prompt") < ids.index("use_edited_text")
    assert schema.inputs[-1].optional


# -- Prompt Composer (A15 Phase 1.1: `mode` used to do nothing - the port has no composer JS) ----------------------

def _composed(**kw):
    from mmx_nodes.promptor import MiniMaxH3_PromptComposer
    out = MiniMaxH3_PromptComposer.execute(**kw)
    return out.args[0] if hasattr(out, "args") else out[0]


def test_composer_mode_selects_the_scaffold_when_the_text_is_empty():
    from mmx_nodes.promptor import COMPOSER_MODES, COMPOSER_TEMPLATES
    assert set(COMPOSER_TEMPLATES) == set(COMPOSER_MODES)
    outs = {m: _composed(mode=m, _composer_prompt="") for m in COMPOSER_MODES}
    assert len(set(outs.values())) == len(COMPOSER_MODES), "every mode must give its own scaffold"
    assert "<Picture 2>" in outs["FL2VA (First & Last Frame)"] and "<Audio 1>" in outs["A2V (Audio to Video)"]
    for text in outs.values():
        assert "integrated_multimodal_description:" in text


def test_composer_typed_text_wins_over_the_scaffold():
    assert _composed(mode="A2V (Audio to Video)", _composer_prompt="  my prompt  ") == "my prompt"
