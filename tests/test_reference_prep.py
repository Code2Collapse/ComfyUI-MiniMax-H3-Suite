"""Keyframe + reference conditioning arithmetic.

The node itself needs a VAE, a CLIP and a GPU. What is testable here is the
part that goes wrong quietly:

  * a reference video off H3's 17k+5 grid is reinterpreted, not rejected, and
    its motion comes out at the wrong speed
  * a soundtrack matched by POSITION instead of by number puts one person's
    voice on another
  * encoder and DiT reference lists that drift renders the wrong subject with
    total confidence

None of those raise on their own. Each one is pinned below.

CPU-only, no torch, no comfy.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PACK = Path(__file__).resolve().parents[1]
if str(PACK) not in sys.path:
    sys.path.insert(0, str(PACK))

from mmx_utils.reference_prep import (  # noqa: E402
    CANVAS_MULTIPLE,
    MIN_REF_VIDEO_FRAMES,
    REF_IMAGE_SHORT_EDGE,
    ReferencePrepError,
    canvas_size,
    check_reference_order,
    describe,
    reference_scale,
    sample_indices,
    sample_timestamps,
    snap_to_latent_grid,
    soundtrack_key,
)


# ── the latent grid ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("n", [5, 22, 39, 124, 362, 1000, 2001])
def test_every_snapped_count_lands_on_the_grid(n):
    assert snap_to_latent_grid(n) % 17 == 5


def test_snapping_rounds_down_never_up():
    """Rounding up pads a reference video with frames that do not exist,
    which invents motion at its tail."""
    for n in range(MIN_REF_VIDEO_FRAMES, 200):
        out = snap_to_latent_grid(n)
        assert out <= n, f"{n} snapped UP to {out}"
        assert n - out < 17, f"{n} lost {n - out} frames - more than one chunk"


def test_a_count_already_on_the_grid_is_untouched():
    for n in (5, 22, 39, 124):
        assert snap_to_latent_grid(n) == n


def test_a_clip_too_short_to_carry_motion_is_named_with_the_alternative():
    with pytest.raises(ReferencePrepError, match="reference IMAGE"):
        snap_to_latent_grid(4)


# ── reference sizing ────────────────────────────────────────────────────────

def test_match_brings_a_reference_to_the_generation_pixel_area():
    scale = reference_scale(1344, 768, 2688, 1536, "match")
    assert scale == pytest.approx(0.5, abs=1e-6)


def test_max_uses_the_short_edge():
    assert reference_scale(1344, 768, 4096, 3000, "max") == pytest.approx(
        REF_IMAGE_SHORT_EDGE / 3000.0)


@pytest.mark.parametrize("mode", ["match", "max"])
def test_a_reference_is_never_upscaled(mode):
    """Upscaling adds tokens without adding information - and reference tokens
    cost on every sampling step, so it is pure loss."""
    assert reference_scale(1344, 768, 64, 64, mode) == 1.0


def test_an_unknown_size_mode_names_both_options_and_the_cost():
    with pytest.raises(ReferencePrepError, match="slower"):
        reference_scale(1344, 768, 512, 512, "huge")


def test_a_degenerate_reference_is_named():
    with pytest.raises(ReferencePrepError, match="zero dimension"):
        reference_scale(1344, 768, 0, 512, "match")


# ── canvas snapping ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("w,h,scale", [
    (2688, 1536, 0.5), (1000, 700, 1.0), (33, 17, 1.0), (4096, 3000, 0.68),
])
def test_canvas_sizes_land_on_the_grid(w, h, scale):
    cw, ch = canvas_size(w, h, scale)
    assert cw % CANVAS_MULTIPLE == 0 and ch % CANVAS_MULTIPLE == 0


def test_a_tiny_reference_still_gets_one_whole_cell():
    """Rounding to 0 would produce an empty latent rather than an error."""
    assert canvas_size(3, 3, 0.1) == (CANVAS_MULTIPLE, CANVAS_MULTIPLE)


# ── soundtrack pairing ──────────────────────────────────────────────────────

def test_a_soundtrack_is_paired_by_number_not_by_position():
    """Autogrow inputs are a dict. Filling video 1 and 3 and leaving 2 empty
    is normal; matching by position would put video 3's audio on video 1 -
    one person's voice on another, with nothing raised."""
    assert soundtrack_key("ref_video_1") == "ref_video_audio_1"
    assert soundtrack_key("ref_video_3") == "ref_video_audio_3"

    videos = {"ref_video_1": "A", "ref_video_3": "C"}
    audios = {"ref_video_audio_1": "a1", "ref_video_audio_3": "a3"}
    paired = {v: audios.get(soundtrack_key(k)) for k, v in videos.items()}
    assert paired == {"A": "a1", "C": "a3"}


def test_a_video_with_no_soundtrack_pairs_with_nothing():
    assert {}.get(soundtrack_key("ref_video_2")) is None


# ── encoder sampling ────────────────────────────────────────────────────────

def test_the_encoder_sees_two_frames_a_second():
    idx = sample_indices(48)               # two seconds at 24 fps
    assert idx == list(range(0, 48, 12))
    assert len(idx) == 4


def test_timestamps_line_up_with_the_sampled_frames():
    idx = sample_indices(48)
    ts = sample_timestamps(idx)
    assert len(ts) == len(idx)
    assert ts == [0.0, 0.5, 1.0, 1.5]


def test_an_empty_clip_samples_nothing_rather_than_raising():
    assert sample_indices(0) == []
    assert sample_timestamps([]) == []


# ── the ordering guard ──────────────────────────────────────────────────────

def test_matching_lists_pass():
    items = [{"type": "image"}, {"type": "video"}]
    blocks = [{"kind": "image"}, {"kind": "video"}]
    check_reference_order(items, blocks)


def test_a_video_with_audio_contributes_two_items_and_one_block():
    """The real shape of the data, which is why the guard compares by kind
    rather than by length."""
    items = [{"type": "audio"}, {"type": "video"}]
    blocks = [{"kind": "video_audio"}]
    check_reference_order(items, blocks)


def test_a_dropped_encoder_image_is_caught():
    """THE failure: the model gets three references, the encoder is told about
    two, and the prompt's <Picture 3> silently means something else."""
    items = [{"type": "image"}, {"type": "image"}]
    blocks = [{"kind": "image"}, {"kind": "image"}, {"kind": "image"}]
    with pytest.raises(ReferencePrepError, match="wrong subject"):
        check_reference_order(items, blocks)


def test_a_video_count_mismatch_is_caught():
    items = [{"type": "video"}, {"type": "video"}]
    blocks = [{"kind": "video"}]
    with pytest.raises(ReferencePrepError, match="<Video n>"):
        check_reference_order(items, blocks)


def test_keyframe_images_appended_later_do_not_trip_the_guard():
    """Keyframes are added to the encoder list after the references, on
    purpose - so <Picture 2> keeps meaning the same reference whether or not a
    keyframe is wired. That makes encoder images legitimately outnumber
    reference blocks."""
    items = [{"type": "image"}, {"type": "image"}, {"type": "image"}]
    blocks = [{"kind": "image"}]
    check_reference_order(items, blocks)


# ── the report ──────────────────────────────────────────────────────────────

def test_the_report_says_when_neither_half_of_the_node_is_used():
    text = describe(0, 0, 0, 0, "match", 124, 124)
    assert "plain text-to-video" in text


def test_the_report_warns_about_the_cost_of_max():
    text = describe(0, 2, 0, 0, "max", 124, 124)
    assert "EVERY sampling step" in text
    assert "slower" in text


def test_the_report_says_when_the_length_moved():
    text = describe(1, 0, 0, 0, "match", 130, 124)
    assert "130 -> 124" in text
    assert "wrong speed" in text


def test_the_report_is_quiet_when_the_length_did_not_move():
    assert "snapped" not in describe(1, 0, 0, 0, "match", 124, 124)


def test_the_report_warns_where_keyframes_and_reference_video_fight():
    text = describe(2, 0, 1, 0, "match", 124, 124)
    assert "handover" in text


def test_the_report_distinguishes_keyframes_from_references():
    text = describe(2, 3, 0, 0, "match", 124, 124)
    assert "real frame positions" in text
    assert "NO frame position" in text
