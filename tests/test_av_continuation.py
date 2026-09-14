"""AV latent plumbing and continuation, ported from LongMedia.

H3's NestedTensor is INJECTED as a factory rather than imported, which is what
lets all of this be tested on CPU with a stub - no H3 build, no weights. That
design is also why these nodes register on a ComfyUI that has no H3 at all.

The most valuable test here is the cross-validation one: two independently
ported implementations of H3's frame grid must agree, or one of them is quietly
corrupting layouts.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

PACK = Path(__file__).resolve().parents[1]
if str(PACK) not in sys.path:
    sys.path.insert(0, str(PACK))

from mmx_utils import av_latent_ops as ops  # noqa: E402
from mmx_utils import context_windows as cw  # noqa: E402


class FakeNested:
    """Stands in for comfy.ldm.minimax NestedTensor.

    The only contract av_latent_ops relies on is `.is_nested` and `.unbind()`.
    """

    is_nested = True

    def __init__(self, streams):
        self._streams = tuple(streams)

    def unbind(self):
        return self._streams


def _video(frames: int, b: int = 1, h: int = 8, w: int = 8) -> torch.Tensor:
    return torch.rand(b, 24, ops.video_latent_t(frames), h, w)


def _audio(frames: int, b: int = 1) -> torch.Tensor:
    return torch.rand(b, 32, 2, ops.audio_latent_t(frames))


def _av(frames: int) -> dict:
    return ops.pack_av_latents({"samples": _video(frames)},
                               {"samples": _audio(frames)}, FakeNested)


# ── the grid: two ports must agree ──────────────────────────────────────────

def test_ported_grid_matches_the_context_window_planner():
    # INVARIANT: av_latent_ops (from LongMedia) and context_windows (written
    # here) implement H3's frame grid independently. If they disagree, one of
    # them is producing layouts that are silently wrong - and nothing else in
    # the suite would notice.
    for n in (5, 22, 39, 56, 81, 90, 124, 200, 409):
        assert ops.align_frame_count(n) == cw.snap_frame_count(n, "up"), n
    for fc in (5, 22, 39, 90, 124, 396):
        assert ops.video_latent_t(fc) == cw.latent_rows(fc), fc


def test_frame_count_round_trips_through_latent_rows():
    # INVARIANT: frames -> rows -> frames must be the identity on legal counts,
    # or a stitched segment comes out a different length than planned.
    for fc in (5, 22, 39, 90, 124, 396):
        assert ops.frame_count_from_video_t(ops.video_latent_t(fc)) == fc


def test_audio_clock_is_40hz_against_24fps_video():
    # INVARIANT: audio latents run on H3's 40 Hz clock while video is 24 fps -
    # the same ratio the temporal RoPE uses. Getting it wrong desyncs picture
    # and sound over a long clip, which is invisible for a few seconds.
    assert ops.FPS == 24 and ops.AUDIO_LATENT_FPS == 40
    assert ops.audio_latent_t(24 + 5) >= ops.audio_latent_t(5)
    frames, v_t, a_t = ops.temporal_shape(124)
    assert frames == ops.align_frame_count(124)
    assert v_t == ops.video_latent_t(frames)
    assert a_t == ops.audio_latent_t(frames)


# ── pack / split ────────────────────────────────────────────────────────────

def test_pack_then_split_round_trips_both_streams():
    # INVARIANT: the joint latent is how H3 samples picture and sound together;
    # losing either on the way through would be silent until decode.
    v, a = _video(90), _audio(90)
    av = ops.pack_av_latents({"samples": v}, {"samples": a}, FakeNested)
    video_latent, audio_latent = ops.split_av_latent(av)
    assert torch.equal(video_latent["samples"], v)
    assert torch.equal(audio_latent["samples"], a)


def test_split_does_not_leak_a_nested_mask_into_either_stream():
    # INVARIANT: a nested noise mask handed to a single stream would constrain
    # the sampler in a way the user never asked for.
    av = _av(90)
    video_latent, audio_latent = ops.split_av_latent(av)
    for side in (video_latent, audio_latent):
        mask = side.get("noise_mask")
        assert mask is None or not getattr(mask, "is_nested", False)


def test_unpack_rejects_a_plain_tensor():
    # INVARIANT: an ordinary LATENT wired into an AV socket must be named as
    # such, not indexed into and crash somewhere deeper.
    with pytest.raises(ValueError, match="NestedTensor"):
        ops.unpack_av_samples({"samples": torch.rand(1, 24, 7, 8, 8)})


def test_wrong_channel_counts_are_named():
    # INVARIANT: H3 video is 24-channel and audio is [B,32,2,T]. Saying which
    # is wrong saves guessing which wire is at fault.
    with pytest.raises(ValueError, match="24 channels"):
        ops.pack_av_latents({"samples": torch.rand(1, 16, 7, 8, 8)},
                            {"samples": _audio(90)}, FakeNested)
    with pytest.raises(ValueError, match=r"\[B, 32, 2, T\]"):
        ops.pack_av_latents({"samples": _video(90)},
                            {"samples": torch.rand(1, 8, 2, 40)}, FakeNested)


# ── continuation ────────────────────────────────────────────────────────────

def test_prepare_continuation_snaps_the_overlap_to_the_grid():
    # INVARIANT: the overlap is snapped DOWN to 17k+5, and the snapped value is
    # what Stitch Continuation must be given. Reporting the requested value
    # instead is how a seam lands in the wrong place.
    out, frames, actual = ops.prepare_continuation(_av(124), 124, 30, 0.0, 0.0,
                                                   FakeNested)
    assert cw.is_legal_frame_count(actual)
    assert actual <= 30
    assert frames == ops.align_frame_count(124)
    v, a = ops.unpack_av_samples(out)
    assert v.shape[2] == ops.video_latent_t(frames)
    assert a.shape[-1] == ops.audio_latent_t(frames)


def test_prepare_continuation_rejects_out_of_range_denoise():
    # INVARIANT: named refusal, not a silent clamp - a context denoise above 1
    # means the user misunderstood the control.
    with pytest.raises(ValueError, match="denoise"):
        ops.prepare_continuation(_av(124), 124, 22, 1.5, 0.0, FakeNested)


def test_stitch_length_is_previous_plus_next_minus_overlap():
    # INVARIANT: the arithmetic the whole chain rests on. If this is off by a
    # frame, every subsequent segment inherits the error.
    prev = _av(124)
    cont, frames, overlap = ops.prepare_continuation(prev, 124, 22, 0.0, 0.0,
                                                     FakeNested)
    stitched, total = ops.stitch_continuation(prev, cont, overlap, FakeNested)
    prev_v, _ = ops.unpack_av_samples(prev)
    prev_frames = ops.frame_count_from_video_t(prev_v.shape[2])
    assert int(total) == prev_frames + frames - overlap
    out_v, out_a = ops.unpack_av_samples(stitched)
    assert ops.frame_count_from_video_t(out_v.shape[2]) == int(total)
    # picture and sound must still describe the same duration
    assert int(out_a.shape[-1]) == ops.audio_latent_t(int(total))


def test_stitching_twice_keeps_growing_in_sync():
    # INVARIANT: long form is many passes, so the accumulator must stay correct
    # across repeated joins - a per-join drift only shows up minutes in.
    acc = _av(124)
    for _ in range(2):
        cont, _frames, overlap = ops.prepare_continuation(acc, 124, 22, 0.0, 0.0,
                                                          FakeNested)
        acc, total = ops.stitch_continuation(acc, cont, overlap, FakeNested)
        v, a = ops.unpack_av_samples(acc)
        assert ops.frame_count_from_video_t(v.shape[2]) == int(total)
        assert int(a.shape[-1]) == ops.audio_latent_t(int(total))


# ── the H3 guard ────────────────────────────────────────────────────────────

def test_missing_h3_gives_a_sentence_not_an_import_error():
    # INVARIANT: these nodes REGISTER on a ComfyUI without H3 and explain
    # themselves when run. A module-scope import would instead delete them from
    # the menu with no message - which already cost six nodes once.
    from mmx_nodes import continuation as cont_mod

    try:
        cont_mod._nested_factory()
    except RuntimeError as exc:
        msg = str(exc)
        assert "comfy.ldm.minimax" in msg
        assert "installed correctly" in msg
    except Exception as exc:  # pragma: no cover - only if H3 IS present
        pytest.fail(f"expected RuntimeError with a sentence, got {type(exc).__name__}")
