"""H3 Continuation Next / Commit over many segments, without a model (A16 Part IV).

The "sampler" keeps the masked rows exactly like ComfyUI's KSamplerX0Inpaint (out = new * mask + latent * (1 - mask));
the "decoder" writes each frame's GLOBAL index into its pixels and each audio sample's global index into the
waveform. So the committed output proves, frame by frame and sample by sample, that nothing is lost, duplicated or
shifted - over 30 segments - and the session machinery (crash, approval, rejection, checksum, geometry) is exercised.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest
import torch

pytest.importorskip("comfy.nested_tensor", reason="needs ComfyUI")
pytest.importorskip("cv2")
pytest.importorskip("soundfile")

import mmx_nodes.h3_continuation as hc  # noqa: E402
from mmx_utils import h3_session as hs  # noqa: E402

SR = 8000
L, T, TA = 124, 37, 207          # one segment: 124 frames, 37 latent rows, 207 audio ticks
H, W = 4, 6                      # latent size (64 x 96 px)


@pytest.fixture
def base(tmp_path, monkeypatch):
    monkeypatch.setattr(hc, "_base_dir", lambda: str(tmp_path))
    return tmp_path


def _template():
    import comfy.nested_tensor as nt
    return {"samples": nt.NestedTensor((torch.zeros(1, 24, T, H, W), torch.zeros(1, 32, 2, TA)))}


def _cond():
    return [[torch.zeros(1, 4, 8), {"minimax_keyframes": [{"resolved_frame_index": 0, "latent": None},
                                                         {"resolved_frame_index": 60, "latent": None}]}]]


def _sample(lat, seed):
    """Fake sampler honouring the noise mask exactly like core's inpaint wrapper."""
    import comfy.nested_tensor as nt
    v, a = lat["samples"].unbind()
    g = torch.Generator().manual_seed(seed)
    nv, na = torch.rand(v.shape, generator=g), torch.rand(a.shape, generator=g)
    if "noise_mask" in lat:
        mv, ma = lat["noise_mask"].unbind()
        nv = nv * mv + v * (1 - mv)
        na = na * ma + a * (1 - ma)
    return {"samples": nt.NestedTensor((nv, na))}


def _decode(info, audio_origin_ticks):
    """Frames whose pixels ARE their global index; audio whose samples ARE their global index."""
    start = info["start"]
    idx = torch.arange(start, start + L)
    img = torch.zeros(L, H * 16, W * 16, 3)
    img[..., 0] = ((idx % 256).float() / 255.0)[:, None, None]
    img[..., 1] = ((idx // 256 % 256).float() / 255.0)[:, None, None]
    origin = int(round(audio_origin_ticks / 40 * SR))
    n = int(round(TA / 40 * SR))                     # the model's sound is exactly TA ticks long from its origin
    wav = (torch.arange(origin, origin + n).float() % 1000.0) / 1000.0
    return img, {"waveform": wav[None, None].repeat(1, 2, 1), "sample_rate": SR}


def _segment(base, *, decision="keep waiting", approval="accept", overlap="39", seed=0, restart=False):
    out = hc.MiniMaxH3_ContinuationNext.execute("film", _cond(), _template(), overlap, decision, restart)
    lat, pos, seg_json, report = out.args[:4]
    info = json.loads(seg_json)
    sess = json.load(open(os.path.join(info["root"], "session.json")))
    seg = next(g for g in sess["segments"] if g["id"] == info["id"])
    img, aud = _decode(info, seg["audio_origin_ticks"])
    res = hc.MiniMaxH3_ContinuationCommit.execute(seg_json, _sample(lat, seed), img, aud, approval)
    return info, lat, pos, res


def _frames(root):
    import cv2
    names = sorted(os.listdir(os.path.join(root, "frames")))
    out = []
    for n in names:
        a = cv2.imread(os.path.join(root, "frames", n), cv2.IMREAD_UNCHANGED)[..., ::-1].astype(np.float64) / 65535.0
        out.append(int(round(a[0, 0, 0] * 255)) + 256 * int(round(a[0, 0, 1] * 255)))
    return names, out


def test_thirty_segments_no_frame_or_sample_lost_duplicated_or_shifted(base):
    for k in range(30):
        info, lat, pos, res = _segment(base, seed=k)
    root = info["root"]
    sess = json.load(open(os.path.join(root, "session.json")))
    expected = L + 29 * (L - 39)
    assert sess["accepted_frames"] == expected
    names, idx = _frames(root)
    assert len(names) == expected and idx == list(range(expected))          # every frame once, in order
    import soundfile as sf
    wavs = sorted(os.listdir(os.path.join(root, "audio")))
    audio = np.concatenate([sf.read(os.path.join(root, "audio", f))[0][:, 0] for f in wavs])
    assert audio.shape[0] == hs.samples_at(expected, SR)                    # exact total, no drift
    assert np.allclose(audio, (np.arange(audio.shape[0]) % 1000) / 1000.0, atol=1e-6)   # every sample in place
    offsets = [g.get("audio_offset_ms", 0.0) for g in sess["segments"]]
    bound = hs.phase_bound_ms(hs.audio_slack_seconds(L, TA))                # 16.7 ms for 124 frames / 207 ticks
    assert max(abs(o) for o in offsets) <= bound + 1e-3                     # bounded, never accumulating
    assert min(offsets) >= -hs.audio_slack_seconds(L, TA) * 1000 - 1e-3     # the sound always covers the last frame


@pytest.mark.parametrize("frames,ticks", [(124, 207), (56, 93), (39, 65), (22, 37), (73, 122), (243, 405)])
def test_phase_choice_keeps_every_segment_covered(frames, ticks):
    slack = hs.audio_slack_seconds(frames, ticks)
    prev = {"audio_origin_ticks": 0}
    for start in range(0, 5000, 7):
        prev["audio_origin_ticks"] = 0
        t = hs.choose_tail_tick(prev, start, slack)
        err = t / 40 - start / 24
        assert err >= -slack - 1e-9                                         # covers the segment's last frame
        assert abs(err) * 1000 <= hs.phase_bound_ms(slack) + 1e-6


def test_the_pinned_head_is_the_previous_segments_exact_tail(base):
    _segment(base, seed=1)
    out = hc.MiniMaxH3_ContinuationNext.execute("film", _cond(), _template(), "39", "keep waiting", False)
    lat = out.args[0]
    v, a = lat["samples"].unbind()
    mv, ma = lat["noise_mask"].unbind()
    R = hs.rows_for_frames(39)
    assert torch.all(mv[:, :, :R] == 0) and torch.all(mv[:, :, R:] == 1)
    assert torch.all(ma[..., :65] == 0) and torch.all(ma[..., 65:] == 1)
    assert float(v[:, :, :R].abs().sum()) > 0                                # real tokens, not the empty template
    kfs = out.args[1][0][1]["minimax_keyframes"]
    assert [k["resolved_frame_index"] for k in kfs] == [60]                 # the guide at frame 0 is inside the head


def test_a_crash_is_generated_again(base):
    _segment(base)
    hc.MiniMaxH3_ContinuationNext.execute("film", _cond(), _template(), "39", "keep waiting", False)   # never committed
    info, *_ = _segment(base)
    sess = json.load(open(os.path.join(info["root"], "session.json")))
    statuses = [g["status"] for g in sess["segments"]]
    assert statuses == ["accepted", "failed", "accepted"]
    assert sess["segments"][2]["retry_of"] == 1 and sess["segments"][2]["start"] == sess["segments"][1]["start"]


def test_approval_holds_then_accept_places_it(base):
    _segment(base)
    info, *_ = _segment(base, approval="always ask")
    root = info["root"]
    assert json.load(open(os.path.join(root, "session.json")))["accepted_frames"] == L
    with pytest.raises(hs.SessionError, match="awaiting approval"):
        hc.MiniMaxH3_ContinuationNext.execute("film", _cond(), _template(), "39", "keep waiting", False)
    _segment(base, decision="accept the pending segment")
    sess = json.load(open(os.path.join(root, "session.json")))
    assert sess["accepted_frames"] == L + 2 * (L - 39)
    _, idx = _frames(root)
    assert idx == list(range(L + 2 * (L - 39)))


def test_reject_generates_the_same_interval_again(base):
    _segment(base)
    held, *_ = _segment(base, approval="always ask")
    again, *_ = _segment(base, decision="reject the pending segment")
    sess = json.load(open(os.path.join(held["root"], "session.json")))
    assert again["start"] == held["start"]
    assert [g["status"] for g in sess["segments"]] == ["accepted", "rejected", "accepted"]


def test_a_damaged_tail_is_refused(base):
    info, *_ = _segment(base)
    tail = os.path.join(info["root"], "tails", "seg_0000.safetensors")
    with open(tail, "r+b") as fh:
        fh.seek(-8, os.SEEK_END)
        fh.write(b"\x00" * 8)
    with pytest.raises(hs.SessionError, match="checksum"):
        hc.MiniMaxH3_ContinuationNext.execute("film", _cond(), _template(), "39", "keep waiting", False)


def test_a_sampler_that_ignored_the_mask_is_caught(base):
    _segment(base)
    out = hc.MiniMaxH3_ContinuationNext.execute("film", _cond(), _template(), "39", "keep waiting", False)
    lat, _, seg_json, _ = out.args[:4]
    bad = dict(lat)
    bad.pop("noise_mask")
    info = json.loads(seg_json)
    img, aud = _decode(info, 0)
    with pytest.raises(ValueError, match="did not honour the noise mask"):
        hc.MiniMaxH3_ContinuationCommit.execute(seg_json, _sample(bad, 3), img, aud, "accept")
    sess = json.load(open(os.path.join(info["root"], "session.json")))
    assert sess["segments"][-1]["status"] == "failed"


def test_another_size_is_refused(base):
    import comfy.nested_tensor as nt
    _segment(base)
    other = {"samples": nt.NestedTensor((torch.zeros(1, 24, T, H + 2, W), torch.zeros(1, 32, 2, TA)))}
    with pytest.raises(hs.SessionError, match="cannot be resized"):
        hc.MiniMaxH3_ContinuationNext.execute("film", _cond(), other, "39", "keep waiting", False)


def test_seam_offsets_per_overlap():
    assert hs.seam_offset_ms(39) == pytest.approx(0.0)
    assert hs.seam_offset_ms(22) == pytest.approx(8.333, abs=1e-3)


def test_restart_archives_and_starts_from_zero(base):
    info, *_ = _segment(base)
    info2, *_ = _segment(base, restart=True)
    assert info2["start"] == 0 and info2["id"] == 0
    assert any(d.startswith("film_archived_") for d in os.listdir(base))


def test_drift_holds_a_segment_that_changes_colour(base):
    _segment(base)
    out = hc.MiniMaxH3_ContinuationNext.execute("film", _cond(), _template(), "39", "keep waiting", False)
    lat, _, seg_json, _ = out.args[:4]
    info = json.loads(seg_json)
    sess = json.load(open(os.path.join(info["root"], "session.json")))
    img, aud = _decode(info, sess["segments"][-1]["audio_origin_ticks"])
    img[..., 2] = 1.0                                                        # the new segment turns blue
    res = hc.MiniMaxH3_ContinuationCommit.execute(seg_json, _sample(lat, 5), img, aud, "accept unless drift")
    assert "awaiting approval" in res.args[2] and "colour moved" in res.args[2]
