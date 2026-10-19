"""H3 Continuation slice 2 (L8.45): long plates, mask videos, control videos and soundtracks cut per segment.

Real files, no model: lossless PNG-in-MOV plates whose frames carry their index (one of them with its first frame 2
frames late - core's time trim alone returns frame 84 for 85 there), PCM sound whose samples carry their index. The fake
H3 VAE writes the index of the FIRST frame of each latent row group (H3's 1,4,4,4,4) into the latent, the fake audio VAE
the sample at each 1/40 s tick - so every segment's source latent is checked frame by frame and tick by tick.
"""

from __future__ import annotations

import fractions
import json
import os
from types import SimpleNamespace

import numpy as np
import pytest
import torch

av = pytest.importorskip("av")
pytest.importorskip("comfy.nested_tensor", reason="needs ComfyUI")
pytest.importorskip("comfy.ldm.minimax.model", reason="needs ComfyUI with MiniMax H3")
pytest.importorskip("cv2")
pytest.importorskip("soundfile")

import mmx_nodes.h3_continuation as hc  # noqa: E402
from mmx_utils import h3_session as hs  # noqa: E402
from mmx_utils import h3_video_slice as vs  # noqa: E402

SR = 8000
L, T, TA = 124, 37, 207
W, H = 96, 64


def write_plate(path, n, *, offset=0, fps=24, w=W, h=H, sound=True, mask_box=None):
    c = av.open(str(path), "w")
    v = c.add_stream("png", rate=fps)
    v.width, v.height, v.pix_fmt = w, h, "rgb24"
    a = None
    if sound:
        a = c.add_stream("pcm_s16le", rate=SR)
        a.layout = "stereo"
    for k in range(n):
        img = np.zeros((h, w, 3), np.uint8)
        if mask_box is None:
            img[..., 0] = k % 256
            img[..., 1] = k // 256
        elif mask_box(k):
            img[8:40, 16:64, :] = 255
        f = av.VideoFrame.from_ndarray(img, format="rgb24")
        f.pts, f.time_base = k + offset, fractions.Fraction(1, fps)
        for p in v.encode(f):
            c.mux(p)
    if a is not None:
        total, s0 = int(round(n / fps * SR)), int(round(offset / fps * SR))
        idx = np.arange(total)
        for b in range(0, total, 800):
            seg = (idx[b:b + 800] % 32000).astype(np.int16)
            af = av.AudioFrame.from_ndarray(np.stack([seg, seg], 1).reshape(1, -1), format="s16", layout="stereo")
            af.sample_rate, af.pts, af.time_base = SR, s0 + b, fractions.Fraction(1, SR)
            for p in a.encode(af):
                c.mux(p)
        for p in a.encode():
            c.mux(p)
    for p in v.encode():
        c.mux(p)
    c.close()


def load(path):
    from comfy_api.latest._input_impl.video_types import VideoFromFile
    return VideoFromFile(str(path))


def idx_of(images):
    return ((images[:, 0, 0, 0] * 255).round() + 256 * (images[:, 0, 0, 1] * 255).round()).long().tolist()


def group_starts(t_rows):
    """First frame of each latent row: H3 rows cover (1,4,4,4,4) frames, repeating every 5 rows (17 frames)."""
    out, f = [], 0
    for r in range(t_rows):
        out.append(f)
        f += (1, 4, 4, 4, 4)[r % 5]
    return out


class CodingVAE:
    """H3 video VAE stand-in: latent row r, channel 0/1 = R/G of the first frame of row r's group (1,4,4,4,4)."""

    def __init__(self):
        self.first_stage_model = SimpleNamespace(vae_ratio_t=4, tokens_chunk_size=5, frame_pre_padding=3,
                                                 token_drop=3)
        self.downscale_ratio = (lambda a: max(1, (a - 5) // 17 * 5 + 2) if a > 1 else 1, 16, 16)
        self.upscale_ratio = (lambda a: max(1, (a - 2) // 5 * 17 + 5), 16, 16)

    def spacial_compression_encode(self):
        return 16

    def encode(self, images):
        n, h, w = images.shape[:3]
        t = self.downscale_ratio[0](n)
        out = torch.zeros(1, 24, t, h // 16, w // 16)
        for r, f in enumerate(group_starts(t)):
            out[0, 0, r] = images[f, 0, 0, 0]
            out[0, 1, r] = images[f, 0, 0, 1]
        return out


class CodingAudioVAE:
    """H3 audio VAE stand-in at 8 kHz: tick t = the sample at t / 40 s (one value per 200 samples)."""
    audio_sample_rate = SR

    def encode(self, wav):                       # [B, N, C]
        x = wav[0, ::SR // 40, 0]
        out = torch.zeros(1, 32, 2, int(x.shape[0]))
        out[0, 0, 0] = x
        return out


def sample_index(tick_value):
    return int(round(float(tick_value) * 32768))


@pytest.fixture
def base(tmp_path, monkeypatch):
    monkeypatch.setattr(hc, "_base_dir", lambda: str(tmp_path / "out"))
    return tmp_path


def _template():
    import comfy.nested_tensor as nt
    return {"samples": nt.NestedTensor((torch.zeros(1, 24, T, H // 16, W // 16), torch.zeros(1, 32, 2, TA)))}


def _cond():
    return [[torch.zeros(1, 4, 8), {}]]


def _sample(lat, seed):
    import comfy.nested_tensor as nt
    v, a = lat["samples"].unbind()
    g = torch.Generator().manual_seed(seed)
    nv, na = torch.rand(v.shape, generator=g), torch.rand(a.shape, generator=g)
    if "noise_mask" not in lat:
        return {"samples": nt.NestedTensor((nv, na))}
    mv, ma = lat["noise_mask"].unbind()
    return {"samples": nt.NestedTensor((nv * mv + v * (1 - mv), na * ma + a * (1 - ma)))}


def _decode(info):
    idx = torch.arange(info["start"], info["end"])
    img = torch.zeros(L, H, W, 3)
    img[..., 0] = ((idx % 256).float() / 255.0)[:, None, None]
    img[..., 1] = ((idx // 256).float() / 255.0)[:, None, None]
    return img


def _next(**kw):
    args = dict(session="film", positive=_cond(), latent=_template(), overlap_frames="39", decision="keep waiting",
                restart=False)
    args.update(kw)
    return hc.MiniMaxH3_ContinuationNext.execute(**args)


def _commit(out, seed=0):
    lat, _, seg_json = out.args[:3]
    info = json.loads(seg_json)
    return hc.MiniMaxH3_ContinuationCommit.execute(seg_json, _sample(lat, seed), _decode(info), None, "accept"), info


# ── the slicer ───────────────────────────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("offset", [0, 2])
def test_slices_are_frame_exact_even_when_the_first_frame_is_late(tmp_path, offset):
    p = tmp_path / f"plate{offset}.mov"
    write_plate(p, 300, offset=offset)
    vid = load(p)
    for start in (0, 85, 170, 255):
        images, got, aud = vs.read_frames(vid, start, L, with_audio=True)
        want = min(L, 300 - start)
        assert got == want and images.shape[0] == L
        assert idx_of(images[:got]) == list(range(start, start + got))
        assert idx_of(images[got:]) == [299] * (L - got)                     # padding = last frame repeated
        first = sample_index(aud["waveform"][0, 0, 0])
        assert (first - int(round(start / 24 * SR))) % 32000 in (0, 1, 31999)  # the window's sound, within a sample


def test_other_frame_rates_are_refused_and_2398_is_conformed(tmp_path):
    write_plate(tmp_path / "p30.mov", 40, fps=30, sound=False)
    with pytest.raises(ValueError, match="30.000 fps"):
        vs.check_rate(load(tmp_path / "p30.mov"), "source_video")


def test_cut_audio_pads_silence_outside():
    wav = torch.arange(100.0)[None, None]
    out = vs.cut_audio({"waveform": wav, "sample_rate": 10}, -0.5, 2.0)["waveform"][0, 0]
    assert out[:5].tolist() == [0.0] * 5 and out[5:].tolist() == list(range(15))


# ── masked long-form editing of a plate ────────────────────────────────────────────────────────────────────────────

def test_a_plate_is_edited_segment_by_segment_to_its_last_frame(base):
    write_plate(base / "plate.mov", 300)
    write_plate(base / "mask.mov", 300, sound=False, mask_box=lambda k: 100 <= k < 200)
    plate, mask = load(base / "plate.mov"), load(base / "mask.mov")
    kw = dict(video_vae=CodingVAE(), source_video=plate, source_mask=mask)
    starts = []
    for k in range(4):
        out = _next(**kw)
        lat = out.args[0]
        info = json.loads(out.args[2])
        starts.append(info["start"])
        v, _ = lat["samples"].unbind()
        mv, _ = lat["noise_mask"].unbind()
        R = 0 if k == 0 else hs.rows_for_frames(39)
        # rows after the pinned head are the PLATE's frames of this segment, frame for frame
        expect = [min(info["start"] + f, 299) for f in group_starts(T)][R:]
        got = (v[0, 0, R:, 0, 0] * 255).round().long() + 256 * (v[0, 1, R:, 0, 0] * 255).round().long()
        assert got.tolist() == expect
        assert torch.all(mv[:, :, :R] == 0)                                   # head pinned
        # the mask: regenerate only the rows whose frames are masked (frames 100-199), inside the box
        for r in range(R, T):
            frames = range(info["start"] + group_starts(T)[r],
                           info["start"] + (group_starts(T)[r + 1] if r + 1 < T else L))
            masked = any(100 <= f < 200 and f < 300 for f in frames)
            assert bool(mv[0, 0, r].max() > 0) == masked, (k, r)
        src, pm = out.args[4], out.args[5]
        assert idx_of(src[: min(L, 300 - info["start"])]) == list(range(info["start"], min(info["start"] + L, 300)))
        assert pm.shape == (L, H, W)
        res, _ = _commit(out, seed=k)
    assert starts == [0, 85, 170, 255]
    sess = json.load(open(os.path.join(info["root"], "session.json")))
    assert sess["accepted_frames"] == 300 and sess["complete"] is True
    assert sess["segments"][-1]["source_end"] == 300
    assert len(os.listdir(os.path.join(info["root"], "frames"))) == 300
    with pytest.raises(hs.SessionError, match="complete"):
        _next(**kw)


def test_a_plate_ending_exactly_on_a_segment_completes_without_an_empty_segment(base):
    write_plate(base / "plate.mov", 209, sound=False)
    kw = dict(video_vae=CodingVAE(), source_video=load(base / "plate.mov"))
    for k in range(2):
        _commit(_next(**kw), seed=k)
    with pytest.raises(hs.SessionError, match="every frame is done"):
        _next(**kw)
    root = os.path.join(str(base / "out"), "film")
    sess = json.load(open(os.path.join(root, "session.json")))
    assert sess["accepted_frames"] == 209 and [g["status"] for g in sess["segments"]] == ["accepted"] * 2
    with pytest.raises(hs.SessionError, match="complete"):
        _next(**kw)


def test_the_plate_size_must_match_the_latent(base):
    write_plate(base / "big.mov", 130, w=128, sound=False)
    with pytest.raises(ValueError, match="Set the empty latent's width / height"):
        _next(video_vae=CodingVAE(), source_video=load(base / "big.mov"))


def test_plain_refusals():
    with pytest.raises(ValueError, match="video_vae"):
        hc._check_sources(hc.SOUNDS[0], None, None, object(), None, None)
    with pytest.raises(ValueError, match="needs the soundtrack"):
        hc._check_sources(hc.SOUNDS[2], None, object(), None, None, None)
    with pytest.raises(ValueError, match="needs source_mask"):
        hc._check_sources(hc.SOUNDS[3], object(), object(), object(), None, None)
    with pytest.raises(ValueError, match="audio VAE"):
        hc._check_sources(hc.SOUNDS[1], object(), None, object(), None, None)


# ── sound-driven long video ────────────────────────────────────────────────────────────────────────────────────────

def test_a_soundtrack_is_held_fixed_on_global_time_and_ends_the_session(base):
    n = int(10.0 * SR)                                                        # 10 s = 240 frames
    track = {"waveform": ((torch.arange(n) % 32000).float() / 32768.0)[None, None].repeat(1, 2, 1), "sample_rate": SR}
    kw = dict(audio_vae=CodingAudioVAE(), soundtrack=track, sound=hc.SOUNDS[2])
    for k in range(3):
        out = _next(**kw)
        lat, info = out.args[0], json.loads(out.args[2])
        sess = json.load(open(os.path.join(info["root"], "session.json")))
        seg = next(g for g in sess["segments"] if g["id"] == info["id"])
        _, a = lat["samples"].unbind()
        mv, ma = lat["noise_mask"].unbind()
        assert torch.all(ma == 0)                                             # every tick held
        A = seg.get("pinned_ticks", 0)
        origin = int(round(seg["audio_origin_ticks"] / 40 * SR))
        for t in range(A, TA):                                                # fresh ticks: the soundtrack at global time
            want = origin + t * (SR // 40)
            assert sample_index(a[0, 0, 0, t]) == (want % 32000 if want < n else 0), (k, t)
        assert torch.all(mv[:, :, (hs.rows_for_frames(39) if k else 0):] == 1)
        _commit(out, seed=k)
    sess = json.load(open(os.path.join(info["root"], "session.json")))
    assert sess["accepted_frames"] == 240 and sess["complete"] is True       # 124, 209, then capped at 240
    with pytest.raises(hs.SessionError, match="complete"):
        _next(**kw)


def test_the_source_videos_own_sound_is_kept_from_the_tick_origin(base):
    write_plate(base / "plate.mov", 300)
    kw = dict(video_vae=CodingVAE(), audio_vae=CodingAudioVAE(), source_video=load(base / "plate.mov"),
              sound=hc.SOUNDS[1])
    for k in range(3):
        out = _next(**kw)
        lat, info = out.args[0], json.loads(out.args[2])
        sess = json.load(open(os.path.join(info["root"], "session.json")))
        seg = next(g for g in sess["segments"] if g["id"] == info["id"])
        _, a = lat["samples"].unbind()
        _, ma = lat["noise_mask"].unbind()
        assert torch.all(ma == 0)
        origin = int(round(seg["audio_origin_ticks"] / 40 * SR))
        for t in range(seg.get("pinned_ticks", 0), TA):
            want = origin + t * (SR // 40)
            if want < int(round(300 / 24 * SR)):
                assert (sample_index(a[0, 0, 0, t]) - want) % 32000 in (0, 1, 31999), (k, t)   # within a sample
        _commit(out, seed=k)


def test_control_video_slice_and_unconnected_outputs(base):
    write_plate(base / "pose.mov", 300, sound=False)
    out = _next(control_video=load(base / "pose.mov"))
    _commit(out)
    out = _next(control_video=load(base / "pose.mov"))
    assert idx_of(out.args[6]) == list(range(85, 209))
    assert out.args[4] is None and out.args[5] is None                       # no plate: no stand-in frames


def test_regenerate_where_masked_frees_only_the_ticks_of_masked_frames(base):
    write_plate(base / "plate.mov", 300)
    write_plate(base / "mask.mov", 300, sound=False, mask_box=lambda k: 100 <= k < 200)
    out = _next(video_vae=CodingVAE(), audio_vae=CodingAudioVAE(), source_video=load(base / "plate.mov"),
                source_mask=load(base / "mask.mov"), sound=hc.SOUNDS[3])
    _, ma = out.args[0]["noise_mask"].unbind()
    free = [t for t in range(TA) if float(ma[0, 0, 0, t]) > 0]
    # frames 100..123 of this first segment are masked: ticks from 100/24 s on (tick 166) are regenerated
    assert free == list(range(166, TA))
