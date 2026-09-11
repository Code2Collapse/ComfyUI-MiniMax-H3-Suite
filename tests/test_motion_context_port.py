"""CPU-only tests for ComfyUI-H3-Motion-Context port (layout_contract + pure helpers)."""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch

PACK = Path(__file__).resolve().parents[1]
if str(PACK) not in sys.path:
    sys.path.insert(0, str(PACK))

for _comfy in (
    PACK.parent.parent / "ComfyUI_windows_portable" / "ComfyUI",
    PACK.parent / "third_party" / "ComfyUI",
):
    if (_comfy / "comfy").is_dir():
        _cp = str(_comfy)
        if _cp not in sys.path:
            sys.path.insert(0, _cp)
        break
else:
    pytest.skip("ComfyUI tree not found for motion context port tests", allow_module_level=True)

# comfy_kitchen on this box is version-skewed: importing comfy.ldm.minimax
# raises AttributeError, NOT ImportError, so a plain `except ImportError`
# does not catch it. Left unguarded this aborts COLLECTION of the whole
# suite, not just this file.
try:
    from mmx_utils.motion_context import (  # noqa: E402
        CORR_CREDIBLE,
        FRAME_PER_TOKEN,
        continuation_report,
        mono,
        norm_xcorr,
        pixel_frames,
        run_seam_probe,
        step_offsets,
        steps_for_frames,
        trim_clip,
    )
except (ImportError, AttributeError) as _exc:  # noqa: BLE001
    pytest.skip(
        "motion context helpers need the H3 modules, which do not import on "
        "this box (comfy_kitchen skew): " + str(_exc),
        allow_module_level=True,
    )

FRAME_RESCALE = 5.0 / 3.0


def test_pixel_frames_matches_token_cycle():
    # INVARIANT: 7 latent steps cover 1+4+4+4+4+1+4 = 22 pixel frames.
    assert pixel_frames(7) == 22
    assert pixel_frames(2) == 5
    assert pixel_frames(1) == 1


def test_step_offsets_starts_at_zero():
    # INVARIANT: first latent step always begins at pixel frame 0.
    assert step_offsets(7) == [0, 1, 5, 9, 13, 17, 18]


def test_steps_for_frames_legal_windows():
    # INVARIANT: only whole-step windows are reachable on the 1,4,4,4,4 grid.
    assert steps_for_frames(5) == 2
    assert steps_for_frames(22) == 7
    assert steps_for_frames(39) == 12
    assert steps_for_frames(56) == 17
    assert steps_for_frames(4) is None


def test_trim_clip_matches_picture_and_sound():
    # INVARIANT: trim removes the same duration from images and audio.
    images = torch.rand(30, 64, 64, 3)
    sr = 32000
    audio = {"waveform": torch.randn(1, 2, sr * 2), "sample_rate": sr}
    out_images, out_audio, report = trim_clip(images, 5, audio=audio, fps=24.0)
    assert out_images.shape[0] == 25
    assert "trimmed 5 leading frames" in report
    want_samples = int(round(25 / 24.0 * sr))
    assert abs(int(out_audio["waveform"].shape[-1]) - want_samples) <= 1


def test_norm_xcorr_peaks_at_the_true_offset():
    # INVARIANT: the PEAK LOCATION is what the seam probe reads - it turns the
    # argmax into a lag in milliseconds. A magnitude threshold says nothing
    # about that: a periodic signal correlates ~0.98 at many offsets, so the
    # old '> 0.99' check passed or failed on luck of the window. Pin WHERE the
    # peak lands instead.
    rng = np.random.default_rng(7)
    ref = rng.standard_normal(400)          # noise -> one unambiguous match
    offset = 137
    win = ref[offset:offset + 50]
    ncc = norm_xcorr(win, ref)
    assert ncc is not None
    assert int(np.argmax(ncc)) == offset
    assert float(ncc.max()) > 0.99          # an exact copy of noise does hit ~1


def test_norm_xcorr_rejects_an_unrelated_window():
    # INVARIANT: the probe must be able to say 'these do not match'. Without
    # this, the argmax test above would still pass if everything scored 1.0.
    rng = np.random.default_rng(11)
    ref = rng.standard_normal(400)
    win = rng.standard_normal(50)           # independent noise
    ncc = norm_xcorr(win, ref)
    assert ncc is not None
    assert float(ncc.max()) < 0.75



def test_seam_probe_reports_missing_context():
    # INVARIANT: probe passes audio through and explains missing inputs.
    sr = 32000
    span = int(round(22 / 24.0 * sr))
    b = np.random.default_rng(0).standard_normal(span * 2)
    audio = {"waveform": np.stack([b, b])[None, ...], "sample_rate": sr}
    passthrough, report = run_seam_probe(audio, trim_frames=22)
    assert passthrough["sample_rate"] == sr
    assert "no clip_a_latent wired" in report


# --- layout_contract mock harness (from upstream _mock_harness.py) ---


def _frame_grid(h, w):
    n_h, n_w = h // 2, w // 2
    hh, ww = np.meshgrid(np.arange(n_h, dtype=np.float64),
                         np.arange(n_w, dtype=np.float64), indexing="ij")
    return np.stack([hh.reshape(-1), ww.reshape(-1)], axis=-1)


def _video_t_spans(latent_t):
    return [FRAME_RESCALE * FRAME_PER_TOKEN[k % 5] for k in range(latent_t)]


def _audio_grid(cursor, t, rows_per_step=2):
    g = np.zeros((t * rows_per_step, 3), dtype=np.float64)
    g[:, 0] = np.tile(cursor + np.arange(t, dtype=np.float64), rows_per_step)
    if rows_per_step == 2:
        g[:t, 2] = -1.0
        g[t:, 2] = 1.0
    return g


def _video_t_grid(n, origin):
    spans = np.array(_video_t_spans(n), dtype=np.float64)
    head = np.cumsum(spans[:-1]) if n > 1 else np.zeros(0, dtype=np.float64)
    return float(origin) + np.concatenate(([0.0], head))


def _video_grid(vt, frame, cursor):
    g = np.empty((vt, frame.shape[0], 3), dtype=np.float64)
    g[:, :, 0] = _video_t_grid(vt, cursor)[:, None]
    g[:, :, 1:] = frame[None]
    return g.reshape(-1, 3)


def _make_mm(audio_rows_per_step=2, int_index=False, audio_anchor="start",
             compensate_refs=True, legacy=False):
    mm = types.ModuleType("comfy.ldm.minimax.model")
    mm.FRAME_RESCALE = FRAME_RESCALE
    mm.FRAME_PER_TOKEN = FRAME_PER_TOKEN
    mm._video_t_spans = _video_t_spans

    def _ref_t_span(blk):
        kind = blk["kind"]
        if kind == "image":
            return 1.0
        if kind == "audio":
            return float(blk["ref_audio_t"])
        if kind in ("video", "video_audio"):
            return max(float(blk["ref_audio_t"]),
                       sum(_video_t_spans(blk["latent_t"])))
        raise ValueError("mock: unsupported ref kind %r" % kind)

    def _build(self, text_len, latent_t, latent_h, latent_w, audio_t,
               keyframes, refs, frame_count):
        frame = _frame_grid(latent_h, latent_w)
        frame_rows = frame.shape[0]
        segs, blocks = [], []

        def emit(kind, g):
            segs.append((kind, g.shape[0]))
            blocks.append(g)

        g = np.zeros((text_len, 3), dtype=np.float64)
        g[:, 0] = np.arange(text_len, dtype=np.float64)
        emit("text", g)

        origin = float(text_len)
        if compensate_refs:
            for blk in (refs or []):
                origin += _ref_t_span(blk)

        for kf in (keyframes or []):
            p = kf["resolved_frame_index"]
            if legacy:
                if p == 0:
                    cond_t = float(text_len)
                elif frame_count is not None and p == frame_count - 1:
                    cond_t = (float(text_len) + sum(_video_t_spans(latent_t))
                              - FRAME_RESCALE)
                else:
                    raise ValueError("only first/last keyframe anchors are supported")
                g = np.empty((frame_rows, 3), dtype=np.float64)
                g[:, 0] = cond_t
                g[:, 1:] = frame
                emit("cond", g)
                continue
            if int_index:
                p = int(p)
            cond_t = origin + FRAME_RESCALE * p
            video_latent = kf.get("latent")
            if video_latent is not None:
                emit("cond", _video_grid(video_latent.shape[2], frame, cond_t))
            audio_latent = kf.get("audio_latent")
            if audio_latent is not None:
                rt = audio_latent.shape[-1]
                start = cond_t if audio_anchor == "start" else cond_t - rt
                emit("cond_audio", _audio_grid(start, rt, audio_rows_per_step))

        cursor = float(text_len)
        for blk in (refs or []):
            kind = blk["kind"]
            if kind == "image":
                r_frame = _frame_grid(blk["latent_h"], blk["latent_w"])
                g = np.empty((r_frame.shape[0], 3), dtype=np.float64)
                g[:, 0] = cursor
                g[:, 1:] = r_frame
                emit("ref_img", g)
                cursor += 1.0
            elif kind == "audio":
                rt = int(blk["ref_audio_t"])
                if rt > 0:
                    emit("ref_audio", _audio_grid(cursor, rt, audio_rows_per_step))
                cursor += float(rt)
            elif kind in ("video", "video_audio"):
                rt = int(blk["ref_audio_t"])
                vt = int(blk["latent_t"])
                r_frame = _frame_grid(blk["latent_h"], blk["latent_w"])
                if rt > 0:
                    emit("ref_audio", _audio_grid(cursor, rt, audio_rows_per_step))
                emit("ref_img", _video_grid(vt, r_frame, cursor))
                cursor += max(float(rt), sum(_video_t_spans(vt)))

        emit("audio", _audio_grid(cursor, audio_t))
        emit("video", _video_grid(latent_t, frame, cursor))

        seg_abs, off = [], 0
        for kind, n in segs:
            seg_abs.append((off, off + n, kind))
            off += n
        self.segments = seg_abs
        self.seq_len = off
        self.position_ids = np.concatenate(blocks)

    if legacy:
        class PackedLayout:
            def __init__(self, text_len, latent_t, latent_h, latent_w,
                         audio_t, keyframes=None, refs=None, frame_count=None):
                _build(self, text_len, latent_t, latent_h, latent_w, audio_t,
                       keyframes, refs, frame_count)
    else:
        class PackedLayout:
            def __init__(self, text_len, latent_t, latent_h, latent_w,
                         audio_t, keyframes=None, refs=None):
                _build(self, text_len, latent_t, latent_h, latent_w, audio_t,
                       keyframes, refs, None)

    mm.PackedLayout = PackedLayout
    return mm


def _load_layout_contract(mm):
    for name in ("comfy", "comfy.ldm", "comfy.ldm.minimax"):
        sys.modules.setdefault(name, types.ModuleType(name))
    sys.modules["comfy.ldm.minimax.model"] = mm
    sys.modules["comfy"].ldm = sys.modules["comfy.ldm"]
    sys.modules["comfy.ldm"].minimax = sys.modules["comfy.ldm.minimax"]
    sys.modules["comfy.ldm.minimax"].model = mm
    sys.modules.pop("mmx_utils.motion_layout_contract", None)
    return importlib.import_module("mmx_utils.motion_layout_contract")


def test_layout_contract_passes_faithful_layout():
    # INVARIANT: contract accepts ComfyUI 0.33+ layout semantics.
    lc = _load_layout_contract(_make_mm())
    lc.ensure()
    assert lc.is_checked()
    lc.ensure()


def test_layout_contract_refuses_int_cast_on_anchor():
    # INVARIANT: integer cast on fractional audio anchor is caught.
    lc = _load_layout_contract(_make_mm(int_index=True))
    with pytest.raises(RuntimeError, match="fractional or negative"):
        lc.ensure()
    assert not lc.is_checked()


def test_layout_contract_refuses_legacy_constructor():
    # INVARIANT: older frame_count constructor is refused before build.
    lc = _load_layout_contract(_make_mm(legacy=True))
    with pytest.raises(RuntimeError, match="older H3 layout"):
        lc.ensure()


def test_continuation_report_separates_correlation_from_placement():
    # INVARIANT: the report exists to distinguish 'audio matches but sits at
    # the wrong offset' from 'audio does not match'. A 110 Hz tone is PERIODIC,
    # so a seam can correlate at ~1.0 while being shifted a whole cycle, and the
    # report must SAY so rather than call it clean. The previous assertion
    # demanded 'clean continuation' from a fixture that carries a real 18 ms
    # offset - it asserted the opposite of correct behaviour.
    import re as _re
    sr = 32000
    span = int(round(22 / 24.0 * sr))
    tone = np.sin(2 * np.pi * 110.0 * np.arange(span * 3) / sr)
    a = tone[:span * 2]
    b = np.concatenate([a[-span:], tone[span * 2:]])
    report = "\n".join(continuation_report(a, b, sr, span, 50.0, 40.0))

    assert 'mean corr' in report
    corr = float(_re.search(r'mean corr\s+([0-9.]+)', report).group(1))
    assert corr > 0.9, report
    assert 'reading:' in report
    assert ('Placement' in report) or ('offset' in report), report
    assert CORR_CREDIBLE < 1.0

