"""Seam tone compensation. CPU-only, no weights.

The failure this node exists to prevent is a brightness STEP at a chained-segment
seam, so the tests are about whether the step is actually removed and whether a
wrong overlap is visible rather than silent.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

PACK = Path(__file__).resolve().parents[1]
if str(PACK) not in sys.path:
    sys.path.insert(0, str(PACK))

from mmx_utils.tone_compensate import (  # noqa: E402
    ToneCompensateError,
    compensate,
    compensate_report,
)


def _flat(frames: int, value: float, c: int = 3) -> torch.Tensor:
    return torch.full((frames, 8, 8, c), float(value))


def test_frame_shift_removes_a_uniform_darkening():
    # INVARIANT: the headline case. A segment generated 0.08 darker than its
    # anchor must come back to the anchor's level, or the seam pulses.
    out, stats = compensate(_flat(60, 0.50), _flat(90, 0.42), "frame_shift", 48)
    assert float(out.mean()) == pytest.approx(0.50, abs=1e-5)
    assert stats["drift_mean"] == pytest.approx(-0.08, abs=1e-5)


def test_correction_extends_past_the_overlap():
    # INVARIANT: only the first `overlap` frames can be fitted, but the WHOLE
    # segment drifted. Correcting just the overlap would move the step later
    # instead of removing it.
    out, _ = compensate(_flat(60, 0.50), _flat(90, 0.42), "frame_shift", 48)
    assert float(out[80:].mean()) == pytest.approx(0.50, abs=1e-5)


def test_gain_bias_recovers_an_affine_drift():
    # INVARIANT: gain_bias exists for a lift/compression, not just an offset.
    src = torch.rand(40, 8, 8, 3).clamp(0.05, 0.95)
    tgt = (src * 0.8 + 0.05).clamp(0, 1)
    out, _ = compensate(src, tgt, "gain_bias", 40)
    assert torch.allclose(out, src, atol=2e-2)


def test_lut_recovers_a_monotone_nonlinear_drift():
    # INVARIANT: the reason lut mode exists - a curve gain_bias cannot express.
    src = torch.rand(40, 16, 16, 3).clamp(0.02, 0.98)
    tgt = src.pow(1.6)
    out, _ = compensate(src, tgt, "lut", 40, lut_bins=64)
    assert float((out - src).abs().mean()) < 0.05


def test_no_drift_reports_that_the_seam_was_not_tonal():
    # INVARIANT: saying "essentially no drift" stops someone tuning this node
    # for an hour when the real problem is motion continuity.
    out, stats = compensate(_flat(30, 0.5), _flat(30, 0.5), "frame_shift", 20)
    assert stats["drift_abs_max"] < 1e-6
    assert "not a tone problem" in compensate_report(stats)


def test_overlap_longer_than_the_footage_is_reported_not_swallowed():
    # INVARIANT: overlap must equal the KEYFRAME count used at generation. If it
    # is silently clamped the fit runs over the wrong frames and the node
    # appears to do nothing - the single most likely way to misuse this.
    _out, stats = compensate(_flat(10, 0.5), _flat(10, 0.4), "frame_shift", 48)
    assert stats["overlap_clamped"] is True
    assert stats["overlap_used"] == 10
    r = compensate_report(stats)
    assert "clamped" in r and "48" in r


def test_per_frame_drift_is_reported_for_every_overlap_frame():
    # INVARIANT: the UI plots this. One row per fitted frame, one value per
    # channel - a shape change here silently empties the widget.
    _out, stats = compensate(_flat(30, 0.5), _flat(30, 0.4), "frame_shift", 12)
    assert len(stats["drift_per_frame"]) == 12
    assert all(len(row) == 3 for row in stats["drift_per_frame"])


def test_output_never_leaves_zero_one():
    # INVARIANT: the result is an IMAGE. An out-of-range value propagates as
    # clipping or NaN much further down the graph.
    out, _ = compensate(_flat(20, 0.02), _flat(20, 0.97), "frame_shift", 10)
    assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0


def test_bad_inputs_are_named():
    # INVARIANT: every refusal says which input is wrong.
    with pytest.raises(ToneCompensateError, match="overlap"):
        compensate(_flat(10, 0.5), _flat(10, 0.5), "frame_shift", 0)
    with pytest.raises(ToneCompensateError, match="channels"):
        compensate(_flat(10, 0.5, c=3), _flat(10, 0.5, c=1), "frame_shift", 5)
    with pytest.raises(ToneCompensateError, match="Unknown mode"):
        compensate(_flat(10, 0.5), _flat(10, 0.5), "nope", 5)
    with pytest.raises(ToneCompensateError, match="IMAGE batches"):
        compensate(torch.rand(8, 8, 3), _flat(10, 0.5), "frame_shift", 5)
