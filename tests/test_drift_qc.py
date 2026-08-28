import pytest
import torch

torchvision = pytest.importorskip("torchvision")

from mmx_utils.drift_qc import inject_pixel_shift, measure_exterior_drift


def _synthetic_pair(h=64, w=64, dx=0.0, dy=0.0):
    torch.manual_seed(0)
    original = torch.rand(1, h, w, 3)
    edited = inject_pixel_shift(original, dx, dy)
    mask = torch.zeros(h, w)
    mask[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4] = 1.0
    return original, edited, mask


@pytest.mark.parametrize("shift", [0.0, 0.5, 1.0, 2.0])
def test_drift_calibration_within_tolerance(shift):
    original, edited, mask = _synthetic_pair(dx=shift, dy=0.0)
    result = measure_exterior_drift(original, edited, mask, threshold_px=10.0)
    measured = result.drift_per_frame[0]
    if shift == 0.0:
        assert measured <= 0.25, f"zero shift measured {measured:.4f}px (limit 0.25)"
    else:
        assert abs(measured - shift) <= 0.25, (
            f"shift {shift}px measured {measured:.4f}px (±0.25 calibration gate)"
        )


def test_interior_change_does_not_move_exterior_drift():
    original, edited, mask = _synthetic_pair(dx=0.0, dy=0.0)
    h, w = original.shape[1:3]
    edited = edited.clone()
    edited[:, h // 4 : 3 * h // 4, w // 4 : 3 * w // 4, :] = torch.rand(1, h // 2, w // 2, 3)
    base = measure_exterior_drift(original, original, mask, threshold_px=10.0)
    result = measure_exterior_drift(original, edited, mask, threshold_px=10.0)
    assert result.drift_per_frame[0] <= base.drift_per_frame[0] + 0.5


def test_drift_qc_node_reports_pass_on_zero_shift():
    from mmx_nodes.drift_qc import MiniMaxH3_DriftQC

    original = torch.rand(1, 48, 48, 3)
    edited = inject_pixel_shift(original, 0.0, 0.0)
    mask = torch.zeros(48, 48)
    mask[12:36, 12:36] = 1.0
    drift_json, heat, passed, report = MiniMaxH3_DriftQC.execute(original, edited, mask, 2.0)
    assert isinstance(drift_json, str)
    assert heat.shape[0] == 1
    assert isinstance(passed, bool)
    assert "exterior drift" in report


@pytest.mark.parametrize("size", [192, 256])
@pytest.mark.parametrize("shift", [0.0, 0.5, 1.0, 2.0])
def test_drift_calibration_at_real_crop_sizes(size, shift):
    """The calibration gate must hold at the sizes this pack actually runs at.

    The 64x64 test above passes partly by accident: `_prep_for_raft` upscales anything
    under RAFT_MIN_SIDE (128) to 128, so a 64px frame silently gets a 2x supersample.
    Real crops are 128-512px and got none, where RAFT-small under-reports badly —
    measured before the fix, at 192x192 with no supersample:
        1.0px -> 0.7285 (err 0.272), 2.0px -> 1.6848 (err 0.315)
    i.e. the +/-0.25 gate was failing exactly where it matters while CI stayed green.

    This test pins the operating regime so that can't happen again.
    """
    torch.manual_seed(0)
    original = torch.rand(1, size, size, 3)
    edited = inject_pixel_shift(original, shift, 0.0)
    q = size // 4
    mask = torch.zeros(size, size)
    mask[q : 3 * q, q : 3 * q] = 1.0

    measured = measure_exterior_drift(
        original, edited, mask, threshold_px=10.0
    ).drift_per_frame[0]
    assert abs(measured - shift) <= 0.25, (
        f"{size}px frame, {shift}px shift measured {measured:.4f}px "
        "(±0.25 calibration gate at a real crop size)"
    )


@pytest.mark.parametrize("size", [16, 32, 48])
def test_drift_calibration_below_supersample_reach(size):
    """Sub-64px frames stay inside the gate.

    supersample=2 alone only reaches 128 (RAFT's floor) from a 64px input, so the
    concern was that smaller frames would fall short. They do not: `_prep_for_raft`
    enforces RAFT_MIN_SIDE=128 independently, which hands a 16px frame an 8x effective
    upscale. Measured 1.0px injected -> 16px: 1.0114 (err 0.011), 32px: 1.0019
    (err 0.002), 48px: 0.8670 (err 0.133). No adaptive factor or raise is needed;
    this test exists so that stays true if _prep_for_raft is ever touched.
    """
    torch.manual_seed(0)
    original = torch.rand(1, size, size, 3)
    q = max(1, size // 4)
    mask = torch.zeros(size, size)
    mask[q : 3 * q, q : 3 * q] = 1.0
    for shift in (0.0, 1.0):
        measured = measure_exterior_drift(
            original, inject_pixel_shift(original, shift, 0.0), mask, threshold_px=10.0
        ).drift_per_frame[0]
        assert abs(measured - shift) <= 0.25, f"{size}px / {shift}px -> {measured:.4f}"
