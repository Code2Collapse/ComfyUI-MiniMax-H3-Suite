import torch

from mmx_utils.sigma_schedule import (
    TRAINED_SHIFT_RATIO,
    apply_ratio_lock,
    format_sigma_inspector_table,
    time_shift_sigma,
    validate_h3_sigmas,
)


def test_time_shift_sigma_scalar():
    out = time_shift_sigma(0.5, 12.0, 3.0)
    assert 0.0 < out < 1.0


def test_apply_ratio_lock_enforces_ratio():
    sv, sa, note = apply_ratio_lock(12.0, 5.0, ratio_lock=True, ratio=4.0)
    assert sv == 12.0
    assert sa == 3.0
    assert "ratio_lock" in note


def test_apply_ratio_lock_off_preserves_audio():
    sv, sa, note = apply_ratio_lock(12.0, 2.5, ratio_lock=False)
    assert sa == 2.5
    assert note == ""


def test_validate_h3_sigmas_ok():
    sigmas = torch.linspace(1.0, 0.0, 10)
    ok, report = validate_h3_sigmas(sigmas, scheduler="simple", denoise=1.0)
    assert ok
    assert "OK" in report


def test_validate_h3_sigmas_rejects_split():
    sigmas = torch.linspace(1.0, 0.0, 5)
    ok, report = validate_h3_sigmas(sigmas, scheduler="split_sigmas")
    assert not ok
    assert "SplitSigmas" in report


def test_format_sigma_inspector_table_has_header():
    sigmas = torch.tensor([1.0, 0.5, 0.0])
    table = format_sigma_inspector_table(sigmas)
    assert "sigma_v" in table
    assert "sigma_a" in table
    assert table.count("|") >= 3


def test_build_sigma_inspector_json_matches_table_steps():
    from mmx_utils.sigma_schedule import build_sigma_inspector_json

    sigmas = torch.tensor([1.0, 0.5, 0.0])
    payload = build_sigma_inspector_json(sigmas)
    import json

    data = json.loads(payload)
    assert data["sigmas_v"] == [1.0, 0.5, 0.0]
    assert len(data["steps"]) == 3
    assert data["steps"][0]["sigma_v"] == 1.0
