import torch

from mmx_utils.sigma_schedule import (
    TRAINED_SHIFT_AUDIO,
    TRAINED_SHIFT_RATIO,
    TRAINED_SHIFT_VIDEO,
    apply_ratio_lock,
    build_sigma_inspector_json,
    d_sigma_dt,
    dsigma_a_over_dsigma_v,
    format_sigma_inspector_table,
    sigma_to_t,
    synthesize_linear_sigmas,
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
    sigmas = torch.tensor([1.0, 0.5, 0.0])
    payload = build_sigma_inspector_json(sigmas)
    import json

    data = json.loads(payload)
    assert data["sigmas_v"] == [1.0, 0.5, 0.0]
    assert len(data["steps"]) == 3
    assert data["steps"][0]["sigma_v"] == 1.0
    assert "dsigma_a_dsigma_v_fd" in data["steps"][0]


def test_analytic_dsigma_ratio_at_sigma_v_one():
    t = sigma_to_t(1.0, TRAINED_SHIFT_VIDEO)
    ratio = dsigma_a_over_dsigma_v(t, TRAINED_SHIFT_VIDEO, TRAINED_SHIFT_AUDIO)
    assert abs(ratio - 4.0) <= 1e-6


def test_analytic_dsigma_ratio_as_t_approaches_zero():
    t = 1e-9
    ratio = dsigma_a_over_dsigma_v(t, TRAINED_SHIFT_VIDEO, TRAINED_SHIFT_AUDIO)
    assert abs(ratio - 0.25) <= 1e-6


def test_analytic_dsigma_ratio_invariant_to_steps():
    import json

    ratios_21 = []
    for sig_v in synthesize_linear_sigmas(20).tolist():
        t = sigma_to_t(sig_v, TRAINED_SHIFT_VIDEO)
        ratios_21.append(dsigma_a_over_dsigma_v(t, TRAINED_SHIFT_VIDEO, TRAINED_SHIFT_AUDIO))
    ratios_50 = []
    for sig_v in synthesize_linear_sigmas(49).tolist():
        t = sigma_to_t(sig_v, TRAINED_SHIFT_VIDEO)
        ratios_50.append(dsigma_a_over_dsigma_v(t, TRAINED_SHIFT_VIDEO, TRAINED_SHIFT_AUDIO))
    # Compare at matched sigma_v values (first schedule is subset spacing — compare at σ_v=1)
    t1 = sigma_to_t(1.0, TRAINED_SHIFT_VIDEO)
    r_ref = dsigma_a_over_dsigma_v(t1, TRAINED_SHIFT_VIDEO, TRAINED_SHIFT_AUDIO)
    payload_21 = json.loads(build_sigma_inspector_json(synthesize_linear_sigmas(20)))
    payload_50 = json.loads(build_sigma_inspector_json(synthesize_linear_sigmas(49)))
    r21 = payload_21["steps"][0]["dsigma_a_dsigma_v"]
    r50 = payload_50["steps"][0]["dsigma_a_dsigma_v"]
    assert abs(r21 - r_ref) <= 1e-6
    assert abs(r50 - r_ref) <= 1e-6
    assert abs(r21 - r50) <= 1e-6


def test_finite_difference_differs_from_analytic_at_sigma_v_one():
    import json

    sigmas = synthesize_linear_sigmas(20)
    payload = json.loads(build_sigma_inspector_json(sigmas))
    analytic = payload["steps"][0]["dsigma_a_dsigma_v"]
    fd = payload["steps"][0]["dsigma_a_dsigma_v_fd"]
    assert abs(analytic - 4.0) <= 1e-6
    assert abs(fd - analytic) > 0.1


def test_d_sigma_dt_matches_closed_form():
    t = 0.5
    shift = 12.0
    expected = shift / (1.0 + (shift - 1.0) * t) ** 2
    assert abs(d_sigma_dt(t, shift) - expected) <= 1e-12


def test_split_sigmas_caught_structurally_not_by_name():
    """A real SplitSigmas node leaves scheduler='simple' — the name check misses it.

    Upstream SplitSigmas hands downstream the truncated tensor while the scheduler
    widget still reads 'simple', so matching on the scheduler STRING let an illegal
    H3 configuration validate clean. The tensor itself is the evidence: a split high
    half stops above zero.
    """
    import torch

    from mmx_utils.sigma_schedule import validate_h3_sigmas

    # high half of a split — scheduler name is entirely innocent
    high = torch.linspace(1.0, 0.42, 11)
    ok, report = validate_h3_sigmas(high, scheduler="simple", denoise=1.0)
    assert ok is False, "split high half must be refused"
    assert "0.42" in report and "SplitSigmas" in report
    assert "denoise" in report, "refusal must name the legal alternative"


def test_partial_denoise_still_allowed():
    """The split check must not fire on legitimate BasicScheduler partial denoise.

    Partial denoise starts BELOW 1.0 but still lands on 0.0. If this ever fails,
    the split fingerprint has been broadened into something that breaks real work.
    """
    import torch

    from mmx_utils.sigma_schedule import validate_h3_sigmas

    partial = torch.linspace(0.62, 0.0, 13)
    ok, report = validate_h3_sigmas(partial, scheduler="simple", denoise=0.62)
    assert ok is True, f"partial denoise wrongly refused: {report}"
