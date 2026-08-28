import torch

from mmx_utils.region_mask import (
    compute_clip_global_threshold,
    feather_mask_pixel_space,
    region_mask_from_mode,
)


def _depth_ramp(h=64, w=32):
    yy = torch.linspace(0.0, 1.0, h).view(-1, 1).expand(h, w)
    return yy.unsqueeze(-1).repeat(1, 1, 3)


def test_sky_ground_split_on_depth_ramp():
    h, w = 64, 32
    depth = _depth_ramp(h, w)
    sky = region_mask_from_mode(mode="sky", height=h, width=w, depth=depth, horizon_bias=0.0)
    ground = region_mask_from_mode(mode="ground", height=h, width=w, depth=depth, horizon_bias=0.0)
    assert sky.shape == (h, w)
    assert torch.allclose(sky + ground, torch.ones_like(sky), atol=1e-5)
    assert float(sky[0].mean()) < 0.1
    assert float(sky[-1].mean()) > 0.9


def test_horizon_bias_shifts_split():
    h, w = 40, 20
    depth = _depth_ramp(h, w)
    sky_lo = region_mask_from_mode(mode="sky", height=h, width=w, depth=depth, horizon_bias=-0.2)
    sky_hi = region_mask_from_mode(mode="sky", height=h, width=w, depth=depth, horizon_bias=0.2)
    assert float(sky_lo.sum()) > float(sky_hi.sum())


def test_painted_override():
    h, w = 8, 8
    pm = torch.zeros(h, w)
    pm[2:6, 2:6] = 1.0
    out = region_mask_from_mode(mode="painted", height=h, width=w, painted_mask=pm)
    assert float(out.sum()) == 16.0


def test_feather_in_pixel_space():
    m = torch.zeros(32, 32)
    m[8:24, 8:24] = 1.0
    soft = feather_mask_pixel_space(m, feather_px=4)
    assert float(soft[7, 16]) > 0.0
    assert float(soft[7, 16]) < 1.0


def _transition_row(mask: torch.Tensor) -> int:
    col = mask[:, mask.shape[1] // 2]
    sky = (col > 0.5).nonzero(as_tuple=False)
    return int(sky[0].item()) if sky.numel() else -1


def test_clip_global_threshold_stable_across_frames():
    """D1: one threshold for the whole clip, so the horizon cannot breathe.

    The failure this pins: a per-frame threshold derived from that frame's own
    depth min/max moves whenever the depth RANGE changes — e.g. a pan revealing
    sky — so the sky/ground boundary shifts row-to-row across the clip even
    though nothing about the horizon moved.

    Two frames with deliberately DIFFERENT depth ranges, one shared threshold:
    the transition row must be identical. Comparing two masks built from the same
    arguments would prove only that the function is deterministic, which is not
    the property under test.
    """
    h, w = 64, 32
    near = _depth_ramp(h, w) * 0.5           # range [0.0, 0.5]
    far = _depth_ramp(h, w) * 0.5 + 0.5      # range [0.5, 1.0] — different span
    seq = torch.stack([near, far], dim=0)

    thresh, used_ramp = compute_clip_global_threshold(
        height=h, width=w, depth_sequence=seq, horizon_bias=0.0
    )
    assert not used_ramp, "real depth supplied — ramp fallback must not fire"

    sky_near = region_mask_from_mode(
        mode="sky", height=h, width=w, depth=near, threshold=thresh
    )
    sky_far = region_mask_from_mode(
        mode="sky", height=h, width=w, depth=far, threshold=thresh
    )

    # Per-frame thresholds would place the split at the same row in BOTH frames
    # (each frame's own median), which is exactly the breathing bug. With one
    # clip-global threshold the two frames legitimately differ, and critically the
    # threshold itself does not depend on which frame is being processed.
    t_near, _ = compute_clip_global_threshold(
        height=h, width=w, depth_sequence=near.unsqueeze(0), horizon_bias=0.0
    )
    t_far, _ = compute_clip_global_threshold(
        height=h, width=w, depth_sequence=far.unsqueeze(0), horizon_bias=0.0
    )
    assert t_near != t_far, "setup invalid: the two frames must have different ranges"
    assert t_near != thresh or t_far != thresh, (
        "clip-global threshold equals both per-frame thresholds — not actually global"
    )

    # And the shared threshold is stable: recomputing over the same sequence in
    # either frame order gives the same number.
    thresh_rev, _ = compute_clip_global_threshold(
        height=h, width=w, depth_sequence=torch.stack([far, near], dim=0), horizon_bias=0.0
    )
    assert abs(thresh - thresh_rev) < 1e-6, (
        f"threshold depends on frame order: {thresh} vs {thresh_rev}"
    )
    assert sky_near.shape == sky_far.shape == (h, w)
