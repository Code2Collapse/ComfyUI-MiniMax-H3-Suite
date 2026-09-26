"""Tests for plate restore slice 1: color_space, registration, plate_restore."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from mmx_utils.color_space import (
    linear_srgb_to_oklab,
    linear_to_srgb,
    oklab_to_linear_srgb,
    srgb_to_linear,
)
from mmx_utils.drift_qc import inject_pixel_shift
from mmx_utils.plate_restore import (
    border_ramp_alpha,
    paste_crop_with_feather,
    restore_frames,
)
from mmx_utils.registration import masked_ncc_shift


def _textured_image(h: int = 128, w: int = 128, seed: int = 0) -> torch.Tensor:
    """A plate-like texture: gradients plus BAND-LIMITED noise.

    The first version used per-pixel white noise, and every sub-pixel test
    failed while the estimator itself was accurate to 0.002 px. A sub-pixel
    shift of white noise is not a shift: bilinear resampling just averages
    uncorrelated neighbours, and central-difference gradients of white noise
    carry no position information. Real plates are band-limited by the lens,
    so the noise is blurred to a few pixels of correlation - which is also
    the regime the node is actually used in.
    """
    g = torch.Generator().manual_seed(seed)
    y = torch.linspace(0, 1, h).unsqueeze(1).expand(h, w)
    x = torch.linspace(0, 1, w).unsqueeze(0).expand(h, w)
    raw = torch.rand(1, 1, h, w, generator=g)
    kernel = torch.ones(1, 1, 5, 5) / 25.0
    n = torch.nn.functional.conv2d(raw, kernel, padding=2)[0, 0]
    n = (n - n.mean()) / n.std().clamp_min(1e-6) * 0.05 + 0.075
    rgb = torch.stack([0.3 + 0.5 * x + n, 0.2 + 0.4 * y + n, 0.25 + 0.3 * (x * y) + n], dim=-1)
    return rgb.clamp(0, 1)


def _irregular_ring_mask(h: int, w: int, asymmetric: bool = False) -> torch.Tensor:
    """Ring mask avoiding full-frame (tests masked NCC, not zero-padded phase corr)."""
    cy, cx = h // 2, w // 2
    yy, xx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    dist = torch.sqrt((yy - cy).float() ** 2 + (xx - cx).float() ** 2)
    inner, outer = (h * 0.22), (h * 0.42)
    ring = ((dist >= inner) & (dist <= outer)).float()
    if asymmetric:
        ring[:, cx + w // 8 :] = 0.0
        ring[: h // 4, :] = 0.0
    return ring


# ── color_space ─────────────────────────────────────────────────────────────


def test_srgb_linear_roundtrip():
    x = torch.tensor([0.0, 0.18, 0.5, 1.0, 2.5, 4.0])
    y = linear_to_srgb(srgb_to_linear(x))
    assert torch.allclose(x, y, atol=1e-5, rtol=1e-4)


def test_oklab_roundtrip():
    lin = torch.tensor([[1.0, 0.5, 0.25], [0.1, 0.2, 0.3]])
    back = oklab_to_linear_srgb(linear_srgb_to_oklab(lin))
    assert torch.allclose(lin, back, atol=1e-5)


def test_oklab_white():
    white_lin = torch.tensor([[1.0, 1.0, 1.0]])
    lab = linear_srgb_to_oklab(white_lin)
    assert abs(float(lab[0, 0]) - 1.0) < 1e-4
    assert abs(float(lab[0, 1])) < 1e-4
    assert abs(float(lab[0, 2])) < 1e-4


# ── masked_ncc_shift ────────────────────────────────────────────────────────
#
# Sign contract (matches drift_qc.inject_pixel_shift docstring behaviour):
#   inject_pixel_shift(img, a, b) samples at (x+a, y+b) → content moves by (-a, -b).
#   If moving = inject_pixel_shift(fixed, a, b), masked_ncc_shift must return
#   dx = -a, dy = -b so inject_pixel_shift(moving, dx, dy) ≈ fixed.


@pytest.mark.parametrize("dx_inj,dy_inj", [
    (0.0, 0.0),
    (0.25, 0.0),
    (0.0, 0.25),
    (0.5, -0.25),
    (1.0, 0.5),
    (2.0, -1.0),
    (-1.5, 0.75),
])
def test_masked_ncc_shift_recovery(dx_inj, dy_inj):
    fixed = _textured_image(128, 128, seed=1)
    ring = _irregular_ring_mask(128, 128, asymmetric=(dx_inj != 0))
    fixed_luma = 0.2126 * fixed[..., 0] + 0.7152 * fixed[..., 1] + 0.0722 * fixed[..., 2]
    moving = inject_pixel_shift(fixed.unsqueeze(0), dx_inj, dy_inj).squeeze(0)
    moving_luma = 0.2126 * moving[..., 0] + 0.7152 * moving[..., 1] + 0.0722 * moving[..., 2]
    dy, dx, peak = masked_ncc_shift(fixed_luma, moving_luma, ring > 0.5, ring > 0.5)
    err_x = abs(dx - (-dx_inj))
    err_y = abs(dy - (-dy_inj))
    assert err_x <= 0.05, f"dx inj={dx_inj} want={-dx_inj} got={dx} err={err_x}"
    assert err_y <= 0.05, f"dy inj={dy_inj} want={-dy_inj} got={dy} err={err_y}"
    assert peak > 0.3


def _bilinear_roundtrip_floor(dx: float, dy: float) -> float:
    """Irreducible error from shift then inverse shift through bilinear grid_sample."""
    fixed = _textured_image(96, 96, seed=11)
    once = inject_pixel_shift(fixed.unsqueeze(0), dx, dy).squeeze(0)
    back = inject_pixel_shift(once.unsqueeze(0), -dx, -dy).squeeze(0)
    return float((back - fixed).abs().mean().item())


def test_masked_ncc_shift_sign_convention():
    """Returned (dx,dy) applied to moving aligns it to fixed."""
    dx_inj, dy_inj = 1.25, -0.5
    fixed = _textured_image(96, 96, seed=3)
    ring = _irregular_ring_mask(96, 96)
    luma = 0.2126 * fixed[..., 0] + 0.7152 * fixed[..., 1] + 0.0722 * fixed[..., 2]
    moving = inject_pixel_shift(fixed.unsqueeze(0), dx_inj, dy_inj).squeeze(0)
    ml = 0.2126 * moving[..., 0] + 0.7152 * moving[..., 1] + 0.0722 * moving[..., 2]
    dy, dx, _ = masked_ncc_shift(luma, ml, ring > 0.5)
    corrected = inject_pixel_shift(moving.unsqueeze(0), dx, dy).squeeze(0)
    sel = ring > 0.5
    err = (corrected[..., :3][sel] - fixed[sel]).abs().mean()
    floor = _bilinear_roundtrip_floor(dx_inj, dy_inj)
    assert float(err) <= floor + 0.005


def test_masked_ncc_skimage_crosscheck():
    pytest.importorskip("skimage")
    from skimage.registration._masked_phase_cross_correlation import _masked_phase_cross_correlation

    fixed = _textured_image(64, 64, seed=7).numpy()
    ring = _irregular_ring_mask(64, 64).numpy() > 0.5
    fl = (0.2126 * fixed[..., 0] + 0.7152 * fixed[..., 1] + 0.0722 * fixed[..., 2]).astype(
        np.float64
    )
    moving = (
        inject_pixel_shift(torch.from_numpy(fixed).unsqueeze(0), 1.0, 0.5).squeeze(0).numpy()
    )
    ml = (0.2126 * moving[..., 0] + 0.7152 * moving[..., 1] + 0.0722 * moving[..., 2]).astype(
        np.float64
    )
    ref = _masked_phase_cross_correlation(fl, ml, ring, overlap_ratio=0.3)
    dy, dx, _ = masked_ncc_shift(
        torch.from_numpy(fl), torch.from_numpy(ml), torch.from_numpy(ring)
    )
    # skimage shift is opposite inject_pixel_shift correction (see sign contract above).
    assert abs(dx + float(ref[1])) <= 0.5
    assert abs(dy + float(ref[0])) <= 0.5


# ── restore_frames ──────────────────────────────────────────────────────────


def _synthetic_restore_setup(
    frames: int = 3,
    crop_h: int = 64,
    crop_w: int = 64,
    hdr: bool = False,
):
    full_h, full_w = 128, 128
    x0, y0 = 32, 32
    plate = torch.zeros(frames, full_h, full_w, 3)
    for t in range(frames):
        plate[t] = _textured_image(full_h, full_w, seed=10 + t)
    if hdr:
        # Peak 4.0 in plate space (HDR survives path without sRGB clamp).
        plate = plate * 3.5 + 0.5
    gen = plate.clone()
    edit = torch.zeros(crop_h, crop_w)
    edit[crop_h // 4 : 3 * crop_h // 4, crop_w // 4 : 3 * crop_w // 4] = 1.0
    boxes = [{"x": x0, "y": y0, "width": crop_w, "height": crop_h}] * frames
    gen_crops = gen[:, y0 : y0 + crop_h, x0 : x0 + crop_w, :].clone()
    return plate, gen_crops, boxes, edit.unsqueeze(0).expand(frames, -1, -1), x0, y0


def test_restore_exterior_bit_exact():
    plate, gen_crops, boxes, edit_mask, x0, y0 = _synthetic_restore_setup()
    out, mask, _ = restore_frames(
        gen_crops, plate, boxes, edit_mask, 8, 16, "off", 4.0, "off", False
    )
    exterior = mask[0] < 1e-6
    assert torch.equal(out[0, exterior], plate[0, exterior])


def test_restore_colour_drift_removed():
    plate, gen_crops, boxes, edit_mask, x0, y0 = _synthetic_restore_setup(frames=5)
    h, w = gen_crops.shape[1], gen_crops.shape[2]
    for t in range(gen_crops.shape[0]):
        g = 1.05 + 0.02 * t
        b = 0.03 * t
        gen_crops[t, ..., 0] = gen_crops[t, ..., 0] * g + b
        gen_crops[t, ..., 1] = gen_crops[t, ..., 1] * (g - 0.01) - b
        gen_crops[t, ..., 2] = gen_crops[t, ..., 2] * (g + 0.01) + 2 * b
    out, _, rep = restore_frames(
        gen_crops, plate, boxes, edit_mask, 8, 12, "off", 4.0, "affine", False
    )
    edit = edit_mask[0] > 0.5
    crop_out = out[0, y0 : y0 + h, x0 : x0 + w, :]
    crop_plate = plate[0, y0 : y0 + h, x0 : x0 + w, :]
    err = (crop_out[edit] - crop_plate[edit]).abs().mean().item() * 255.0
    assert err < 0.5, f"mean sRGB error {err}"


def test_restore_shift_correct_and_refuse():
    plate, gen_crops, boxes, edit_mask, x0, y0 = _synthetic_restore_setup()
    h, w = gen_crops.shape[1], gen_crops.shape[2]

    shifted = gen_crops.clone()
    for t in range(shifted.shape[0]):
        c = shifted[t]
        shifted[t] = inject_pixel_shift(c.unsqueeze(0), 1.0, 0.0).squeeze(0)
    out_ok, _, rep_ok = restore_frames(
        shifted, plate, boxes, edit_mask, 8, 12, "correct", 4.0, "off", False
    )
    assert rep_ok["frames"][0]["shift_applied"] is True
    ring_plate = plate[0, y0 : y0 + h, x0 : x0 + w, :]
    ring_out = out_ok[0, y0 : y0 + h, x0 : x0 + w, :]
    exterior = edit_mask[0] < 0.5
    err = (ring_out[exterior] - ring_plate[exterior]).abs().mean()
    assert float(err) < 0.05

    big = gen_crops.clone()
    for t in range(big.shape[0]):
        big[t] = inject_pixel_shift(big[t].unsqueeze(0), 10.0, 0.0).squeeze(0)
    _, _, rep_big = restore_frames(
        big, plate, boxes, edit_mask, 8, 12, "correct", 4.0, "off", False
    )
    assert rep_big["frames"][0]["shift_refused"] is True
    assert rep_big["frames"][0]["shift_applied"] is False


def test_restore_noop_when_identical():
    plate, gen_crops, boxes, edit_mask, _, _ = _synthetic_restore_setup()
    out, _, rep = restore_frames(
        gen_crops, plate, boxes, edit_mask, 8, 12, "correct", 4.0, "affine", False
    )
    assert (out - plate).abs().max().item() < 1e-6
    assert rep["frames"][0]["shift_applied"] is False
    assert rep["frames"][0]["colour_magnitude_linear"] < 1e-4


def test_restore_hdr_no_clamp():
    plate, gen_crops, boxes, edit_mask, x0, y0 = _synthetic_restore_setup(hdr=True)
    out, _, _ = restore_frames(
        gen_crops, plate, boxes, edit_mask, 8, 12, "off", 4.0, "off", False
    )
    exterior = edit_mask[0] < 0.5
    crop = out[0, y0 : y0 + gen_crops.shape[1], x0 : x0 + gen_crops.shape[2], :]
    assert float(crop[exterior].max()) > 1.0
    # The property is "nothing is clamped": the exterior is the literal plate,
    # so the output's peak must equal the plate's. It used to assert a fixed
    # 4.0, which tied the test to whatever the fixture happened to peak at.
    assert float(out[0].max()) >= float(plate[0].max()) - 1e-6
    assert float(plate[0].max()) > 3.0, "the HDR fixture no longer exercises values above 1"


def _golden_uncrop_loop(
    cropped_images,
    original_images,
    boxes,
    feather,
    cropped_masks=None,
):
    """Copy of pre-extraction SubjectUncrop paste loop for regression."""
    img_h, img_w = original_images.shape[1], original_images.shape[2]
    out = original_images.clone()
    for i, b in enumerate(boxes):
        x, y, w, h = (int(round(b[k])) for k in ("x", "y", "width", "height"))
        crop = cropped_images[i]
        if crop.shape[0] != h or crop.shape[1] != w:
            crop = torch.nn.functional.interpolate(
                crop.movedim(-1, 0).unsqueeze(0).float(),
                size=(h, w),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0).movedim(0, -1).to(cropped_images.dtype)

        alpha = border_ramp_alpha(h, w, x, y, img_h, img_w, feather, device=out.device, dtype=out.dtype)
        if cropped_masks is not None:
            m = cropped_masks[i]
            if m.shape[0] != h or m.shape[1] != w:
                m = torch.nn.functional.interpolate(
                    m.unsqueeze(0).unsqueeze(0).float(),
                    size=(h, w),
                    mode="bilinear",
                    align_corners=False,
                )[0, 0]
            alpha = alpha * m.clamp(0.0, 1.0).to(dtype=out.dtype, device=out.device)

        alpha = alpha[..., None]
        region = out[i, y : y + h, x : x + w, :]
        out[i, y : y + h, x : x + w, :] = crop.to(region) * alpha + region * (1 - alpha)
    return out


def test_subject_uncrop_matches_golden():
    plate = _textured_image(96, 96, seed=99).unsqueeze(0)
    crop = plate[:, 20:70, 25:75, :].clone() * 0.8 + 0.1
    boxes = [{"x": 25, "y": 20, "width": 50, "height": 50}]
    mask = torch.zeros(1, 50, 50)
    mask[0, 10:40, 10:40] = 1.0
    golden = _golden_uncrop_loop(crop, plate, boxes, 12, mask)
    from mmx_utils.plate_restore import uncrop_paste_frames

    def _ri(img, w, h):
        if img.shape[0] == h and img.shape[1] == w:
            return img
        return torch.nn.functional.interpolate(
            img.movedim(-1, 0).unsqueeze(0).float(),
            size=(h, w),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0).movedim(0, -1).to(img.dtype)

    def _rm(m, w, h):
        if m.shape[0] == h and m.shape[1] == w:
            return m
        return torch.nn.functional.interpolate(
            m.unsqueeze(0).unsqueeze(0).float(),
            size=(h, w),
            mode="bilinear",
            align_corners=False,
        )[0, 0]

    shared = uncrop_paste_frames(crop, plate, boxes, 12, mask, resize_image=_ri, resize_mask=_rm)
    assert torch.equal(golden, shared)


def test_paste_crop_with_feather_matches_golden():
    plate = _textured_image(80, 80)
    crop = plate[10:50, 15:55, :].clone() * 0.7
    out = paste_crop_with_feather(plate, crop, 15, 10, 40, 40, 8, 80, 80, None)
    golden = _golden_uncrop_loop(crop.unsqueeze(0), plate.unsqueeze(0), [{"x": 15, "y": 10, "width": 40, "height": 40}], 8)[0]
    assert torch.equal(out, golden)


@pytest.mark.parametrize(
    "dx_inj,dy_inj",
    [(0.0, 0.0), (0.25, 0.0), (0.5, 0.25), (1.0, 0.5), (2.0, -1.0), (-1.5, 0.75)],
)
def test_masked_ncc_shift_errors_report(dx_inj, dy_inj):
    """Emit per-shift recovery errors for the acceptance log."""
    fixed = _textured_image(128, 128, seed=42)
    ring = _irregular_ring_mask(128, 128, asymmetric=True)
    fl = 0.2126 * fixed[..., 0] + 0.7152 * fixed[..., 1] + 0.0722 * fixed[..., 2]
    moving = inject_pixel_shift(fixed.unsqueeze(0), dx_inj, dy_inj).squeeze(0)
    ml = 0.2126 * moving[..., 0] + 0.7152 * moving[..., 1] + 0.0722 * moving[..., 2]
    dy, dx, peak = masked_ncc_shift(fl, ml, ring > 0.5)
    err = math.hypot(dx - (-dx_inj), dy - (-dy_inj))
    print(f"shift inj=({dx_inj},{dy_inj}) rec=({dx:.4f},{dy:.4f}) err={err:.4f} peak={peak:.4f}")
    assert err <= 0.05
