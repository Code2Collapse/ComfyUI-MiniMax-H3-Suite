"""Tests for N3 DetailMatch, N4 ReferenceColorMatch, N5 PixelRepair (slice 3)."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from mmx_utils.color_space import linear_srgb_to_oklab, linear_to_srgb, oklab_to_linear_srgb, srgb_to_linear
from mmx_utils.detail_match import detail_match_frames
from mmx_utils.frequency import laplacian_pyramid
from mmx_utils.pixel_repair import repair_frames
from mmx_utils.reference_color_match import reference_color_match_frames


def _band_limited_texture(h: int, w: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    raw = torch.rand(1, 1, h, w, generator=g)
    kernel = torch.ones(1, 1, 5, 5) / 25.0
    n = F.conv2d(raw, kernel, padding=2)[0, 0]
    n = (n - n.mean()) / n.std().clamp_min(1e-6) * 0.05
    y = torch.linspace(0, 1, h).unsqueeze(1).expand(h, w)
    x = torch.linspace(0, 1, w).unsqueeze(0).expand(h, w)
    rgb = torch.stack([0.45 + 0.35 * x + n, 0.38 + 0.3 * y + n, 0.42 + 0.25 * (x * y) + n], dim=-1)
    return rgb.clamp(0, 1)


def _gaussian_blur_hwc(img: torch.Tensor, sigma: float) -> torch.Tensor:
    r = max(1, int(round(sigma * 2)))
    x = img.permute(2, 0, 1).unsqueeze(0).float()
    k = 2 * r + 1
    t = torch.arange(k, dtype=torch.float32) - r
    g = torch.exp(-(t * t) / (2 * sigma * sigma))
    g = (g / g.sum()).view(1, 1, 1, k)
    c = x.shape[1]
    gk = g.repeat(c, 1, 1, 1)
    gv = gk.transpose(2, 3)
    x = F.conv2d(F.pad(x, (r, r, 0, 0), mode="reflect"), gk, groups=c)
    x = F.conv2d(F.pad(x, (0, 0, r, r), mode="reflect"), gv, groups=c)
    return x.squeeze(0).permute(1, 2, 0).to(img.dtype)


def _crop_scene(h: int = 96, w: int = 96, n: int = 5, seed: int = 3):
    plate_tex = _band_limited_texture(h, w, seed)
    full = torch.zeros(n, h + 32, w + 32, 3)
    ox, oy = 16, 16
    for i in range(n):
        full[i, oy : oy + h, ox : ox + w, :] = plate_tex
    bbox = {"x": ox, "y": oy, "width": w, "height": h}
    yy, xx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    cy, cx = h // 2, w // 2
    edit = ((xx - cx).float() ** 2 + (yy - cy).float() ** 2 < (h * 0.22) ** 2).float()
    edit_mask = edit.unsqueeze(0).expand(n, -1, -1)
    cropped = full[:, oy : oy + h, ox : ox + w, :].clone()
    return full, cropped, edit_mask, bbox


def _edit_band_energy(img_hwc: torch.Tensor, edit_mask: torch.Tensor, level: int) -> float:
    pyr = laplacian_pyramid(img_hwc)
    band = pyr[min(level, len(pyr) - 1)]
    m = edit_mask > 0.5
    h, w = band.shape[-2], band.shape[-1]
    if m.shape[0] != h:
        m = F.interpolate(m.unsqueeze(0).unsqueeze(0).float(), size=(h, w), mode="nearest")[0, 0] > 0.5
    e = 0.0
    for ch in range(min(3, band.shape[1])):
        b = band[0, ch][m]
        if b.numel():
            e += float((b * b).mean().item())
    return e / 3.0


# ── N3 DetailMatch ────────────────────────────────────────────────────────────


def _level_ratios(plate, out, edit_mask):
    return [T_edit(out, edit_mask, k) / max(T_edit(plate, edit_mask, k), 1e-12) for k in range(4)]


def T_edit(img, edit_mask, level):
    return _edit_band_energy(img, edit_mask, level)


def test_detail_match_restores_a_softening_it_can_invert():
    """A 0.8 px softening: its inverse fits under max_gain at the mid
    frequencies, so the edit region's band energies should come most of the
    way back - and no band may end up ABOVE the plate's. The first version of
    this node overshot level 1 to 1.44x the plate (a Laplacian pyramid is not
    a clean filter bank); the gain is now measured and applied as one radial
    FFT filter, so what is measured is what is applied.

    The VAE softens the WHOLE crop, not just the edit - that is what makes
    the loss measurable on the ring around it.

    Crop size matters and is measured: on a 96 px crop only 16 px patches fit
    the ring, a 16-point Hann FFT smears the spectrum, and level 1 closes 57%
    of its gap - under-correction, the safe direction. From 192 px, 32 px
    patches fit and it closes 88-89%. Real H3 crops are ~2 MP, so the test uses
    a realistic size rather than a looser bar.
    """
    full, cropped, edit_mask, bbox = _crop_scene(h=192, w=192, n=3)
    plate = cropped.clone()
    gen = torch.stack([_gaussian_blur_hwc(f, 0.8) for f in plate])
    out, report = detail_match_frames(
        gen, full, [bbox] * gen.shape[0], edit_mask,
        grow_px=8, sharpen=1.0, max_gain=2.5, grain=0.0, seed=0,
    )
    before = _level_ratios(plate[0], gen[0], edit_mask[0])
    after = _level_ratios(plate[0], out[0], edit_mask[0])
    gap_before = 1.0 - before[1]
    gap_after = abs(1.0 - after[1])
    assert gap_after < 0.3 * gap_before, f"level 1: {before[1]:.2f} -> {after[1]:.2f}"
    for k, r in enumerate(after):
        assert r <= 1.05, f"level {k} overshot to {r:.2f}x the plate"
    for frame in report["frames"]:
        for g in frame["gains"]:
            assert g >= 1.0 - 1e-6, "the node must never blur"


def test_detail_match_improves_a_heavy_softening_without_overshoot():
    """A 1.2 px softening attenuates the top frequencies ~30x; inverting that
    would mostly amplify noise, which is what max_gain is for. So: every band
    must move toward the plate, and none may pass it."""
    full, cropped, edit_mask, bbox = _crop_scene()
    plate = cropped.clone()
    gen = torch.stack([_gaussian_blur_hwc(f, 1.2) for f in plate])
    out, _ = detail_match_frames(
        gen, full, [bbox] * gen.shape[0], edit_mask,
        grow_px=8, sharpen=1.0, max_gain=2.5, grain=0.0, seed=0,
    )
    before = _level_ratios(plate[0], gen[0], edit_mask[0])
    after = _level_ratios(plate[0], out[0], edit_mask[0])
    for k in range(3):
        assert after[k] > before[k] + 0.01 or before[k] > 0.97, f"level {k} did not improve"
        assert after[k] <= 1.05, f"level {k} overshot to {after[k]:.2f}x"


def test_detail_match_contrast_only_gains_near_one():
    full, cropped, edit_mask, bbox = _crop_scene(n=3)
    plate = cropped.clone()
    gen = plate.clone()
    sel = edit_mask[0] > 0.5
    gen[:, sel] = plate[:, sel] * 0.8 + 0.1

    out, report = detail_match_frames(
        gen, full, [bbox] * 3, edit_mask,
        grow_px=8, sharpen=1.0, max_gain=2.5, grain=0.0, seed=0,
    )
    for frame in report["frames"]:
        for g in frame["gains"]:
            assert abs(g - 1.0) < 0.15


def test_detail_match_noop_on_identical():
    full, cropped, edit_mask, bbox = _crop_scene(n=2)
    out, _ = detail_match_frames(
        cropped, full, [bbox] * 2, edit_mask,
        grow_px=8, sharpen=0.7, max_gain=2.5, grain=1.0, seed=42,
    )
    assert (out - cropped).abs().max().item() < 1e-5


def test_detail_match_grain_brings_nlf_closer():
    full, cropped, edit_mask, bbox = _crop_scene(n=3, seed=7)
    plate = cropped.clone()
    g = torch.Generator().manual_seed(99)
    noise = torch.randn(*plate.shape, generator=g) * 0.02
    plate = (plate + noise).clamp(0, 1)
    for i in range(plate.shape[0]):
        full[i, 16:112, 16:112] = plate[i]
    # The VAE also strips grain from the WHOLE crop: the generated crop is
    # the clean texture everywhere (ring included), with new content in the
    # edit. The deficit is measured plate-ring vs generated-ring and applied
    # inside the edit.
    gen = cropped.clone()
    gen[:, edit_mask[0] > 0.5] = _band_limited_texture(96, 96, seed=1)[edit_mask[0] > 0.5]

    def _nlf(img, mask):
        med = F.avg_pool2d(img.permute(0, 3, 1, 2), 3, stride=1, padding=1)
        med = med.permute(0, 2, 3, 1)
        r = (img - med).float()
        sel = (mask > 0.5).unsqueeze(-1).expand_as(r)
        return float(r[sel].std().item())

    before = _nlf(gen, edit_mask)
    out, _ = detail_match_frames(
        gen, full, [bbox] * 3, edit_mask,
        grow_px=8, sharpen=0.0, max_gain=2.5, grain=1.0, seed=123,
    )
    after = _nlf(out, edit_mask)
    target = _nlf(plate, edit_mask)
    assert abs(after - target) < abs(before - target) * 0.8 + 1e-6


def test_detail_match_deterministic_seed():
    full, cropped, edit_mask, bbox = _crop_scene()
    gen = cropped.clone()
    sel = edit_mask[0] > 0.5
    gen[:, sel] = _gaussian_blur_hwc(cropped[0], 1.0)[sel]
    a, _ = detail_match_frames(gen, full, [bbox] * 5, edit_mask, 8, 0.5, 2.5, 1.0, 77)
    b, _ = detail_match_frames(gen, full, [bbox] * 5, edit_mask, 8, 0.5, 2.5, 1.0, 77)
    assert torch.allclose(a, b)


# ── N4 ReferenceColorMatch ──────────────────────────────────────────────────


def _skin_reference(h: int = 64, w: int = 64) -> torch.Tensor:
    lin = torch.tensor([0.55, 0.38, 0.32])
    rgb = linear_to_srgb(lin).view(1, 1, 3).expand(h, w, 3)
    return rgb.clamp(0, 1)


def test_reference_color_match_removes_red_drift():
    h, w, n = 64, 64, 6
    ref = _skin_reference(h, w)
    images = ref.unsqueeze(0).expand(n, -1, -1, -1).clone()
    yy, xx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    edit = ((xx - w // 2).float() ** 2 + (yy - h // 2).float() ** 2 < (w * 0.3) ** 2).float()
    edit_mask = edit.unsqueeze(0).expand(n, -1, -1)

    for t in range(n):
        lin = srgb_to_linear(images[t, ..., :3])
        lab = linear_srgb_to_oklab(lin)
        shift = 0.03 * (t + 1) / n
        lab[..., 1] = lab[..., 1] + shift * edit   # the a channel is [H, W]
        images[t, ..., :3] = linear_to_srgb(oklab_to_linear_srgb(lab))

    out, report = reference_color_match_frames(
        images, edit_mask, ref, None, 1.0, "mean_std", False, 5,
    )
    for t in range(n):
        lin = srgb_to_linear(out[t, ..., :3])
        lab = linear_srgb_to_oklab(lin)
        sel = edit > 0.5
        ref_lab = linear_srgb_to_oklab(srgb_to_linear(ref[..., :3]))
        err = (lab[..., 1][sel] - ref_lab[..., 1][sel]).abs().mean().item()
        assert err < 0.005
    assert "red shift" in report["frames"][0]["red_note"].lower()


def test_reference_color_match_outside_mask_unchanged():
    h, w, n = 48, 48, 3
    ref = _skin_reference(h, w)
    images = _band_limited_texture(h, w, seed=2).unsqueeze(0).expand(n, -1, -1, -1).clone()
    edit = torch.zeros(h, w)
    edit[10:30, 10:30] = 1.0
    edit_mask = edit.unsqueeze(0).expand(n, -1, -1)
    outside = edit < 0.5
    before = images.clone()
    out, _ = reference_color_match_frames(images, edit_mask, ref, None, 1.0, "mean_std", False, 3)
    for t in range(n):
        assert torch.allclose(out[t, outside], before[t, outside], atol=1e-6)


def test_reference_color_match_lightness_untouched():
    h, w = 48, 48
    ref = _skin_reference(h, w)
    images = ref.unsqueeze(0).clone()
    edit = torch.ones(h, w)
    edit_mask = edit.unsqueeze(0)
    lin = srgb_to_linear(images[0, ..., :3])
    lab = linear_srgb_to_oklab(lin)
    lab[..., 1] += 0.04
    images[0, ..., :3] = linear_to_srgb(oklab_to_linear_srgb(lab))
    l_before = linear_srgb_to_oklab(srgb_to_linear(images[0, ..., :3]))[..., 0]
    out, _ = reference_color_match_frames(images, edit_mask, ref, None, 1.0, "mean_std", False, 1)
    l_after = linear_srgb_to_oklab(srgb_to_linear(out[0, ..., :3]))[..., 0]
    assert (l_after - l_before).abs().max().item() < 1e-4


def test_reference_color_match_noop_when_already_matches():
    ref = _skin_reference(32, 32)
    images = ref.unsqueeze(0)
    edit_mask = torch.ones(1, 32, 32)
    out, _ = reference_color_match_frames(images, edit_mask, ref, None, 1.0, "mean_std", False, 1)
    assert (out - images).abs().max().item() < 1e-5


# ── N5 PixelRepair ──────────────────────────────────────────────────────────


def test_pixel_repair_noop_on_moving_band_limited_clip():
    # A camera pan: a window sliding across a LARGER plate. The first version
    # used torch.roll, which wraps - it manufactured a hard colour seam and a
    # small rectangle of foreign colour in the corner of every frame, which is
    # exactly what a blotch is, so the "clean" clip was not clean. Real
    # panning reveals new plate at one edge; it does not wrap.
    h, w, n = 64, 64, 8
    tex = _band_limited_texture(h + n, w + 2 * n, seed=4)
    frames = [tex[t:t + h, 2 * t:2 * t + w] for t in range(n)]
    clip = torch.stack(frames, dim=0)
    out, _, _ = repair_frames(clip, None, 0.5, 0.5, False)
    assert (out - clip).abs().max().item() < 1e-4


def test_pixel_repair_salt_noise():
    h, w, n = 64, 64, 5
    truth = _band_limited_texture(h, w, seed=8).unsqueeze(0).expand(n, -1, -1, -1).clone()
    clip = truth.clone()
    g = torch.Generator().manual_seed(1)
    n_bad = int(0.005 * h * w)
    salt_yy, salt_xx = [], []
    for t in range(n):
        idx = torch.randperm(h * w, generator=g)[:n_bad]
        yy = idx // w
        xx = idx % w
        salt_yy.append(yy)
        salt_xx.append(xx)
        clip[t, yy, xx, :] = torch.rand(n_bad, 3, generator=g)

    out, dmg, _ = repair_frames(clip, None, 0.5, 0.0, False)
    repaired = 0
    flagged = 0
    for t in range(n):
        yy, xx = salt_yy[t], salt_xx[t]
        flagged += int((dmg[t, yy, xx] > 0.5).sum().item())
        err = (out[t, yy, xx, :] - truth[t, yy, xx, :]).abs().max(dim=-1).values
        repaired += int((err < 0.05).sum().item())
    assert repaired >= int(0.9 * n_bad * n)
    assert flagged >= int(0.9 * n_bad * n)
    total_dmg = int((dmg > 0.5).sum().item())
    assert total_dmg < 0.001 * h * w * n + n_bad * n


def test_pixel_repair_red_blotch():
    h, w = 80, 80
    base = _band_limited_texture(h, w, seed=5)
    lin = srgb_to_linear(base)
    lab = linear_srgb_to_oklab(lin)
    yy, xx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    cy, cx = h // 2, w // 2
    disc = ((xx - cx).float() ** 2 + (yy - cy).float() ** 2) < 6 ** 2
    surround_a = lab[..., 1][~disc].mean()
    lab[..., 1] = lab[..., 1] + 0.05 * disc.float()
    rgb = linear_to_srgb(oklab_to_linear_srgb(lab))
    clip = rgb.unsqueeze(0)
    excess_before = float((lab[..., 1][disc].mean() - surround_a).abs().item())
    out, _, _ = repair_frames(clip, None, 0.0, 0.8, False)
    lab_out = linear_srgb_to_oklab(srgb_to_linear(out[0, ..., :3]))
    excess_after = float((lab_out[..., 1][disc].mean() - surround_a).abs().item())
    assert excess_after < excess_before * 0.3
    l_out = lab_out[..., 0]
    l_in = linear_srgb_to_oklab(srgb_to_linear(clip[0, ..., :3]))[..., 0]
    assert (l_out - l_in).abs().max().item() < 1e-4


def test_pixel_repair_damage_mask_marks_salt():
    h, w = 48, 48
    truth = _band_limited_texture(h, w, seed=9).unsqueeze(0)
    clip = truth.clone()
    yy, xx = torch.tensor([10]), torch.tensor([20])
    clip[0, yy, xx, :] = 1.0
    out, dmg, _ = repair_frames(clip, None, 0.5, 0.0, False)
    assert dmg[0, yy, xx].max().item() > 0.5
