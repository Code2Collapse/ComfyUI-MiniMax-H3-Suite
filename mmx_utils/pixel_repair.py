"""Pixel repair — impulse snow and chroma blotches with motion guard (N5)."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from .color_space import (
    linear_srgb_to_oklab,
    linear_to_srgb,
    oklab_to_linear_srgb,
    srgb_to_linear,
)
from .feather_composite import gaussian_blur_mask

_MAD_EPS = 1e-8


def _median3x3_hw(x: torch.Tensor) -> torch.Tensor:
    """Spatial median on [H,W], reflect-padded.

    F.unfold's own `padding=1` pads with ZEROS. At a corner 5 of the 9 samples
    were then 0, so the "median" was 0; every corner and most edge pixels
    looked like extreme outliers, and on the first and last frames (spatial
    test only) they were "repaired" to that zero - black pixels written into
    the borders of a clean plate. Reflect padding keeps the border median made
    of real image values.
    """
    t = F.pad(x.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode="reflect")
    patches = F.unfold(t, 3).squeeze(0)
    return patches.median(dim=0).values.view(x.shape)


def _box_blur_hw(x: torch.Tensor, radius: int) -> torch.Tensor:
    if radius < 1:
        return x
    k = 2 * radius + 1
    t = x.unsqueeze(0).unsqueeze(0)
    # Kernels on the image's device and dtype: a CPU kernel against a CUDA
    # image is a crash, and ComfyUI hands this node CUDA tensors routinely.
    kh = torch.ones(1, 1, 1, k, device=x.device, dtype=t.dtype) / k
    kv = torch.ones(1, 1, k, 1, device=x.device, dtype=t.dtype) / k
    t = F.conv2d(F.pad(t, (radius, radius, 0, 0), mode="reflect"), kh)
    t = F.conv2d(F.pad(t, (0, 0, radius, radius), mode="reflect"), kv)
    return t.squeeze()


def _frame_mad(x: torch.Tensor) -> float:
    med = x.median()
    return float((x - med).abs().median().item()) + _MAD_EPS


def _neighbours8(img: torch.Tensor) -> torch.Tensor:
    """The 8 spatial neighbours of every pixel, reflect-padded -> [8, H, W]."""
    t = F.pad(img.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode="reflect")
    patches = F.unfold(t, 3).squeeze(0)                  # [9, H*W]
    keep = [0, 1, 2, 3, 5, 6, 7, 8]                      # drop the centre
    return patches[keep].view(8, *img.shape)


def _noise_sigma(img: torch.Tensor) -> float:
    """Robust per-frame noise scale: 1.4826 * MAD of the 3x3-median residual.

    The residual is full resolution; only the MAD - a robust scale, stable on
    a fraction of a frame's pixels - is taken on every third pixel each way.
    A median over 2 million values per channel per frame was a measurable
    share of the node's time for no change in the estimate.
    """
    resid = img - _median3x3_hw(img)
    if resid.numel() > 262144:
        resid = resid[::3, ::3]
    return 1.4826 * _frame_mad(resid)


# Absolute floors in display units (0-1). The noise-scaled threshold alone
# collapses to ~0 on a clean synthetic or a heavily denoised plate, where it
# would then flag every local extremum of smooth texture.
_IMPULSE_FLOOR = (0.12, 0.04)      # impulse = 0 -> 0.12, impulse = 1 -> 0.04
_IMPULSE_K = (10.0, 4.0)


def _impulse_mask_frame(
    frame: torch.Tensor,
    prev: torch.Tensor | None,
    nxt: torch.Tensor | None,
    impulse: float,
    k_spatial_extra: float = 0.0,
) -> torch.Tensor:
    """Extremum-excess spike detection for one frame [H,W,C] -> bool [H,W].

    A pixel is damaged only if it lies BEYOND EVERY neighbour - its 8 spatial
    neighbours, plus the same position in the previous and next frame when
    they exist - by a noise-scaled margin, in some channel.

    This replaced a "deviates from the spatial median AND the temporal median,
    and the temporal neighbours agree" test, which failed on the most ordinary
    real footage: panning over texture. Under steady motion a symmetric
    texture peak has its t-1 and t+1 samples equidistant on either side of it,
    so they AGREE, the guard passed, and a real feature was "repaired". Smooth
    structure never exceeds all of its neighbours by much; a spike does. This
    is the extremum form of the spike-detection indices used in film dust
    busting.

    Known limit, stated in the node tooltip: a genuine one-pixel feature that
    moves between frames (a star, a specular glint) is indistinguishable from a
    spike without motion compensation. Keep `impulse` low on such shots.
    """
    h, w, c = frame.shape
    s = float(impulse)
    k = _IMPULSE_K[0] + (_IMPULSE_K[1] - _IMPULSE_K[0]) * s + k_spatial_extra
    floor = _IMPULSE_FLOOR[0] + (_IMPULSE_FLOOR[1] - _IMPULSE_FLOOR[0]) * s
    damage = torch.zeros(h, w, dtype=torch.bool, device=frame.device)
    for ch in range(min(c, 3)):
        img = frame[..., ch].float()
        nb = [_neighbours8(img)]
        if prev is not None:
            nb.append(prev[..., ch].float().unsqueeze(0))
        if nxt is not None:
            nb.append(nxt[..., ch].float().unsqueeze(0))
        nb = torch.cat(nb, dim=0)
        thr = max(k * _noise_sigma(img), floor)
        above = img - nb.amax(dim=0)
        below = nb.amin(dim=0) - img
        damage |= (above > thr) | (below > thr)
    return damage


def _repair_impulse_frame(
    frame: torch.Tensor,
    prev: torch.Tensor | None,
    nxt: torch.Tensor | None,
    damage: torch.Tensor,
) -> torch.Tensor:
    """Replace damaged pixels with the median of their spatiotemporal
    neighbours (the centre excluded - it is the damage)."""
    out = frame.clone()
    if not damage.any():
        return out
    for ch in range(min(frame.shape[-1], 3)):
        img = out[..., ch].float()
        nb = [_neighbours8(img)]
        if prev is not None:
            nb.append(prev[..., ch].float().unsqueeze(0))
        if nxt is not None:
            nb.append(nxt[..., ch].float().unsqueeze(0))
        repl = torch.cat(nb, dim=0).median(dim=0).values
        out[..., ch] = torch.where(damage, repl, img).to(frame.dtype)
    return out


# Chroma distance in Oklab (a, b). A just-noticeable chroma difference is
# around 0.02; a cheek blush that reads as a red patch is several times that.
_BLOTCH_THRESH = (0.08, 0.015)     # blotch = 0 -> 0.08, blotch = 1 -> 0.015


def _masked_box(x: torch.Tensor, weight: torch.Tensor, radius: int) -> torch.Tensor:
    """Box mean of x over the pixels where weight is 1."""
    num = _box_blur_hw(x * weight, radius)
    den = _box_blur_hw(weight, radius).clamp_min(1e-6)
    return num / den


def _half_window_kernels(radius: int, device, dtype):
    """Four separable half-windows: left, right, up, down. Each excludes the
    centre row/column, so a pixel is compared only with one SIDE of itself."""
    k = 2 * radius + 1
    full = torch.ones(k, device=device, dtype=dtype)
    first = torch.zeros(k, device=device, dtype=dtype)
    first[:radius] = 1.0
    last = torch.zeros(k, device=device, dtype=dtype)
    last[radius + 1:] = 1.0
    # (horizontal taps, vertical taps)
    return [(first, full), (last, full), (full, first), (full, last)]


def _sep_conv(x: torch.Tensor, h_taps: torch.Tensor, v_taps: torch.Tensor) -> torch.Tensor:
    r = (h_taps.numel() - 1) // 2
    t = x.unsqueeze(0).unsqueeze(0)
    t = F.conv2d(F.pad(t, (r, r, 0, 0), mode="reflect"), h_taps.view(1, 1, 1, -1))
    t = F.conv2d(F.pad(t, (0, 0, r, r), mode="reflect"), v_taps.view(1, 1, -1, 1))
    return t[0, 0]


def _half_window_means(chans, weight: torch.Tensor, radius: int):
    """For each of the four half-windows, the weighted mean of every channel."""
    out = []
    for h_taps, v_taps in _half_window_kernels(radius, weight.device, weight.dtype):
        den = _sep_conv(weight, h_taps, v_taps).clamp_min(1e-6)
        out.append([_sep_conv(c * weight, h_taps, v_taps) / den for c in chans])
    return out


def _blotch_mask_and_repair(
    frame: torch.Tensor,
    blotch: float,
    short_edge: int,
    allowed: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Oklab chroma blotches -> (damage_mask, repaired_frame).

    A blotch changes COLOUR, not shape: its chroma departs from the local
    chroma while its lightness does not depart from the local lightness.

    Three things the first version got wrong, each of which kept it from
    working at all:
      - the threshold was a multiple of the frame's own chroma MAD, which on a
        near-neutral plate is tiny, so texture chroma noise read as blotches;
        the threshold is now an absolute Oklab distance;
      - the local estimate included the blotch itself, diluting its own
        residual below threshold; it is re-estimated with the first-pass
        candidates EXCLUDED;
      - `blotch` was both the sensitivity and the repair strength, so a
        detected blotch could never be more than half removed at 0.5. It is
        the sensitivity only: what is detected is removed.
    The fix is applied to a and b in Oklab and converted once; L is carried
    through untouched, so lightness cannot move by construction. The repair
    is confined to `allowed` - before, only the REPORTED mask was.
    """
    s = float(blotch)
    if s <= 0.0:
        return torch.zeros(frame.shape[:2], dtype=torch.bool, device=frame.device), frame
    lin = srgb_to_linear(frame[..., :3].float())
    lab_full = linear_srgb_to_oklab(lin)
    thr = _BLOTCH_THRESH[0] + (_BLOTCH_THRESH[1] - _BLOTCH_THRESH[0]) * s

    # Detection and the local estimates run at a reduced scale on large
    # frames. They are large-radius smooth fields by construction (radius
    # ~short_edge/30, 36 px at 1080p), so full resolution buys nothing - and
    # at 2 MP it was 80% of the node's time: ~60 separable box filters of 73
    # taps over every pixel, 2.9 s a frame on the CPU. The result is applied
    # at full resolution. Frames up to 1024 px on the short edge are not
    # scaled at all.
    scale = max(1, int(short_edge // 1024) + (1 if short_edge > 1024 else 0))
    if scale > 1:
        hw = lab_full.shape[:2]
        small = F.avg_pool2d(lab_full.permute(2, 0, 1).unsqueeze(0), scale, stride=scale,
                             ceil_mode=True)[0].permute(1, 2, 0)
        allowed_s = None
        if allowed is not None:
            allowed_s = F.max_pool2d(allowed.float()[None, None], scale, stride=scale,
                                     ceil_mode=True)[0, 0] > 0.5
        mask_s, a_loc_s, b_loc_s = _blotch_detect(small, thr, short_edge // scale, allowed_s)
        if not mask_s.any():
            return torch.zeros(hw, dtype=torch.bool, device=frame.device), frame
        up = lambda t, mode: F.interpolate(t[None, None].float(), size=hw, mode=mode,
                                           **({} if mode == "nearest" else {"align_corners": False}))[0, 0]
        mask = up(mask_s, "nearest") > 0.5
        a_loc = up(a_loc_s, "bilinear")
        b_loc = up(b_loc_s, "bilinear")
    else:
        mask, a_loc, b_loc = _blotch_detect(lab_full, thr, short_edge, allowed)
        if not mask.any():
            return mask, frame
    if allowed is not None:
        mask = mask & allowed
    l_ch, a_ch, b_ch = lab_full[..., 0], lab_full[..., 1], lab_full[..., 2]
    return mask, _apply_blotch_fix(frame, mask, l_ch, a_ch, b_ch, a_loc, b_loc, allowed)


def _blotch_detect(lab, thr, short_edge, allowed):
    """Candidates and the full-surround chroma estimate on one Oklab frame."""
    l_ch, a_ch, b_ch = lab[..., 0], lab[..., 1], lab[..., 2]
    radius = max(12, int(round(short_edge / 30.0)))

    def _candidates(weight: torch.Tensor):
        # A blotch differs from its surroundings on EVERY side; a colour edge
        # (a red shirt against a green wall, at similar lightness) matches the
        # side it belongs to. Comparing against the full surround alone
        # flagged such edges as blotches and bled colour across them - the
        # first test clip caught it at a wrap-around seam. So the residual is
        # the MINIMUM over four half-windows: an edge has one near-zero side,
        # a blotch has none.
        sides = _half_window_means([a_ch, b_ch, l_ch], weight, radius)
        r_c = None
        r_l = None
        for a_s, b_s, l_s in sides:
            rc = torch.sqrt((a_ch - a_s) ** 2 + (b_ch - b_s) ** 2)
            rl = (l_ch - l_s).abs()
            r_c = rc if r_c is None else torch.minimum(r_c, rc)
            r_l = rl if r_l is None else torch.minimum(r_l, rl)
        # The repair target is the full-surround estimate, which is what the
        # blotch should have looked like.
        a_loc = _masked_box(a_ch, weight, radius)
        b_loc = _masked_box(b_ch, weight, radius)
        return (r_c > thr) & (r_l < r_c), a_loc, b_loc

    cand, _, _ = _candidates(torch.ones_like(a_ch))
    cand, a_loc, b_loc = _candidates(1.0 - cand.float())
    if allowed is not None:
        cand = cand & allowed

    c = cand.float().unsqueeze(0).unsqueeze(0)
    # Opening with a 5x5 element: a blotch is compact in BOTH directions, so
    # anything thinner than 5 px one way - a coloured line, a cable, a seam -
    # is not one. And the erosion pads with ZEROS: max_pool2d ignores
    # out-of-image pixels, which made the frame edge behave like "inside" and
    # let a 2 px strip along the border survive the opening.
    eroded = -F.max_pool2d(F.pad(-c, (2, 2, 2, 2), value=0.0), 5, stride=1)
    opened = F.max_pool2d(eroded, 5, stride=1, padding=2)
    return opened[0, 0] > 0.5, a_loc, b_loc


def _apply_blotch_fix(frame, mask, l_ch, a_ch, b_ch, a_loc, b_loc, allowed):
    """Pull a/b toward the local estimate inside the grown, feathered mask."""
    # Grow before feathering so the feather lands OUTSIDE the blotch, not
    # across it - otherwise its rim is only half corrected.
    grown = F.max_pool2d(mask.float().unsqueeze(0).unsqueeze(0), 5, stride=1, padding=2)
    feather = gaussian_blur_mask(grown, 3)[0, 0].clamp(0, 1)
    if allowed is not None:
        feather = feather * allowed.float()
    a_fix = a_ch + (a_loc - a_ch) * feather
    b_fix = b_ch + (b_loc - b_ch) * feather
    rgb_lin = oklab_to_linear_srgb(torch.stack([l_ch, a_fix, b_fix], dim=-1))
    fixed = linear_to_srgb(rgb_lin).to(frame.dtype)
    touched = (feather > 0).unsqueeze(-1)
    out = frame.clone()
    out[..., :3] = torch.where(touched, fixed, frame[..., :3])
    return out


def repair_frames(
    images: torch.Tensor,
    mask: torch.Tensor | None,
    impulse: float,
    blotch: float,
    debug: bool,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Repair impulse snow and chroma blotches; return (images, damage_mask, report)."""
    n, h, w, _ = images.shape
    out_dev = images.device
    out_dtype = images.dtype
    streaming = device is not None and device != out_dev
    cd = device if streaming else out_dev
    restrict = mask
    if restrict is not None:
        if restrict.shape[0] == 1:
            restrict = restrict.expand(n, -1, -1)
        restrict = restrict[:n].to(out_dev)

    out = images.clone()
    damage_full = torch.zeros(n, h, w, dtype=torch.float32, device=out_dev)
    per_frame: list[dict[str, Any]] = []
    short_edge = min(h, w)

    for i in range(n):
        frame = out[i]
        if restrict is not None:
            allowed = restrict[i] > 0.5
        else:
            allowed = torch.ones(h, w, dtype=torch.bool, device=out_dev)

        prev = out[i - 1] if i > 0 else None
        nxt = out[i + 1] if i < n - 1 else None
        if streaming:
            frame = frame.to(cd)
            prev = prev.to(cd) if prev is not None else None
            nxt = nxt.to(cd) if nxt is not None else None
            allowed = allowed.to(cd)
        k_extra = 2.0 if (prev is None or nxt is None) else 0.0

        impulse_dmg = _impulse_mask_frame(frame, prev, nxt, impulse, k_extra) & allowed
        frame = _repair_impulse_frame(frame, prev, nxt, impulse_dmg)

        blotch_dmg, frame = _blotch_mask_and_repair(frame, blotch, short_edge, allowed)

        combined = impulse_dmg | blotch_dmg
        if streaming:
            damage_full[i] = combined.float().to(out_dev)
            out[i] = frame.to(out_dev, out_dtype)
        else:
            damage_full[i] = combined.float()
            out[i] = frame

        per_frame.append(
            {
                "frame": i,
                "impulse_pixels": int(impulse_dmg.sum().item()),
                "blotch_pixels": int(blotch_dmg.sum().item()),
            }
        )

    summary = (
        f"frames={n} impulse={impulse:g} blotch={blotch:g} "
        f"total_damage_pixels={int(damage_full.sum().item())}"
    )
    report: dict[str, Any] = {"frames": per_frame, "summary": summary, "debug": debug}
    return out, damage_full, report
