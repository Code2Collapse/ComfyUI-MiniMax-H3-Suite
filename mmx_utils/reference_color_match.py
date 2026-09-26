"""Reference colour match — hold the edit region to <Picture 1> in Oklab (N4)."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from .temporal import centred_window
from .color_space import linear_srgb_to_oklab, linear_to_srgb, oklab_to_linear_srgb, srgb_to_linear

_STD_CLAMP = (0.5, 2.0)
_HIST_BINS = 256


def _masked_stats(
    lab: torch.Tensor,
    mask: torch.Tensor,
    channels: tuple[int, ...],
) -> tuple[torch.Tensor, torch.Tensor]:
    sel = mask > 0.5
    if not sel.any():
        z = torch.zeros(len(channels), device=lab.device, dtype=lab.dtype)
        return z, torch.ones_like(z)
    vals = []
    for ch in channels:
        v = lab[..., ch][sel]
        vals.append(v)
    stacked = torch.stack(vals, dim=0)
    mean = stacked.mean(dim=1)
    std = stacked.std(dim=1, unbiased=False).clamp(min=1e-8)
    return mean, std


def _mean_std_correct(
    lab: torch.Tensor,
    ref_mean: torch.Tensor,
    ref_std: torch.Tensor,
    src_mean: torch.Tensor,
    src_std: torch.Tensor,
    channels: tuple[int, ...],
) -> torch.Tensor:
    out = lab.clone()
    for i, ch in enumerate(channels):
        ratio = (ref_std[i] / src_std[i]).clamp(_STD_CLAMP[0], _STD_CLAMP[1])
        out[..., ch] = (lab[..., ch] - src_mean[i]) * ratio + ref_mean[i]
    return out


def _cdf_match_channel(
    src: torch.Tensor,
    ref: torch.Tensor,
    src_mask: torch.Tensor,
    ref_mask: torch.Tensor,
    bins: int = _HIST_BINS,
) -> torch.Tensor:
    sel_s = src_mask > 0.5
    sel_r = ref_mask > 0.5
    if not sel_s.any() or not sel_r.any():
        return src
    vs = src[sel_s]
    vr = ref[sel_r]
    lo = min(float(vs.min()), float(vr.min()))
    hi = max(float(vs.max()), float(vr.max()))
    if hi - lo < 1e-8:
        return src
    edges = torch.linspace(lo, hi, bins + 1, device=src.device)

    def _cdf(vals, sel):
        hist = torch.histc(vals, bins=bins, min=lo, max=hi)
        hist = hist / hist.sum().clamp_min(1.0)
        return torch.cumsum(hist, dim=0)

    cdf_r = _cdf(vr, sel_r)
    cdf_s = _cdf(vs, sel_s)
    # Map source bin centres through ref CDF
    bin_idx = torch.clamp(
        ((src - lo) / (hi - lo + 1e-8) * bins).long(),
        0,
        bins - 1,
    )
    mapped = torch.zeros_like(src)
    for b in range(bins):
        target_cdf = cdf_s[b]
        # invert ref cdf at target
        idx = (cdf_r - target_cdf).abs().argmin()
        centre = lo + (idx.float() + 0.5) / bins * (hi - lo)
        mapped = torch.where(bin_idx == b, centre, mapped)
    return mapped


def _temporal_median_stack(stack: list[torch.Tensor], i: int, window: int) -> torch.Tensor:
    lo, hi = centred_window(i, len(stack), window)
    chunk = torch.stack(stack[lo:hi], dim=0)
    return chunk.median(dim=0).values


def reference_color_match_frames(
    images: torch.Tensor,
    edit_mask: torch.Tensor,
    reference: torch.Tensor,
    reference_mask: torch.Tensor | None,
    strength: float,
    mode: str,
    match_lightness: bool,
    temporal_smooth: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Match edit-region chroma to a single reference frame in Oklab."""
    n, h, w, _ = images.shape
    ref = reference[0] if reference.ndim == 4 else reference
    if ref.shape[0] != h or ref.shape[1] != w:
        ref_nchw = ref[..., :3].permute(2, 0, 1).unsqueeze(0).float()
        ref_nchw = F.interpolate(ref_nchw, size=(h, w), mode="bilinear", align_corners=False)
        ref = ref_nchw.squeeze(0).permute(1, 2, 0).to(reference.dtype)

    ref_mask = (
        torch.ones(h, w, device=images.device)
        if reference_mask is None
        else reference_mask[0].to(images.device).clamp(0, 1)
    )
    if ref_mask.shape[0] != h or ref_mask.shape[1] != w:
        ref_mask = F.interpolate(
            ref_mask.unsqueeze(0).unsqueeze(0).float(),
            size=(h, w),
            mode="bilinear",
            align_corners=False,
        )[0, 0]

    em = edit_mask
    if em.shape[0] == 1:
        em = em.expand(n, -1, -1)
    em = em[:n].to(images.device).clamp(0, 1)
    if em.shape[-2] != h or em.shape[-1] != w:
        em = F.interpolate(em.unsqueeze(1).float(), size=(h, w), mode="bilinear", align_corners=False)[:, 0]

    ref_lin = srgb_to_linear(ref[..., :3])
    ref_lab = linear_srgb_to_oklab(ref_lin)
    chans = (0, 1, 2) if match_lightness else (1, 2)
    ref_mean, ref_std = _masked_stats(ref_lab, ref_mask, chans)

    corrections: list[torch.Tensor] = []
    per_frame: list[dict[str, Any]] = []

    for i in range(n):
        frame = images[i]
        lin = srgb_to_linear(frame[..., :3])
        lab = linear_srgb_to_oklab(lin)
        edit_sel = em[i] > 0.5
        src_mean, src_std = _masked_stats(lab, em[i], chans)

        if mode == "histogram":
            corrected = lab.clone()
            for ch in chans:
                corrected[..., ch] = _cdf_match_channel(
                    lab[..., ch], ref_lab[..., ch], em[i], ref_mask, _HIST_BINS,
                )
        else:
            corrected = _mean_std_correct(lab, ref_mean, ref_std, src_mean, src_std, chans)

        delta = corrected - lab
        corrections.append(delta)

        a_shift = float(delta[..., 1][edit_sel].mean().item()) if edit_sel.any() else 0.0
        per_frame.append(
            {
                "frame": i,
                "oklab_a_shift_applied": round(a_shift, 5),
                "red_note": f"removed a red shift of {a_shift:+.4f} in Oklab a",
            }
        )

    out = images.clone()
    for i in range(n):
        delta_sm = _temporal_median_stack(corrections, i, temporal_smooth)
        # The weight IS the mask: PlateRestore's restore_mask says how much of
        # each pixel is generated, so correcting by exactly that fraction
        # corrects the generated part and nothing else.
        #
        # This used to blur the mask by 12 px (a feather CENTRED on the edge,
        # so subject pixels near their own boundary got half a correction -
        # a constant 17% of the red left behind on every frame) and then reset
        # everything outside the hard mask to the original, which chopped off
        # the outer half of that feather and left a hard edge anyway. A hard
        # input mask now gives a hard edge; soften it upstream, which is what
        # restore_mask already is.
        alpha = em[i].clamp(0.0, 1.0)
        lab = linear_srgb_to_oklab(srgb_to_linear(out[i, ..., :3]))
        lab_corr = lab + delta_sm * float(strength)
        rgb_lin = oklab_to_linear_srgb(lab_corr)
        rgb = linear_to_srgb(rgb_lin)
        m = alpha.unsqueeze(-1)
        blended = out[i, ..., :3].float() * (1.0 - m) + rgb.to(out.dtype).float() * m
        # Pure plate (alpha 0) stays bit-exact - not recomputed through the
        # colour round trip, which is exact only to float precision.
        blended = torch.where(m > 0, blended, images[i, ..., :3].float())
        if out.shape[-1] > 3:
            out[i] = torch.cat([blended.to(out.dtype), out[i, ..., 3:]], dim=-1)
        else:
            out[i] = blended.to(out.dtype)

    summary = (
        f"frames={n} mode={mode} strength={strength:g} "
        f"match_lightness={match_lightness} temporal_smooth={temporal_smooth}"
    )
    report: dict[str, Any] = {"frames": per_frame, "summary": summary}
    return out, report
