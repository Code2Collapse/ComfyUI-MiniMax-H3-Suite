# MIT License — nuke-nodes-comfyui / ComfyUI-MiniMaxSuite
# PORTED FROM: nuke-nodes-comfyui :: colorspace_nodes.py (round-trip concept) @ HEAD

"""Colour round-trip QC — HDR probe and tolerance reporting."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .ocio_bridge import roundtrip_plane

HDR_PROBE_VALUES: tuple[float, ...] = (0.0, 0.5, 1.0, 4.0, 16.0)


@dataclass(frozen=True)
class RoundTripResult:
    passed: bool
    max_error: float
    report: str
    error_map: np.ndarray
    hdr_preserved: bool


def build_hdr_probe_image(
    width: int = 5,
    height: int = 1,
    channels: int = 3,
) -> torch.Tensor:
    """1×H×W×C probe with [0, 0.5, 1, 4, 16] across the width (replicated per channel)."""
    w = max(width, len(HDR_PROBE_VALUES))
    row = np.array(HDR_PROBE_VALUES[:w], dtype=np.float32)
    if w > len(HDR_PROBE_VALUES):
        row = np.resize(row, w)
    img = np.tile(row.reshape(1, w, 1), (height, 1, channels))
    return torch.from_numpy(img).unsqueeze(0)


def _error_stats(original: np.ndarray, recovered: np.ndarray) -> tuple[float, np.ndarray, str]:
    diff = np.abs(original.astype(np.float64) - recovered.astype(np.float64))
    err_map = diff.max(axis=-1, keepdims=True).astype(np.float32)
    max_err = float(diff.max())
    idx = np.unravel_index(int(diff.argmax()), diff.shape)
    ch = idx[2] if len(idx) > 2 else 0
    ch_names = ("R", "G", "B")
    ch_name = ch_names[min(ch, 2)]
    return max_err, err_map, ch_name


def _hdr_probe_preserved(original: np.ndarray, recovered: np.ndarray) -> tuple[bool, str]:
    """Values that were >1.0 in the original must remain >1.0 after round trip."""
    o = original.astype(np.float64)
    r = recovered.astype(np.float64)
    mask = o > 1.0
    if not bool(mask.any()):
        return True, "no HDR probe samples >1.0"
    preserved = bool(np.all(r[mask] > 1.0))
    worst = float(np.min(r[mask])) if mask.any() else 0.0
    note = f"HDR probe: min recovered value where original>1 was {worst:.6g}"
    return preserved, note


def evaluate_roundtrip(
    original: np.ndarray,
    recovered: np.ndarray,
    *,
    tolerance: float,
    context: str = "",
) -> RoundTripResult:
    max_error, err_map, ch = _error_stats(original, recovered)
    hdr_ok, hdr_note = _hdr_probe_preserved(original, recovered)
    passed = max_error <= float(tolerance) and hdr_ok
    prefix = f"{context}\n" if context else ""
    if passed:
        report = (
            f"{prefix}round-trip OK — max_error={max_error:.6g} <= tolerance={tolerance:g}\n"
            f"{hdr_note}"
        )
    else:
        reasons = []
        if max_error > float(tolerance):
            reasons.append(
                f"round-trip error {max_error:.6g} exceeds tolerance {tolerance:g} at channel {ch}"
            )
        if not hdr_ok:
            reasons.append(f"HDR highlights were clamped or lost — {hdr_note}")
        report = prefix + "; ".join(reasons)
    return RoundTripResult(
        passed=passed,
        max_error=max_error,
        report=report,
        error_map=err_map,
        hdr_preserved=hdr_ok,
    )


def _detect_scene_linear_clamp(plane: np.ndarray, scene_colorspace: str) -> str | None:
    """Flag scene-linear buffers crushed to SDR (max <= 1.0) — the de-clamp anti-pattern."""
    label = scene_colorspace.lower()
    if "linear" not in label and scene_colorspace not in ("ACEScg", "ACES2065-1"):
        return None
    rgb = plane[..., :3] if plane.ndim == 3 else plane
    peak = float(np.max(rgb))
    if peak <= 1.0 + 1e-5:
        return (
            f"scene-linear input appears clamped to SDR (peak={peak:.6g} <= 1.0) "
            f"in {scene_colorspace!r} — HDR headroom missing"
        )
    return None


def run_display_roundtrip_qc(
    image: torch.Tensor,
    *,
    scene_colorspace: str,
    display: str,
    view: str,
    tolerance: float = 1e-4,
    config=None,
) -> RoundTripResult:
    if image.ndim == 3:
        image = image.unsqueeze(0)
    plane = image[0].detach().cpu().numpy()
    clamp_note = _detect_scene_linear_clamp(plane, scene_colorspace)
    recovered = roundtrip_plane(
        plane,
        scene_colorspace=scene_colorspace,
        display=display,
        view=view,
        config=config,
    )
    result = evaluate_roundtrip(plane, recovered, tolerance=tolerance)
    if clamp_note:
        return RoundTripResult(
            passed=False,
            max_error=result.max_error,
            report=clamp_note + f"; {result.report}",
            error_map=result.error_map,
            hdr_preserved=False,
        )
    return result


def clamp_for_vacuous_test(image: np.ndarray) -> np.ndarray:
    """Deliberately destroy HDR — QC must fail."""
    return np.clip(image, 0.0, 1.0).astype(np.float32)
