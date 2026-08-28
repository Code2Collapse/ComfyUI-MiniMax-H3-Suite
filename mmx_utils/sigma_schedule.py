"""H3 sigma schedule math — video/audio shift pair and scheduler validation."""

from __future__ import annotations

import json
import math

import torch

TRAINED_SHIFT_VIDEO = 12.0
TRAINED_SHIFT_AUDIO = 3.0
TRAINED_SHIFT_RATIO = TRAINED_SHIFT_VIDEO / TRAINED_SHIFT_AUDIO


def time_shift_sigma(sigma: float | torch.Tensor, from_shift: float, to_shift: float):
    """Map sigma through from_shift grid, re-apply to_shift (core model.py)."""
    if isinstance(sigma, torch.Tensor):
        base = sigma / (from_shift + sigma * (1.0 - from_shift))
        return to_shift * base / (1.0 + (to_shift - 1.0) * base)
    base = float(sigma) / (from_shift + float(sigma) * (1.0 - from_shift))
    return to_shift * base / (1.0 + (to_shift - 1.0) * base)


def apply_ratio_lock(
    shift_video: float,
    shift_audio: float,
    *,
    ratio_lock: bool,
    ratio: float = TRAINED_SHIFT_RATIO,
) -> tuple[float, float, str]:
    """When ratio_lock, derive shift_audio from shift_video / ratio."""
    note = ""
    sv = float(shift_video)
    sa = float(shift_audio)
    r = float(ratio) if ratio > 0 else TRAINED_SHIFT_RATIO
    if ratio_lock:
        expected = sv / r
        if not math.isclose(sa, expected, rel_tol=0.0, abs_tol=1e-4):
            note = f"ratio_lock: shift_audio {sa:g} -> {expected:g} (video/ratio {r:g})"
            sa = expected
    return sv, sa, note


def validate_h3_sigmas(
    sigmas: torch.Tensor,
    *,
    scheduler: str = "simple",
    denoise: float = 1.0,
) -> tuple[bool, str]:
    """Legal scheduler wrapper checks for H3 sampling."""
    lines: list[str] = []
    if not torch.is_tensor(sigmas):
        return False, "sigmas must be a torch.Tensor"
    if sigmas.ndim != 1 or len(sigmas) < 2:
        return False, "sigmas must be 1-D with at least 2 values"
    if not bool(torch.isfinite(sigmas).all()):
        return False, "sigmas must be finite"
    if bool(torch.any(sigmas < -1e-4)) or bool(torch.any(sigmas > 1.0 + 1e-4)):
        return False, f"sigmas must stay in [0, 1]; got min={float(sigmas.min()):.4f} max={float(sigmas.max()):.4f}"
    if not bool(torch.all(sigmas[1:] < sigmas[:-1])):
        return False, "sigmas must be strictly decreasing"
    if not bool(torch.isclose(sigmas[0], sigmas.new_tensor(1.0), atol=1e-3, rtol=0.0)):
        lines.append(f"WARNING: first sigma is {float(sigmas[0]):.4f}, expected ~1.0 for full denoise")
    if not bool(torch.isclose(sigmas[-1], sigmas.new_tensor(0.0), atol=1e-5, rtol=0.0)):
        lines.append(f"WARNING: last sigma is {float(sigmas[-1]):.6f}, expected 0.0")

    sched = (scheduler or "simple").lower()
    if sched not in ("simple", "sgm_uniform", "karras", "exponential", "beta"):
        lines.append(f"WARNING: scheduler '{scheduler}' is uncommon for H3; prefer 'simple'")
    if sched != "simple":
        lines.append(
            "OFF-CONTRACT: non-simple scheduler — run an audio A/B against simple before delivery."
        )
    if float(denoise) < 0.999:
        lines.append(
            f"partial denoise={denoise:.3f} — strength is BasicScheduler denoise, never SplitSigmas on H3"
        )
    if "split" in sched or "bong" in sched:
        return False, "SplitSigmas and bong_tangent are illegal on H3 — use BasicScheduler denoise"
    ok = True
    report = "H3 sigma schedule OK\n" + "\n".join(lines) if lines else "H3 sigma schedule OK"
    return ok, report


def validate_turbo_step_contract(
    sigmas: torch.Tensor,
    contract: tuple[int, int, int],
) -> tuple[bool, list[str]]:
    """Enforce turbo step contract stashed by MiniMaxH3_TurboLoRA.

    contract = (target_steps, warn_above, refuse_above).
    """
    target, warn_above, refuse_above = (int(contract[0]), int(contract[1]), int(contract[2]))
    steps = int(len(sigmas)) - 1
    notes: list[str] = []
    ok = True
    if steps > refuse_above:
        ok = False
        notes.append(
            f"REFUSE: {steps} steps > turbo refuse limit {refuse_above} "
            f"(contract {contract})"
        )
    elif steps > warn_above:
        notes.append(
            f"WARNING: {steps} steps > turbo warn threshold {warn_above} "
            f"(target {target})"
        )
    elif steps != target:
        notes.append(
            f"NOTE: turbo target is {target} steps; schedule has {steps}"
        )
    return ok, notes


def format_sigma_inspector_table(
    sigmas: torch.Tensor,
    *,
    shift_video: float = TRAINED_SHIFT_VIDEO,
    shift_audio: float = TRAINED_SHIFT_AUDIO,
) -> str:
    """P2 backend: markdown table of σv, σa, dσa/dσv per step."""
    sv = float(shift_video)
    sa = float(shift_audio)
    rows = ["step | sigma_v | sigma_a | dsigma_a/dsigma_v | effective_denoise_v"]
    for i, sig_v in enumerate(sigmas.tolist()):
        sig_v = float(sig_v)
        sig_a = float(time_shift_sigma(sig_v, sv, sa))
        if i + 1 < len(sigmas):
            nxt_v = float(sigmas[i + 1].item())
            nxt_a = float(time_shift_sigma(nxt_v, sv, sa))
            ds = (nxt_a - sig_a) / (nxt_v - sig_v) if abs(nxt_v - sig_v) > 1e-12 else 0.0
        else:
            ds = 0.0
        eff = 1.0 - sig_v
        rows.append(f"{i:4d} | {sig_v:7.5f} | {sig_a:7.5f} | {ds:16.5f} | {eff:.5f}")
    return "\n".join(rows)


def build_sigma_inspector_json(
    sigmas: torch.Tensor,
    *,
    shift_video: float = TRAINED_SHIFT_VIDEO,
    shift_audio: float = TRAINED_SHIFT_AUDIO,
) -> str:
    """JSON payload for the sigma plot widget — same math as format_sigma_inspector_table."""
    sv = float(shift_video)
    sa = float(shift_audio)
    steps: list[dict[str, float | int]] = []
    sig_list = [float(s) for s in sigmas.tolist()]
    for i, sig_v in enumerate(sig_list):
        sig_a = float(time_shift_sigma(sig_v, sv, sa))
        if i + 1 < len(sig_list):
            nxt_v = sig_list[i + 1]
            nxt_a = float(time_shift_sigma(nxt_v, sv, sa))
            ds = (nxt_a - sig_a) / (nxt_v - sig_v) if abs(nxt_v - sig_v) > 1e-12 else 0.0
        else:
            ds = 0.0
        steps.append(
            {
                "step": i,
                "sigma_v": sig_v,
                "sigma_a": sig_a,
                "dsigma_a_dsigma_v": float(ds),
                "effective_denoise_v": float(1.0 - sig_v),
            }
        )
    payload = {
        "shift_video": sv,
        "shift_audio": sa,
        "sigmas_v": sig_list,
        "steps": steps,
    }
    return json.dumps(payload, separators=(",", ":"))
