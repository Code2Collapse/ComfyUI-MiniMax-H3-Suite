# Apache-2.0 — ComfyUI-CustomNodePacks
# PORTED FROM: ComfyUI-CustomNodePacks :: nodes/_control_backends.py @ HEAD
#
# L9.39 (MMX row 5): the model backends used to be stubs that always raised. They now run CustomNodePacks' real
# depth runners when that pack is installed (runtime integration - no model code copied here), and the temporal
# window that the node always advertised now does something: per-frame scale / shift alignment of relative depth.

from __future__ import annotations

import importlib
import sys

import torch
import torch.nn.functional as F


class DepthBackendUnavailableError(RuntimeError):
    """Named error when a weight-gated depth backend cannot run."""


# backend name -> (CNP runner, kwargs, what it is)
MODEL_BACKENDS = {
    "depth_anything_v2": ("run_hf_depth", {"version": "v2", "size": "large"}, "Depth Anything V2 (per frame)"),
    "depth_anything_3": ("run_da3", {"size": "large"}, "Depth Anything 3 (per frame)"),
    "depth_pro": ("run_depth_pro", {}, "Apple Depth Pro (per frame, metric)"),
    "depthcrafter": ("run_depthcrafter", {}, "DepthCrafter (video depth)"),
    "dvd": ("run_dvd", {}, "DVD (video depth on Wan 2.1, CC BY-NC weights)"),
}
BACKENDS = ["passthrough"] + list(MODEL_BACKENDS)


def _to_bhwc(image: torch.Tensor) -> torch.Tensor:
    if image.ndim == 3:
        image = image.unsqueeze(0)
    return image


def passthrough_depth_plate(image: torch.Tensor, passthrough_hints: torch.Tensor | None) -> torch.Tensor:
    """External depth hints when wired; otherwise a LUMINANCE proxy (brightness, not depth - the report says so)."""
    if passthrough_hints is not None:
        hints = passthrough_hints
        if hints.ndim == 3:
            hints = hints.unsqueeze(0)
        return hints[..., :3] if hints.shape[-1] >= 3 else hints.repeat(1, 1, 1, 3)
    img = _to_bhwc(image)
    gray = img[..., :3].mean(dim=-1, keepdim=True)
    return gray.repeat(1, 1, 1, 3)


def _cnp_runners():
    """CustomNodePacks' `nodes._control_backends` when that pack is loaded in this ComfyUI, else None."""
    for name, mod in list(sys.modules.items()):
        f = (getattr(mod, "__file__", "") or "").replace("\\", "/")
        if f.endswith("ComfyUI-CustomNodePacks/__init__.py"):
            try:
                return importlib.import_module(name + ".nodes._control_backends")
            except Exception:  # noqa: BLE001
                return None
    return None


def _unavailable(backend: str, why: str) -> DepthBackendUnavailableError:
    label = MODEL_BACKENDS[backend][2]
    return DepthBackendUnavailableError(
        f"{label} is not available here: {why}. Either install / fix it, or run any depth node (for example core's "
        "'Run Depth Anything 3') and wire its depth image into passthrough_hints with backend=passthrough.")


def run_model_backend(backend: str, image_bhwc: torch.Tensor, runners=None) -> tuple[torch.Tensor, str]:
    runners = runners if runners is not None else _cnp_runners()
    if runners is None:
        extra = " (DVD also needs CustomNodePacks' third_party/DVD clone + weights)" if backend == "dvd" else ""
        raise _unavailable(backend, "it runs through ComfyUI-CustomNodePacks, which is not installed / loaded"
                           + extra)
    fn_name, kwargs, _label = MODEL_BACKENDS[backend]
    fn = getattr(runners, fn_name, None)
    if fn is None:
        raise _unavailable(backend, f"this CustomNodePacks version has no {fn_name}")
    out, note = fn(_to_bhwc(image_bhwc).float().cpu(), **kwargs)
    if out is None:
        raise _unavailable(backend, note)
    out = out.to(image_bhwc.device, torch.float32)
    if out.shape[1:3] != image_bhwc.shape[1:3]:
        out = F.interpolate(out.permute(0, 3, 1, 2), size=tuple(image_bhwc.shape[1:3]), mode="bilinear",
                            align_corners=False).permute(0, 2, 3, 1)
    return out.clamp(0, 1), note


def run_depthcrafter(image_bhwc: torch.Tensor, **_kw) -> torch.Tensor:
    return run_model_backend("depthcrafter", image_bhwc)[0]


def run_dvd(image_bhwc: torch.Tensor, **_kw) -> torch.Tensor:
    return run_model_backend("dvd", image_bhwc)[0]


def temporal_align(depth_bhwc: torch.Tensor, window: int = 17) -> tuple[torch.Tensor, dict]:
    """Make per-frame RELATIVE depth consistent over time.

    Per-frame depth models (and per-frame 0..1 normalisation) give every frame its own scale and offset, which shows
    as flicker. Each frame is fitted (least squares, a * d + b) to the median of the previous `window` aligned frames,
    then the clip is normalised once (2nd..98th percentile), not frame by frame. window <= 1 = off.
    Returns (aligned [B,H,W,3], stats) with stats = mean |frame-to-frame change of the mean| before / after.
    """
    d = _to_bhwc(depth_bhwc)[..., 0].float()
    b = d.shape[0]
    flick = lambda x: float((x.mean(dim=(1, 2))[1:] - x.mean(dim=(1, 2))[:-1]).abs().mean()) if x.shape[0] > 1 else 0.0  # noqa: E731

    def clip_norm(x: torch.Tensor) -> torch.Tensor:
        flat = x.reshape(-1)
        k = max(1, flat.numel() // 2_000_000)                # quantile on a subsample for long clips
        lo, hi = torch.quantile(flat[::k], torch.tensor([0.02, 0.98], dtype=flat.dtype, device=flat.device))
        return ((x - lo) / (hi - lo).clamp_min(1e-6)).clamp(0, 1)

    # "before" on the same clip-wide normalisation as "after": the two numbers compare like for like (raw-unit before
    # vs stretched after made a low-contrast clip look like align added flicker it had only made visible, L9.39)
    before = flick(clip_norm(d))
    if window <= 1 or b < 2:
        return depth_bhwc, {"flicker_before": before, "flicker_after": before, "aligned": False}
    out = torch.empty_like(d)
    out[0] = d[0]
    for t in range(1, b):
        ref = out[max(0, t - int(window)):t].median(dim=0).values
        x, y = d[t].reshape(-1), ref.reshape(-1)
        xm, ym = x.mean(), y.mean()
        var = ((x - xm) ** 2).mean()
        a = ((x - xm) * (y - ym)).mean() / var if var > 1e-12 else x.new_tensor(1.0)
        out[t] = a * d[t] + (ym - a * xm)
    out = clip_norm(out)
    after = flick(out)
    return out.unsqueeze(-1).repeat(1, 1, 1, 3), {"flicker_before": before, "flicker_after": after, "aligned": True}


def normals_from_depth(depth_bhwc: torch.Tensor) -> torch.Tensor:
    """Sobel-from-depth Lambertian normals — no model, always works."""
    d = _to_bhwc(depth_bhwc)
    g = d[..., 0].unsqueeze(1)
    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=g.device).view(1, 1, 3, 3)
    ky = kx.transpose(2, 3).contiguous()
    gx = F.conv2d(F.pad(g, (1, 1, 1, 1), mode="replicate"), kx)
    gy = F.conv2d(F.pad(g, (1, 1, 1, 1), mode="replicate"), ky)
    nz = torch.ones_like(gx)
    n = torch.cat([-gx, -gy, nz], dim=1)
    n = n / (n.norm(dim=1, keepdim=True) + 1e-6)
    n = (n * 0.5 + 0.5).permute(0, 2, 3, 1)
    return n.clamp(0, 1).contiguous()


def run_depth_backend(
    backend: str,
    image: torch.Tensor,
    *,
    passthrough_hints: torch.Tensor | None = None,
    runners=None,
) -> tuple[torch.Tensor, str]:
    b = (backend or "passthrough").lower()
    if b == "passthrough":
        out = passthrough_depth_plate(image, passthrough_hints)
        note = ("passthrough: wired depth hints" if passthrough_hints is not None else
                "passthrough: LUMINANCE proxy (brightness, not depth) - wire a depth map into passthrough_hints "
                "or pick a depth backend")
        return out, note
    if b in MODEL_BACKENDS:
        return run_model_backend(b, image, runners)
    raise ValueError(f"Unknown depth backend {backend!r}")
