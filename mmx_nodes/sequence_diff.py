"""H3 Sequence Diff (owner A21 Revision 3 Part 9.9; ledger L9.74 slice 8).

Compares two versions of a sequence frame by frame - pixel difference, block SSIM or multi-scale SSIM on luma (the
"perceptual" proxy: no learned metric is downloaded, the report says so) - optionally inside a mask, optionally
aligned by their frame maps (frames compared where both show the same SOURCE frame number). Streams both inputs in
chunks: only the scores and a bounded set of heatmaps (the worst frames) stay in memory.

Outputs: difference heatmaps, a numeric report, the changed intervals (runs above the threshold, in A's frame
numbers) and a flicker graph of the score over time.

Original node, GPL-3.0. The SSIM constants follow mmx_utils/spectrum_h3/objective_media_bounded.py (block size 8,
C1 = 0.01^2, C2 = 0.03^2). Design: docs/evidence/A21/phase1/DESIGN_slices7_10.md in the workspace.
"""

from __future__ import annotations

import heapq
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from comfy_api.latest import io

_PKG = Path(__file__).resolve().parents[1]
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from mmx_utils.spectrum_h3.objective_media_bounded import _fast_ms_ssim  # noqa: E402
from mmx_vendor.c2c_core.timeline import ledger as L  # noqa: E402
from mmx_vendor.c2c_core.video import media as _media  # noqa: E402
from mmx_vendor.c2c_core.video import reader as _reader  # noqa: E402

from .media_window import C2C_VIDEO, is_handle  # noqa: E402

WHO = "H3 Sequence Diff"
METHODS = ("pixel", "ssim", "ms-ssim")
BLOCK = 8
C1, C2 = 0.01 ** 2, 0.03 ** 2


class _Source:
    """A C2C video handle or an IMAGE batch, read by positions."""

    def __init__(self, name: str, handle, images):
        if handle is not None:
            if not is_handle(handle):
                raise ValueError(f"{WHO}: '{name}' is not a C2C video handle.")
            self.handle, self.images = handle, None
            self.count = len(handle)
            self.numbers = self._handle_numbers(handle)
        elif images is not None:
            self.handle, self.images = None, images
            self.count = int(images.shape[0])
            self.numbers = list(range(self.count))
        else:
            raise ValueError(f"{WHO}: connect '{name}' (a C2C video) or '{name}_images' (an IMAGE batch).")

    @staticmethod
    def _handle_numbers(h) -> list[int]:
        nums = _media.source_numbers(h)
        return [int(nums[i]) if i >= 0 else -1 for i in h.selected_indices()]

    def read(self, positions: list[int]) -> torch.Tensor:
        if self.handle is not None:
            return torch.from_numpy(_reader.read(self.handle, positions, fmt="float32")[..., :3].copy())
        return self.images[positions, ..., :3].float().cpu()


def _numbers_from_map(raw: str, count: int) -> list[int] | None:
    s = str(raw or "").strip()
    if not s:
        return None
    led = json.loads(s)
    if led.get("c2c_contract") != "h3.frame_ledger":
        raise ValueError(f"{WHO}: a frame map is not an h3.frame_ledger contract.")
    src = L.expand(led)
    if len(src) != count:
        raise ValueError(f"{WHO}: a frame map covers {len(src)} frames but its input has {count}.")
    return [L.number_of(led, i) if i >= 0 else -1 for i in src]


def _luma(x: torch.Tensor) -> torch.Tensor:                 # [N, H, W, 3] -> [N, 1, H, W]
    return (x[..., 0] * 0.2126 + x[..., 1] * 0.7152 + x[..., 2] * 0.0722)[:, None]


def _ssim_map(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Block SSIM per 8x8 block: [N, 1, H/8, W/8]."""
    b = max(1, min(BLOCK, int(x.shape[-2]), int(x.shape[-1])))
    mx, my = F.avg_pool2d(x, b, b, ceil_mode=True), F.avg_pool2d(y, b, b, ceil_mode=True)
    vx = (F.avg_pool2d(x * x, b, b, ceil_mode=True) - mx * mx).clamp_min(0)
    vy = (F.avg_pool2d(y * y, b, b, ceil_mode=True) - my * my).clamp_min(0)
    cov = F.avg_pool2d(x * y, b, b, ceil_mode=True) - mx * my
    return ((2 * mx * my + C1) / (mx * mx + my * my + C1)) * ((2 * cov + C2) / (vx + vy + C2))


def _heat(v: torch.Tensor) -> torch.Tensor:
    """[H, W] in 0..1 -> [H, W, 3] black -> red -> yellow -> white."""
    t = v.clamp(0, 1) * 3.0
    return torch.stack([t.clamp(0, 1), (t - 1).clamp(0, 1), (t - 2).clamp(0, 1)], dim=-1)


def compare_chunk(a: torch.Tensor, b: torch.Tensor, method: str, mask: torch.Tensor | None):
    """(difference scores [N], heat maps [N, H, W] in 0..1). Difference: pixel = mean |a - b|, ssim = 1 - SSIM,
    ms-ssim = 1 - MS-SSIM. ``mask`` [N or 1, H, W]: 1 = compare."""
    if b.shape[1:3] != a.shape[1:3]:
        b = F.interpolate(b.movedim(-1, 1), size=a.shape[1:3], mode="bilinear", align_corners=False).movedim(1, -1)
    m = None
    if mask is not None:
        m = mask.float()
        if m.shape[-2:] != a.shape[1:3]:
            m = F.interpolate(m[:, None], size=a.shape[1:3], mode="nearest")[:, 0]
        m = m.expand(a.shape[0], -1, -1) if m.shape[0] == 1 else m
    if method == "pixel":
        d = (a - b).abs().mean(-1)
        if m is not None:
            score = (d * m).sum((1, 2)) / m.sum((1, 2)).clamp_min(1e-6)
            d = d * m
        else:
            score = d.mean((1, 2))
        return score, (d * 4.0).clamp(0, 1)
    la, lb = _luma(a), _luma(b)
    if m is not None:                                       # outside the mask both sides are equal: no difference
        la, lb = la * m[:, None], lb * m[:, None]
    if method == "ssim":
        smap = _ssim_map(la, lb)
        if m is not None:
            mb = F.avg_pool2d(m[:, None], max(1, min(BLOCK, *a.shape[1:3])), ceil_mode=True)
            score = 1.0 - (smap * mb).sum((1, 2, 3)) / mb.sum((1, 2, 3)).clamp_min(1e-6)
        else:
            score = 1.0 - smap.mean((1, 2, 3))
        heat = F.interpolate(1.0 - smap, size=a.shape[1:3], mode="nearest")[:, 0]
        return score, (heat * 2.0).clamp(0, 1)
    score = 1.0 - _fast_ms_ssim(la, lb)
    d = (a - b).abs().mean(-1) * (m if m is not None else 1.0)
    return score, (d * 4.0).clamp(0, 1)


def flicker_graph(scores: list[float], threshold: float, width: int = 0, height: int = 160) -> torch.Tensor:
    """[1, H, W, 3]: the score over frames (white), the threshold (red), drawn with numpy."""
    n = max(1, len(scores))
    w = width or int(min(1024, max(256, n * 4)))
    img = np.full((height, w, 3), 0.08, np.float32)
    top = max(max(scores) if scores else 0.0, threshold * 1.5, 1e-6)
    ty = height - 1 - int(round(min(1.0, threshold / top) * (height - 1)))
    img[ty, :, :] = (0.85, 0.2, 0.2)
    ys = [height - 1 - int(round(min(1.0, s / top) * (height - 1))) for s in scores]
    xs = [int(round(i * (w - 1) / max(1, n - 1))) for i in range(n)]
    for (x0, y0), (x1, y1) in zip(zip(xs, ys), zip(xs[1:], ys[1:])):
        steps = max(abs(x1 - x0), abs(y1 - y0), 1)
        for k in range(steps + 1):
            img[y0 + (y1 - y0) * k // steps, x0 + (x1 - x0) * k // steps] = (0.95, 0.95, 0.95)
    for x, y, v in zip(xs, ys, scores):
        if v > threshold:
            img[max(0, y - 1):y + 2, max(0, x - 1):x + 2] = (1.0, 0.85, 0.2)     # a changed frame: a yellow dot
    return torch.from_numpy(img)[None]


def intervals(numbers: list[int], flags: list[bool]) -> list[list[int]]:
    """Runs of consecutive compared frames above the threshold, as [first, last] frame numbers."""
    out: list[list[int]] = []
    inside = False
    for num, flag in zip(numbers, flags):
        if flag and inside:
            out[-1][1] = num
        elif flag:
            out.append([num, num])
        inside = flag
    return out


class MiniMaxH3_SequenceDiff(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MiniMaxH3_SequenceDiff",
            display_name="H3 Sequence Diff",
            category="MiniMax H3/Media",
            description=(
                "Compares two versions frame by frame (pixel, SSIM or multi-scale SSIM on luma), inside a mask if "
                "given, aligned by their frame maps if given. Streams both in chunks. Outputs heatmaps of the worst "
                "frames, a numeric report, the changed intervals and a flicker graph."
            ),
            inputs=[
                C2C_VIDEO.Input("a", optional=True, tooltip="Version A (a lazy video)."),
                C2C_VIDEO.Input("b", optional=True, tooltip="Version B (a lazy video)."),
                io.Image.Input("a_images", optional=True, tooltip="Version A as an IMAGE batch instead."),
                io.Image.Input("b_images", optional=True, tooltip="Version B as an IMAGE batch instead."),
                io.String.Input("frame_map_a", default="", optional=True, multiline=False,
                                tooltip="h3.frame_ledger of A: with B's, frames are paired by SOURCE frame number."),
                io.String.Input("frame_map_b", default="", optional=True, multiline=False),
                io.Mask.Input("mask", optional=True, tooltip="1 = compare here (one mask for all, or one per frame)."),
                io.Combo.Input("method", options=list(METHODS), default="ssim",
                               tooltip="pixel: mean |A - B|. ssim: 1 - block SSIM. ms-ssim: 1 - multi-scale SSIM "
                                       "(structure at several scales; not a learned perceptual metric)."),
                io.Float.Input("threshold", default=0.02, min=0.0, max=1.0, step=0.001,
                               tooltip="A frame whose difference is above this is 'changed'."),
                io.Int.Input("chunk", default=8, min=1, max=256, tooltip="Frames read at a time (memory)."),
                io.Int.Input("max_heatmaps", default=32, min=1, max=1000,
                             tooltip="Heatmaps kept: the worst frames, in frame order."),
            ],
            outputs=[
                io.Image.Output("diff", tooltip="Heatmaps of the worst frames (black = same, yellow = most)."),
                io.String.Output("report", tooltip="JSON: per-frame scores, mean / p95 / max, worst frames."),
                io.String.Output("changed_intervals", tooltip="JSON: [first, last] frame numbers above the threshold."),
                io.Image.Output("flicker", tooltip="The difference over time; red line = threshold."),
            ],
        )

    @classmethod
    def execute(cls, a=None, b=None, a_images=None, b_images=None, frame_map_a="", frame_map_b="", mask=None,
                method="ssim", threshold=0.02, chunk=8, max_heatmaps=32) -> io.NodeOutput:
        if method not in METHODS:
            raise ValueError(f"{WHO}: method {method!r} - one of {', '.join(METHODS)}")
        A, B = _Source("a", a, a_images), _Source("b", b, b_images)
        na, nb = _numbers_from_map(frame_map_a, A.count), _numbers_from_map(frame_map_b, B.count)
        notes = []
        if na is not None and nb is not None:
            where_b = {}
            for j, num in enumerate(nb):
                if num >= 0:
                    where_b.setdefault(num, j)
            pairs = [(i, where_b[num]) for i, num in enumerate(na) if num >= 0 and num in where_b]
            if not pairs:
                raise ValueError(f"{WHO}: the frame maps share no source frame - nothing to compare.")
            numbers = [na[i] for i, _ in pairs]
            alignment = "frame maps (by source frame number)"
        else:
            n = min(A.count, B.count)
            if A.count != B.count:
                notes.append(f"A has {A.count} frames, B {B.count}: compared the first {n}")
            pairs = [(i, i) for i in range(n)]
            numbers = [(na or A.numbers)[i] for i in range(n)]
            alignment = "position"
        if mask is not None and mask.dim() == 2:
            mask = mask[None]
        scores: list[float] = []
        worst: list[tuple[float, int, torch.Tensor]] = []          # min-heap of the worst frames' heatmaps
        resized = False
        for s in range(0, len(pairs), int(chunk)):
            part = pairs[s:s + int(chunk)]
            xa = A.read([i for i, _ in part])
            xb = B.read([j for _, j in part])
            resized = resized or xb.shape[1:3] != xa.shape[1:3]
            m = None
            if mask is not None:
                m = mask if mask.shape[0] == 1 else mask[[min(i, mask.shape[0] - 1) for i, _ in part]]
            sc, heat = compare_chunk(xa, xb, method, m)
            for k, v in enumerate(sc.tolist()):
                pos = s + k
                scores.append(float(v))
                item = (float(v), pos, _heat(heat[k]))
                if len(worst) < int(max_heatmaps):
                    heapq.heappush(worst, item)
                elif item[0] > worst[0][0]:
                    heapq.heapreplace(worst, item)
        if resized:
            notes.append("B was resized to A's size for the comparison")
        if method == "ms-ssim":
            notes.append("ms-ssim is multi-scale structural similarity, not a learned perceptual metric")
        kept = sorted(worst, key=lambda t: t[1])
        diff = torch.stack([t[2] for t in kept]) if kept else torch.zeros((1, 64, 64, 3))
        flags = [v > float(threshold) for v in scores]
        arr = np.asarray(scores, dtype=np.float64)
        order = np.argsort(-arr)[:10]
        report = {
            "c2c_contract": "h3.sequence_diff", "c2c_contract_version": "1.0", "method": method,
            "alignment": alignment, "compared": len(scores), "threshold": float(threshold),
            "mean": float(arr.mean()), "p95": float(np.percentile(arr, 95)), "max": float(arr.max()),
            "changed": int(sum(flags)),
            "worst": [{"position": int(p), "frame": int(numbers[p]), "difference": float(arr[p])} for p in order],
            "scores": [round(v, 6) for v in scores], "frames": [int(x) for x in numbers],
            "heatmaps_of": [int(numbers[t[1]]) for t in kept], "masked": mask is not None, "notes": notes,
        }
        return io.NodeOutput(diff, json.dumps(report), json.dumps(intervals(numbers, flags)),
                             flicker_graph(scores, float(threshold)))


NODE_CLASS_MAPPINGS = {"MiniMaxH3_SequenceDiff": MiniMaxH3_SequenceDiff}
