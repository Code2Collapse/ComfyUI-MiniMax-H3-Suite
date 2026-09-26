"""The finishing nodes on CUDA tensors.

ComfyUI routinely hands these nodes CUDA images (a GPU VAE decode), and every
CPU-only test passed while three of them crashed on the GPU: blur kernels built
on the CPU in Pixel Repair, grain noise built on the CPU and added to a CUDA
image in Detail Match, and shape tensors on the CPU subtracted from a CUDA
correlation surface in Plate Restore's registration. A final review caught two;
a sweep for tensors created without a device caught the third.

These run the real entry points end to end on CUDA and compare against the CPU
result, so a device bug fails here rather than in a user's render. Skipped
where there is no GPU.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

PACK = Path(__file__).resolve().parents[1]
if str(PACK) not in sys.path:
    sys.path.insert(0, str(PACK))

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")

from mmx_utils.detail_match import detail_match_frames  # noqa: E402
from mmx_utils.pixel_repair import repair_frames  # noqa: E402
from mmx_utils.plate_restore import restore_frames  # noqa: E402
from mmx_utils.reference_color_match import reference_color_match_frames  # noqa: E402
from mmx_utils.registration import masked_ncc_shift  # noqa: E402
from mmx_utils.drift_qc import inject_pixel_shift  # noqa: E402

CUDA = torch.device("cuda")


def _texture(h, w, seed):
    g = torch.Generator().manual_seed(seed)
    n = F.conv2d(torch.rand(1, 1, h, w, generator=g), torch.ones(1, 1, 5, 5) / 25, padding=2)[0, 0]
    n = (n - n.mean()) / n.std().clamp_min(1e-6) * 0.05
    y = torch.linspace(0, 1, h).unsqueeze(1).expand(h, w)
    x = torch.linspace(0, 1, w).unsqueeze(0).expand(h, w)
    return torch.stack([0.45 + 0.3 * x + n, 0.4 + 0.25 * y + n, 0.42 + 0.2 * x * y + n], -1).clamp(0, 1)


def _scene(n=3, size=128):
    plate = _texture(size, size, 3)
    full = torch.zeros(n, size + 32, size + 32, 3)
    for i in range(n):
        full[i, 16:16 + size, 16:16 + size] = plate
    yy, xx = torch.meshgrid(torch.arange(size), torch.arange(size), indexing="ij")
    edit = (((xx - size // 2) ** 2 + (yy - size // 2) ** 2) < (size * 0.22) ** 2).float()
    crops = full[:, 16:16 + size, 16:16 + size].clone()
    bbox = {"x": 16, "y": 16, "width": size, "height": size}
    return full, crops, edit.unsqueeze(0).expand(n, -1, -1).clone(), bbox


def _both(fn, *tensors, **kw):
    """Run fn on CPU and on CUDA; return (cpu_out, cuda_out_moved_to_cpu)."""
    cpu = fn(*tensors, **kw)
    gpu = fn(*[t.to(CUDA) if isinstance(t, torch.Tensor) else t for t in tensors], **kw)
    return cpu, gpu


def test_registration_on_cuda():
    img = _texture(96, 96, 1)
    ring = torch.zeros(96, 96, dtype=torch.bool)
    ring[8:88, 8:30] = True
    ring[8:88, 66:88] = True
    luma = img.mean(-1)
    moving = inject_pixel_shift(img.unsqueeze(0), 1.0, -0.5).squeeze(0).mean(-1)
    dy, dx, _ = masked_ncc_shift(luma.to(CUDA), moving.to(CUDA), ring.to(CUDA))
    assert abs(dx - (-1.0)) < 0.05 and abs(dy - 0.5) < 0.05


def test_plate_restore_on_cuda_matches_cpu():
    full, crops, edit, bbox = _scene()
    shifted = torch.stack([inject_pixel_shift(c.unsqueeze(0), 0.75, 0.0).squeeze(0) for c in crops])
    cpu, _, _ = restore_frames(shifted, full, [bbox] * 3, edit, 8, 12, "correct", 4.0, "affine", True)
    gpu, _, _ = restore_frames(shifted.to(CUDA), full.to(CUDA), [bbox] * 3, edit.to(CUDA),
                               8, 12, "correct", 4.0, "affine", True)
    assert gpu.device.type == "cuda"
    assert (gpu.cpu() - cpu).abs().max().item() < 1e-3


def test_detail_match_on_cuda_matches_cpu():
    full, crops, edit, bbox = _scene()
    soft = torch.stack([
        F.conv2d(F.pad(c.permute(2, 0, 1).unsqueeze(1), (1, 1, 1, 1), mode="reflect"),
                 torch.ones(1, 1, 3, 3) / 9).squeeze(1).permute(1, 2, 0)
        for c in crops])
    kw = dict(grow_px=8, sharpen=0.8, max_gain=2.5, grain=1.0, seed=7)
    cpu, _ = detail_match_frames(soft, full, [bbox] * 3, edit, **kw)
    gpu, _ = detail_match_frames(soft.to(CUDA), full.to(CUDA), [bbox] * 3, edit.to(CUDA), **kw)
    assert gpu.device.type == "cuda"
    # Same seed, noise synthesised on the CPU in both cases: identical grain.
    assert (gpu.cpu() - cpu).abs().max().item() < 1e-3


def test_reference_colour_match_on_cuda_matches_cpu():
    full, crops, edit, bbox = _scene()
    frames = full.clone()
    mask = torch.zeros(3, full.shape[1], full.shape[2])
    mask[:, 16:144, 16:144] = edit
    ref = _texture(64, 64, 9)
    cpu, _ = reference_color_match_frames(frames, mask, ref, None, 0.8, "mean_std", False, 5)
    gpu, _ = reference_color_match_frames(frames.to(CUDA), mask.to(CUDA), ref.to(CUDA), None,
                                          0.8, "mean_std", False, 5)
    assert gpu.device.type == "cuda"
    assert (gpu.cpu() - cpu).abs().max().item() < 1e-3


def test_pixel_repair_on_cuda_matches_cpu():
    base = _texture(96, 96, 4)
    clip = base.unsqueeze(0).repeat(4, 1, 1, 1)
    g = torch.Generator().manual_seed(2)
    idx = torch.randperm(96 * 96, generator=g)[:40]
    clip[1].view(-1, 3)[idx] = torch.rand(40, 3, generator=g)
    cpu, cpu_dmg, _ = repair_frames(clip, None, 0.6, 0.6, False)
    gpu, gpu_dmg, _ = repair_frames(clip.to(CUDA), None, 0.6, 0.6, False)
    assert gpu.device.type == "cuda"
    assert (gpu.cpu() - cpu).abs().max().item() < 1e-3
    assert torch.equal(gpu_dmg.cpu() > 0.5, cpu_dmg > 0.5)


_STREAM_FRAMES = 12
_STREAM_SIZE = 1024
_PEAK_BYTES = int(1.5 * 1024**3)


def _large_scene(n=_STREAM_FRAMES, size=_STREAM_SIZE):
    plate = _texture(size, size, 11)
    full = torch.zeros(n, size + 32, size + 32, 3)
    for i in range(n):
        full[i, 16:16 + size, 16:16 + size] = plate
    yy, xx = torch.meshgrid(torch.arange(size), torch.arange(size), indexing="ij")
    edit = (((xx - size // 2) ** 2 + (yy - size // 2) ** 2) < (size * 0.22) ** 2).float()
    crops = full[:, 16:16 + size, 16:16 + size].clone()
    bbox = {"x": 16, "y": 16, "width": size, "height": size}
    return full, crops, edit.unsqueeze(0).expand(n, -1, -1).clone(), bbox


def _assert_streamed(fn, *args, **kw):
    """CPU inputs + device=cuda: output on CPU, matches reference, peak VRAM bounded."""
    ref = fn(*args, **kw, device=None)
    if isinstance(ref, tuple):
        ref_out = ref[0]
    else:
        ref_out = ref
    torch.cuda.reset_peak_memory_stats()
    out = fn(*args, **kw, device=CUDA)
    if isinstance(out, tuple):
        out_t = out[0]
    else:
        out_t = out
    assert out_t.device.type == "cpu"
    assert (out_t - ref_out).abs().max().item() < 1e-3
    assert torch.cuda.max_memory_allocated() < _PEAK_BYTES


def test_plate_restore_streams_frames_to_cuda():
    full, crops, edit, bbox = _large_scene()
    shifted = torch.stack([inject_pixel_shift(c.unsqueeze(0), 0.5, 0.0).squeeze(0) for c in crops])
    _assert_streamed(
        restore_frames,
        shifted,
        full,
        [bbox] * _STREAM_FRAMES,
        edit,
        8,
        12,
        "correct",
        4.0,
        "affine",
        True,
    )


def test_detail_match_streams_frames_to_cuda():
    full, crops, edit, bbox = _large_scene()
    soft = torch.stack([
        F.conv2d(F.pad(c.permute(2, 0, 1).unsqueeze(1), (1, 1, 1, 1), mode="reflect"),
                 torch.ones(1, 1, 3, 3) / 9).squeeze(1).permute(1, 2, 0)
        for c in crops
    ])
    kw = dict(
        grow_px=8,
        sharpen=0.8,
        max_gain=2.5,
        grain=1.0,
        seed=7,
    )
    ref, _ = detail_match_frames(soft, full, [bbox] * _STREAM_FRAMES, edit, **kw, device=None)
    torch.cuda.reset_peak_memory_stats()
    out, _ = detail_match_frames(soft, full, [bbox] * _STREAM_FRAMES, edit, **kw, device=CUDA)
    assert out.device.type == "cpu"
    assert (out - ref).abs().max().item() < 1e-3
    assert torch.cuda.max_memory_allocated() < _PEAK_BYTES


def test_reference_colour_match_streams_frames_to_cuda():
    full, _, edit, _ = _large_scene()
    n = _STREAM_FRAMES
    size = _STREAM_SIZE
    mask = torch.zeros(n, full.shape[1], full.shape[2])
    mask[:, 16:16 + size, 16:16 + size] = edit
    ref = _texture(64, 64, 9)
    _assert_streamed(
        reference_color_match_frames,
        full,
        mask,
        ref,
        None,
        0.8,
        "mean_std",
        False,
        5,
    )


def test_pixel_repair_streams_frames_to_cuda():
    base = _texture(_STREAM_SIZE, _STREAM_SIZE, 4)
    clip = base.unsqueeze(0).repeat(_STREAM_FRAMES, 1, 1, 1)
    g = torch.Generator().manual_seed(2)
    idx = torch.randperm(_STREAM_SIZE * _STREAM_SIZE, generator=g)[:200]
    clip[3].view(-1, 3)[idx] = torch.rand(200, 3, generator=g)
    ref, ref_dmg, _ = repair_frames(clip, None, 0.6, 0.6, False, device=None)
    torch.cuda.reset_peak_memory_stats()
    out, dmg, _ = repair_frames(clip, None, 0.6, 0.6, False, device=CUDA)
    assert out.device.type == "cpu"
    assert (out - ref).abs().max().item() < 1e-3
    assert torch.equal(dmg > 0.5, ref_dmg > 0.5)
    assert torch.cuda.max_memory_allocated() < _PEAK_BYTES
