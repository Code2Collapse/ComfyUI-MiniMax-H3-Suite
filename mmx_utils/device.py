"""Accelerator helpers — CPU fallback on OOM / missing ops."""

from __future__ import annotations

import logging
from typing import Callable, TypeVar

import torch

_LOG = logging.getLogger(__name__)

try:
    import comfy.model_management as mm
except Exception:
    mm = None

try:
    from comfy.model_management import InterruptProcessingException
except Exception:
    InterruptProcessingException = None  # type: ignore[misc, assignment]

T = TypeVar("T")


def _should_fallback(exc: BaseException) -> bool:
    if isinstance(exc, NotImplementedError):
        return True
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    if isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower():
        return True
    return False


def output_device() -> torch.device:
    """Where ComfyUI keeps IMAGE / MASK between nodes: ``intermediate_device()`` - the CPU unless the server runs
    with --gpu-only. A node that computes on the GPU must hand its result back there: core nodes such as Batch
    Images join it with other nodes' tensors and fail with "Expected all tensors to be on the same device"
    (owner's Linux log 2026-10-12, ledger L7.93)."""
    if mm is not None:
        try:
            return torch.device(mm.intermediate_device())
        except Exception:
            pass
    return torch.device("cpu")


def run_with_cpu_fallback(
    work: Callable[[torch.device], T],
    *,
    device: torch.device,
    label: str,
    logger: logging.Logger | None = None,
) -> T:
    """Run ``work(device)``. On accelerator failure, free VRAM and retry on CPU."""
    log = logger or _LOG

    def _attempt(dev: torch.device) -> T:
        return work(dev)

    try:
        return _attempt(device)
    except Exception as exc:
        if InterruptProcessingException is not None and isinstance(exc, InterruptProcessingException):
            raise
        if device.type == "cpu" or not _should_fallback(exc):
            raise
        log.warning("%s: accelerator failed (%s); retrying on CPU.", label, exc)
        if mm is not None:
            try:
                mm.soft_empty_cache()
            except Exception:
                pass
        return _attempt(torch.device("cpu"))
