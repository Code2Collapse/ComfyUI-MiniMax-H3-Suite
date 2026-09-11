from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import torch

from .er_sde_stochastic import ERSDEStepDescriptor, ERSDEStochasticTracker


REFDELTA_BRIDGE_KEY = "spectrum_h3_refdelta_bridge"
REFDELTA_INTEROP_CONTRACT = (
    "comfyui-refdelta-spectrum",
    1,
    "actual-anchor-history",
    "exact-gated-stochastic-increment",
)


class RefDeltaInteropError(RuntimeError):
    """The reviewed Spectrum/RefDelta interop state became inconsistent."""


@dataclass(slots=True)
class RefDeltaInteropBridge:
    run_id: int
    tracker: ERSDEStochasticTracker | None
    api_version: ClassVar[int] = 1
    interop_contract: ClassVar[tuple[str, int, str, str]] = (
        REFDELTA_INTEROP_CONTRACT
    )
    _descriptor: ERSDEStepDescriptor | None = None

    def note_model_result(self, descriptor: ERSDEStepDescriptor) -> None:
        if descriptor.run_id != self.run_id:
            raise RefDeltaInteropError("stale Spectrum RefDelta run descriptor")
        self._descriptor = descriptor

    def model_result_is_actual(self, step_id: int) -> bool:
        descriptor = self._descriptor
        if descriptor is None or descriptor.step_id != int(step_id):
            raise RefDeltaInteropError(
                "RefDelta requested a model-result classification for the wrong step"
            )
        if descriptor.mode == "actual":
            return True
        if descriptor.mode == "forecast":
            return False
        if descriptor.mode == "replay" and descriptor.replay_source_actual is not None:
            return descriptor.replay_source_actual
        raise RefDeltaInteropError(
            f"unreviewed Spectrum RefDelta step mode {descriptor.mode!r}"
        )

    def publish_stochastic_increment(
        self,
        source_step_id: int,
        increment: torch.Tensor,
    ) -> None:
        descriptor = self._descriptor
        if descriptor is None or descriptor.step_id != int(source_step_id):
            raise RefDeltaInteropError(
                "RefDelta published a stochastic increment for the wrong step"
            )
        if self.tracker is None:
            raise RefDeltaInteropError(
                "RefDelta published a stochastic increment for a deterministic run"
            )
        self.tracker.publish_external_increment(source_step_id, increment)

    def clear(self) -> None:
        self._descriptor = None

    @property
    def is_replay_step(self) -> bool:
        return self._descriptor is not None and self._descriptor.mode == "replay"


__all__ = [
    "REFDELTA_BRIDGE_KEY",
    "REFDELTA_INTEROP_CONTRACT",
    "RefDeltaInteropBridge",
    "RefDeltaInteropError",
]
