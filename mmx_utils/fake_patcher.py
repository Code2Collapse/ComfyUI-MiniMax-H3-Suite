"""Test double for ComfyUI ModelPatcher — no weights, no forward pass."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any


@dataclass
class FakeModelConfig:
    """Stand-in for a ComfyUI model config.

    `ModelSamplingDiscreteFlow.__init__` reads `model_config.sampling_settings`
    (`comfy/model_sampling.py:288`) and `ModelSamplingAV.__init__` reads
    `audio_shift` off the same dict (`:336`). Both use `.get()` with defaults, so
    a plain dict is enough — but it must EXIST, or construction dies with
    `AttributeError: 'FakeModelConfig' object has no attribute 'sampling_settings'`.

    Defaults are H3's trained pair (`comfy/ldm/minimax/model.py:447`) so the fake
    behaves like the real thing rather than like an untuned flow model.
    """

    def __init__(self, sampling_settings: dict | None = None):
        self.sampling_settings = (
            {"shift": 12.0, "audio_shift": 3.0, "multiplier": 1000}
            if sampling_settings is None
            else dict(sampling_settings)
        )


@dataclass
class FakeDiffusionModel:
    blocks: list = field(default_factory=lambda: [object()] * 50)


@dataclass
class FakeInnerModel:
    model_config: FakeModelConfig = field(default_factory=FakeModelConfig)
    diffusion_model: FakeDiffusionModel = field(default_factory=FakeDiffusionModel)


@dataclass
class FakeModelPatcher:
    model_options: dict = field(default_factory=lambda: {"transformer_options": {}})
    model: FakeInnerModel = field(default_factory=FakeInnerModel)
    object_patches: dict = field(default_factory=dict)
    dit_patches: dict = field(default_factory=dict)
    wrappers: dict = field(default_factory=dict)
    _sampling: Any = None

    def clone(self) -> FakeModelPatcher:
        c = FakeModelPatcher(
            model_options=copy.deepcopy(self.model_options),
            model=copy.deepcopy(self.model),
            object_patches=copy.deepcopy(self.object_patches),
            dit_patches=copy.deepcopy(self.dit_patches),
            wrappers=copy.deepcopy(self.wrappers),
            _sampling=self._sampling,
        )
        c.model_options.setdefault("transformer_options", {})
        return c

    def get_model_object(self, key: str) -> Any:
        if key == "model_sampling":
            return self._sampling
        raise KeyError(key)

    def add_object_patch(self, key: str, value: Any) -> None:
        self.object_patches[key] = value

    def set_model_patch_replace(self, patch: Any, patch_type: str, name: str, index: int) -> None:
        key = (name, index)
        existing = self.model_options.get("transformer_options", {}).get("patches_replace", {}).get("dit", {})
        if key in existing or key in self.dit_patches:
            raise ValueError(f"dit patch {key} already occupied")
        self.dit_patches[key] = patch
        to = self.model_options.setdefault("transformer_options", {})
        pr = to.setdefault("patches_replace", {})
        dit = pr.setdefault("dit", {})
        dit[key] = patch

    def add_wrapper_with_key(self, wrapper_type: str, name: str, fn: Any) -> None:
        self.wrappers.setdefault(wrapper_type, {})[name] = fn
