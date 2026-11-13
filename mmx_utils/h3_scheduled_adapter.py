"""Sigma-scheduled H3 LoRA adapter (ledger L9.210, A26 3.5).

One sampler run, no model switch: the LoRA is installed ONCE as bypass hooks (comfy.weight_adapter
BypassInjectionManager, the path the Turbo LoRA uses), and a DIFFUSION_MODEL wrapper reads the actual video sigma of
every model call (transformer_options["sigmas"], set by core's sampler; core 0.39 samplers.py) and sets the strength
of the low-rank contribution for that call:

    out = W x + strength(sigma) * (alpha / rank) * B (A x)          (rows masked when apply_to excludes audio)

strength(sigma): 0 outside [end_sigma, start_sigma]; inside, ``maximum_strength`` times a weight that rises from 0 over
``ramp_in`` (sigma units, from start_sigma downward) and falls to 0 over ``ramp_out`` (toward end_sigma), shaped by
``schedule_curve``. start_sigma 1, end_sigma 0, no ramps = a constant strength, i.e. the legacy merge.

H3 runs text, video and audio rows through one packed sequence; core 0.39 publishes the row spans per call as
transformer_options["minimax_h3_layout"].segments ((start, end, kind); kinds text, cond, cond_audio, ref_img, ref_audio,
audio, video). ``apply_to``:
- "all rows": as the adapter was trained / extracted (identical to the constant merge);
- "text + visual rows": the adapter adds nothing to audio rows (audio, cond_audio, ref_audio). Audio can still change
  INDIRECTLY: audio rows attend to the adapted video rows. That change is measured, never assumed zero.

Patch state: the hooks live on the cloned ModelPatcher as one injection under a fixed key (re-applying replaces it,
never stacks); ComfyUI ejects them when the model is unpatched; the wrapper resets its state in ``finally`` so an
interrupted step leaves nothing behind. Fused int8 ``mlp.fc2`` layers (TensorWiseINT8Layout) never call their own
forward, so they cannot carry a scheduled hook: they are skipped and counted in the report.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

INJECTION_KEY = "mmx_h3_scheduled_hdr"
WRAPPER_KEY = "mmx_h3_scheduled_hdr"
CURVES = ("linear", "smoothstep", "cosine")
APPLY_TO = ("all rows", "text + visual rows")
AUDIO_KINDS = frozenset({"audio", "cond_audio", "ref_audio"})


@dataclass(frozen=True)
class Schedule:
    maximum_strength: float = 1.0
    start_sigma: float = 1.0
    end_sigma: float = 0.0
    ramp_in: float = 0.0
    ramp_out: float = 0.0
    curve: str = "linear"

    def validate(self) -> "Schedule":
        if not 0.0 <= self.end_sigma <= self.start_sigma:
            raise ValueError(f"end_sigma ({self.end_sigma}) must be between 0 and start_sigma ({self.start_sigma}).")
        if self.ramp_in < 0 or self.ramp_out < 0:
            raise ValueError("ramp_in and ramp_out cannot be negative.")
        if self.ramp_in + self.ramp_out > (self.start_sigma - self.end_sigma) + 1e-9:
            raise ValueError("ramp_in + ramp_out is longer than the start_sigma .. end_sigma window.")
        if self.curve not in CURVES:
            raise ValueError(f"schedule_curve must be one of {', '.join(CURVES)}.")
        return self

    @property
    def constant(self) -> bool:
        return self.start_sigma >= 1.0 and self.end_sigma <= 0.0 and self.ramp_in == 0 and self.ramp_out == 0

    def at(self, sigma: float) -> float:
        """Strength at this sigma (sampling runs from high sigma to low)."""
        s = float(sigma)
        if s > self.start_sigma + 1e-9 or s < self.end_sigma - 1e-9:
            return 0.0
        w = 1.0
        if self.ramp_in > 0:
            w = min(w, (self.start_sigma - s) / self.ramp_in)
        if self.ramp_out > 0:
            w = min(w, (s - self.end_sigma) / self.ramp_out)
        w = min(max(w, 0.0), 1.0)
        if self.curve == "smoothstep":
            w = w * w * (3.0 - 2.0 * w)
        elif self.curve == "cosine":
            w = 0.5 - 0.5 * math.cos(math.pi * w)
        return self.maximum_strength * w


class ScheduleState:
    """Per-patched-model state the wrapper writes and the hooks read (one model call at a time)."""

    def __init__(self, schedule: Schedule, apply_to: str, trace_limit: int = 256):
        if apply_to not in APPLY_TO:
            raise ValueError(f"apply_to must be one of {APPLY_TO}.")
        self.schedule = schedule
        self.mask_audio = apply_to != "all rows"
        self.strength = 0.0
        self.topts: dict | None = None
        self.trace: list[tuple[float, float]] = []                     # (sigma, strength) per model call
        self._trace_limit = trace_limit
        self._mask_cache: dict = {}

    def begin(self, sigma: float, topts: dict | None) -> None:
        self.strength = self.schedule.at(sigma)
        self.topts = topts
        if len(self.trace) < self._trace_limit:
            self.trace.append((float(sigma), float(self.strength)))

    def end(self) -> None:
        self.strength = 0.0
        self.topts = None

    def row_mask(self, x: torch.Tensor) -> torch.Tensor | None:
        """[rows, 1] mask of the rows the adapter may change, or None for all rows. Refuses to guess: a packed
        input without core's layout (another H3 implementation) is an error, not a silent all-rows fallback."""
        if not self.mask_audio:
            return None
        layout = (self.topts or {}).get("minimax_h3_layout")
        if layout is None:
            raise RuntimeError("H3 scheduled adapter: apply_to excludes audio rows, but this H3 model did not publish "
                               "transformer_options['minimax_h3_layout'] (ComfyUI core 0.39+ H3 needed).")
        rows = x.shape[-2] if x.ndim >= 2 else x.shape[0]
        key = (id(layout), rows, x.device, x.dtype)
        m = self._mask_cache.get(key)
        if m is None:
            m = torch.ones(rows, 1, device=x.device, dtype=x.dtype)
            for a, b, kind in layout.segments:
                if kind in AUDIO_KINDS:
                    m[a:b] = 0
            if len(self._mask_cache) > 8:
                self._mask_cache.clear()
            self._mask_cache[key] = m
        return m


def scheduled_lora_class():
    import comfy.weight_adapter

    class ScheduledLoRA(comfy.weight_adapter.LoRAAdapter):
        """LoRA bypass whose strength and row mask come from a ScheduleState at call time."""

        state: ScheduleState = None  # set per instance

        def bypass_forward(self, org_forward, x, *args, **kwargs):
            out = org_forward(x, *args, **kwargs)
            st = self.state
            m = (st.strength if st is not None else 0.0) * getattr(self, "multiplier", 1.0)
            if m == 0.0:
                return out
            up, down, alpha = self.weights[0], self.weights[1], self.weights[2]
            scale = (float(alpha) / down.shape[0] if alpha is not None else 1.0) * m
            # the hook parks factors on the compute device; offloaded / CPU activations still meet them here
            down, up = down.to(device=x.device, dtype=x.dtype), up.to(device=x.device, dtype=x.dtype)
            delta = F.linear(F.linear(x, down), up)
            mask = st.row_mask(delta)
            if mask is not None:
                delta = delta * mask
            return out + delta * scale

    return ScheduledLoRA


def sigma_of(args, kwargs) -> tuple[float, dict | None]:
    """The video sigma of this diffusion-model call and its transformer_options. Core 0.39's H3 forward calls the
    wrappers as (x, timestep, context, transformer_options, ...); the sampler stores the real sigma in
    transformer_options['sigmas']. Fallback: H3's timestep is sigma * 1000 (flow, multiplier 1000)."""
    topts = args[3] if len(args) > 3 and isinstance(args[3], dict) else kwargs.get("transformer_options")
    sig = (topts or {}).get("sigmas")
    if sig is not None:
        return float(torch.as_tensor(sig).flatten()[0]), topts
    ts = args[1] if len(args) > 1 else kwargs.get("timestep")
    return float(torch.as_tensor(ts).flatten()[0]) / 1000.0, topts


def install(new_model, loaded: dict, schedule: Schedule, apply_to: str, *, skip=()) -> dict:
    """Install ``loaded`` (comfy.lora.load_lora output, model keys -> adapters) as scheduled bypass hooks on the
    CLONED patcher ``new_model``. Returns counts for the report."""
    import comfy.patcher_extension
    import comfy.weight_adapter

    schedule.validate()
    state = ScheduleState(schedule, apply_to)
    Scheduled = scheduled_lora_class()
    manager = comfy.weight_adapter.BypassInjectionManager()
    sd_keys = set(new_model.model.state_dict().keys())
    n, skipped, unsupported = 0, 0, 0
    for key, adapter in loaded.items():
        if key not in sd_keys:
            continue
        if any(key.startswith(s) for s in skip):
            skipped += 1
            continue
        if not isinstance(adapter, comfy.weight_adapter.LoRAAdapter) or adapter.weights[3] is not None \
                or adapter.weights[4] is not None:
            unsupported += 1                                            # LoCon mid / DoRA: not a plain low-rank pair
            continue
        a = Scheduled(adapter.loaded_keys, adapter.weights)
        a.state = state
        manager.add_adapter(key, a, strength=1.0)
        n += 1
    injections = manager.create_injections(new_model.model)
    if manager.get_hook_count() > 0:
        new_model.set_injections(INJECTION_KEY, injections)            # replaces, never stacks

    def wrap(executor, *args, **kwargs):
        sigma, topts = sigma_of(args, kwargs)
        state.begin(sigma, topts)
        try:
            return executor(*args, **kwargs)
        finally:
            state.end()

    if hasattr(new_model, "remove_wrappers_with_key"):                  # re-applying replaces, never stacks
        new_model.remove_wrappers_with_key(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, WRAPPER_KEY)
    new_model.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, WRAPPER_KEY, wrap)
    return {"hooked": n, "skipped_fused_int8": skipped, "unsupported": unsupported, "state": state}
