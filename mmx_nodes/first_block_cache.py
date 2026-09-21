"""H3 First-Block Cache — skip the block stack when block 0 says nothing moved.

Merged from ComfyUI-MiniMaxH3-FirstBlockCache and ComfyUI_H3FBC; see
mmx_utils/first_block_cache.py for what each contributed and why both were
needed. This file is the wiring: block patches, wrappers, and the checks that
have to happen before any of it is installed.

The comfy imports are deliberately guarded. `comfy.patcher_extension` is safe
on its own, but the pack rule is that no node may need a heavy dependency to
import or to build its schema - a node that cannot be constructed vanishes
from /object_info with no error anyone can see.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from comfy_api.latest import io

_PKG = Path(__file__).resolve().parents[1]
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from mmx_utils.first_block_cache import (
    CUSTOM_MODE,
    FBC_KEY,
    PRESETS,
    CacheConfig,
    FirstBlockCache,
    FirstBlockCacheError,
)

_IMPORT_ERROR: Exception | None = None
try:
    import comfy.patcher_extension
except Exception as _e:  # noqa: BLE001 - see the module docstring
    _IMPORT_ERROR = _e
    comfy = None  # type: ignore[assignment]

log = logging.getLogger(__name__)

MODES = [*PRESETS, CUSTOM_MODE]
DEFAULT_MODE = "Balanced — 0.08, warmup 3, max 2"

# Other accelerators that patch the same blocks. Stacking two of them does not
# raise - it produces a quietly wrong image - so they are named here and
# refused by name.
CONFLICTS = {
    "minimax_h3_block_cache_t8": "H3 Block Cache (T8)",
    "easycache": "EasyCache",
    "lazycache": "LazyCache",
    "cache_dit_turbo": "CacheDiT",
    FBC_KEY: "a second H3 First-Block Cache",
}


def make_block_patch(cache: FirstBlockCache, index: int, last_index: int):
    """One patch per block. Block 0 decides; the rest obey or are skipped."""

    def patch(args, extra):
        original_block = extra["original_block"]

        if index == 0:
            entry = args["img"]
            output = original_block(args)["img"]
            # The residual is what block 0 CHANGED, which is the signal; the
            # output itself is dominated by the input and barely moves.
            cache.decide(output - entry, output)
            return {"img": output}

        state = cache.current
        if state is None:
            raise FirstBlockCacheError(
                "A block patch ran with no active cache context. The block "
                "patches are installed but the diffusion-model wrapper is not.")

        if state.use_cache:
            # Every block between the first and the last is skipped entirely;
            # the last one adds back the tail recorded at the previous full step.
            if index == last_index:
                return {"img": cache.finish_cached_step(args["img"])}
            return {"img": args["img"]}

        output = original_block(args)["img"]
        if index == last_index:
            cache.finish_full_step(output)
        return {"img": output}

    return patch


def make_diffusion_wrapper(cache: FirstBlockCache):
    """Tells the cache which step and branch each model call belongs to."""

    def wrapper(executor, *args, **kwargs):
        transformer_options = (args[3] if len(args) > 3
                               else kwargs.get("transformer_options", {})) or {}
        payload = args[4] if len(args) > 4 else kwargs.get("minimax_payload")
        cache.begin_call(args[0], args[1], transformer_options, payload)
        try:
            return executor(*args, **kwargs)
        finally:
            cache.end_call()

    return wrapper


def make_sample_wrapper(cache: FirstBlockCache, label: str, sink: list[str]):
    """Clears state per run and captures the summary for the report output."""

    def wrapper(executor, *args, **kwargs):
        cache.reset()
        log.info("H3 First-Block Cache enabled: %s", label)
        try:
            return executor(*args, **kwargs)
        finally:
            summary = cache.summary()
            log.info("%s", summary)
            sink.clear()
            sink.append(summary)
            cache.reset()

    return wrapper


class MiniMaxH3_FirstBlockCache(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MiniMaxH3_FirstBlockCache",
            display_name="H3 First-Block Cache",
            category="MiniMax H3/Sampling",
            description=(
                "Skip the transformer stack on steps where block 0 says almost "
                "nothing changed, reusing the previous step's result for blocks "
                "1..N. Typically 1.3-1.8x with no visible difference. Pushed too "
                "far it does not add artefacts - it FREEZES motion, which is "
                "harder to notice, so the temporal guard and the warmup are on "
                "by default and the report says what actually happened."
            ),
            inputs=[
                io.Model.Input("model"),
                io.Combo.Input(
                    "mode", options=MODES, default=DEFAULT_MODE,
                    tooltip="A calibrated preset, or Custom to use the values "
                            "below. In preset mode the manual values are "
                            "IGNORED - the preset name is what ran."),
                io.Float.Input(
                    "threshold", default=0.08, min=0.0, max=1.0, step=0.005,
                    optional=True,
                    tooltip="How much block 0's residual may move before the "
                            "step must run in full. Relative, not absolute: "
                            "0.08 means 8%. Higher is faster and freezes "
                            "motion sooner. Custom mode only."),
                io.Float.Input(
                    "start_percent", default=0.10, min=0.0, max=1.0, step=0.01,
                    optional=True,
                    tooltip="Skip nothing before this point in the denoise. "
                            "Early steps decide composition, so caching there "
                            "changes what the shot is of. Custom mode only."),
                io.Float.Input(
                    "end_percent", default=0.95, min=0.0, max=1.0, step=0.01,
                    optional=True,
                    tooltip="Skip nothing after this point. The last steps are "
                            "fine detail and cheap to get wrong. Custom mode "
                            "only."),
                io.Int.Input(
                    "max_consecutive_hits", default=2, min=1, max=20,
                    optional=True,
                    tooltip="How many steps may be skipped back to back. Error "
                            "compounds: two is recoverable, ten is a still "
                            "frame. Custom mode only."),
                io.Int.Input(
                    "warmup_steps", default=3, min=0, max=50, optional=True,
                    tooltip="Run this many steps in full before caching is "
                            "allowed at all, regardless of the window. Custom "
                            "mode only."),
                io.Boolean.Input(
                    "temporal_guard", default=True, optional=True,
                    tooltip="Decide on the WORST single frame instead of the "
                            "whole-clip average. Without it, one frame can "
                            "change completely while the average stays under "
                            "threshold - that frame then freezes while the rest "
                            "of the shot moves. Leave it on for video. Custom "
                            "mode only."),
                io.Int.Input(
                    "metric_stride", default=1, min=1, max=16, optional=True,
                    tooltip="Compare every Nth row instead of all of them. The "
                            "comparison allocates too; on a long clip a stride "
                            "of 4 decides the same and costs less. Custom mode "
                            "only."),
            ],
            outputs=[
                io.Model.Output("model"),
                io.String.Output(
                    display_name="report",
                    tooltip="What was installed. The RUN statistics - how many "
                            "steps were actually reused - are logged when "
                            "sampling finishes, because they do not exist yet "
                            "when this node runs."),
            ],
        )

    @classmethod
    def execute(cls, model, mode=DEFAULT_MODE, threshold=0.08,
                start_percent=0.10, end_percent=0.95, max_consecutive_hits=2,
                warmup_steps=3, temporal_guard=True, metric_stride=1):
        if _IMPORT_ERROR is not None:
            raise RuntimeError(
                "H3 First-Block Cache cannot run: ComfyUI's patcher extension "
                f"failed to import ({_IMPORT_ERROR}). This is a ComfyUI install "
                "problem, not a problem with the graph.")

        if mode == CUSTOM_MODE:
            config = CacheConfig(
                threshold=float(threshold),
                start_percent=float(start_percent),
                end_percent=float(end_percent),
                max_consecutive_hits=int(max_consecutive_hits),
                temporal_guard=bool(temporal_guard),
                warmup_steps=int(warmup_steps),
                metric_stride=int(metric_stride),
            )
            label = (f"Custom — threshold {config.threshold:.3f}, window "
                     f"{config.start_percent:.2f}-{config.end_percent:.2f}, max "
                     f"{config.max_consecutive_hits}, warmup "
                     f"{config.warmup_steps}, temporal guard "
                     f"{'on' if config.temporal_guard else 'OFF'}")
        else:
            if mode not in PRESETS:
                raise ValueError(
                    f"Unknown mode {mode!r}. Choose one of: "
                    + ", ".join(MODES) + ".")
            config = PRESETS[mode]
            label = mode
        config.validate()

        diffusion_model = model.get_model_object("diffusion_model")
        name = diffusion_model.__class__.__name__
        if name != "MiniMaxH3Model" or not hasattr(diffusion_model, "blocks"):
            raise ValueError(
                f"H3 First-Block Cache only works on MiniMax H3, but this MODEL "
                f"is {name}. The block layout it patches does not exist on other "
                "architectures.")

        block_count = len(diffusion_model.blocks)
        if block_count < 2:
            raise ValueError(
                f"H3 First-Block Cache needs at least two transformer blocks to "
                f"have a tail to reuse, but this model has {block_count}.")

        options = getattr(model, "model_options", {}) or {}
        transformer_options = options.get("transformer_options", {}) or {}
        for key, other in CONFLICTS.items():
            if key in transformer_options:
                raise ValueError(
                    f"H3 First-Block Cache cannot run together with {other}. Two "
                    "caches patching the same blocks do not error - they produce "
                    "a quietly wrong image. Use one.")

        existing = (transformer_options.get("patches_replace", {}) or {}).get("dit", {}) or {}
        if any(("double_block", i) in existing for i in range(block_count)):
            raise ValueError(
                "Another node has already replaced a DiT block on this MODEL, and "
                "this cache needs every block. Connect it directly after the "
                "model loader, before anything else that patches blocks.")

        model_sampling = model.get_model_object("model_sampling")
        start_sigma = float(model_sampling.percent_to_sigma(config.start_percent))
        end_sigma = float(model_sampling.percent_to_sigma(config.end_percent))

        patched = model.clone()
        cache = FirstBlockCache(config, start_sigma, end_sigma, block_count)
        for index in range(block_count):
            patched.set_model_patch_replace(
                make_block_patch(cache, index, block_count - 1),
                "dit", "double_block", index)

        # Marker so the conflict guard, and any accelerator added later, can
        # see that this MODEL already has a block-level cache on it.
        #
        # COPY then reassign, never setdefault(...)[k] = v: clone() can leave
        # the inner transformer_options dict shared with the model we were
        # given, so mutating it in place would mark the ORIGINAL too and a
        # second branch of the graph would report a conflict that is not there.
        options = dict(patched.model_options.get("transformer_options", {}) or {})
        options[FBC_KEY] = True
        patched.model_options["transformer_options"] = options

        sink: list[str] = []
        key = f"{FBC_KEY}_{id(cache)}"
        patched.add_wrapper_with_key(
            comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, key,
            make_diffusion_wrapper(cache))
        patched.add_wrapper_with_key(
            comfy.patcher_extension.WrappersMP.OUTER_SAMPLE, key,
            make_sample_wrapper(cache, label, sink))

        report = "\n".join([
            f"H3 First-Block Cache installed on {block_count} blocks.",
            f"Preset: {label}.",
            f"Window: {config.start_percent:.0%}-{config.end_percent:.0%} of the "
            f"denoise (sigma {start_sigma:.4f} down to {end_sigma:.4f}), after a "
            f"{config.warmup_steps}-step warmup.",
            ("Temporal guard ON: decisions use the worst single frame, so one "
             "moving frame cannot be averaged away and frozen."
             if config.temporal_guard else
             "Temporal guard OFF: decisions use the whole-clip average. One "
             "frame can change completely while the average stays under "
             "threshold, and that frame will freeze. Only turn this off for "
             "stills."),
            "Run statistics are logged when sampling finishes.",
        ])
        return io.NodeOutput(patched, report)
