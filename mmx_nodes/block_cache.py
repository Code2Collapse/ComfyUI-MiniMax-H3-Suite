"""P9 — H3 Block Cache T8 (Apache port)."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from comfy_api.latest import io

# Deep H3 model imports are OPTIONAL AT IMPORT TIME.
#
# `comfy.ldm.minimax.model` transitively imports `comfy.ldm.modules.attention`, which calls into the
# installed `comfy_kitchen` build. When the checked-out core and the installed environment are at
# different versions that raises **AttributeError, not ImportError** — observed here as:
#   AttributeError: module 'comfy_kitchen' has no attribute 'int8_attention_is_available'
# An unguarded module-level import therefore makes this whole node (and every test that touches it)
# uncollectable, which is how a pack silently vanishes from /object_info.
#
# Per the pack rule "no node may require a heavy dependency to import or build its schema": import
# defensively, keep the schema always constructible, and fail LOUDLY in execute() instead.
_H3_IMPORT_ERROR: Exception | None = None
try:
    import comfy.ldm.common_dit
    import comfy.model_patcher
    import comfy.model_prefetch
    import comfy.model_sampling
    import comfy.patcher_extension
    from comfy.ldm.minimax.model import MiniMaxH3Model, unpack_audio, unpatchify_video
except Exception as _e:  # noqa: BLE001 - see the AttributeError note above
    _H3_IMPORT_ERROR = _e
    comfy = None  # type: ignore[assignment]
    MiniMaxH3Model = unpack_audio = unpatchify_video = None  # type: ignore[assignment]

try:
    from comfy.ldm.minimax.model import time_shift_slope as _legacy_time_shift_slope
except Exception:
    _legacy_time_shift_slope = None


def _require_h3_runtime() -> None:
    """Raise a human sentence if the H3 model modules could not be imported."""
    if _H3_IMPORT_ERROR is not None:
        raise RuntimeError(
            "MiniMax H3 Block Cache needs ComfyUI's MiniMax H3 model modules, which failed to "
            f"import in this environment ({type(_H3_IMPORT_ERROR).__name__}: {_H3_IMPORT_ERROR}). "
            "This usually means the running ComfyUI and its installed comfy_kitchen build are at "
            "different versions. Update ComfyUI, or run this node inside a normal ComfyUI process."
        )

_PKG = Path(__file__).resolve().parents[1]
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from mmx_utils.h3_block_cache import CACHE_KEY, H3BlockCache, H3BlockCacheConfig, H3BlockCacheHit, H3BlockPatch
from mmx_utils.sampling_guard import SPECTRUM_BINDING_KEY, check_dit_replace_conflicts_before_install

WRAPPER_KEY = "minimax_h3_block_cache_t8"
# Guarded: `comfy` is None when the deep H3 imports above failed.
H3_RETURNS_RAW_AUDIO_VELOCITY = (
    comfy is not None and hasattr(comfy.model_sampling, "ModelSamplingAV")
)
PREFETCH_CLEANUP_TAKES_MODULE = (
    comfy is not None and hasattr(comfy.model_prefetch, "cleanup_prefetched_modules")
)


def _finalize_audio_velocity(
    audio_out,
    sigma_video,
    shift_video: float,
    shift_audio: float,
    *,
    raw_audio_velocity: bool,
):
    if raw_audio_velocity:
        return -audio_out
    if _legacy_time_shift_slope is None:
        raise RuntimeError("Legacy MiniMax H3 audio slope helper is unavailable")
    audio_slope = _legacy_time_shift_slope(sigma_video, shift_video, shift_audio).to(audio_out.dtype)
    return (-audio_slope) * audio_out


def h3_block_cache_sample_wrapper(executor, *args, **kwargs):
    guider = executor.class_obj
    original_model_options = guider.model_options
    runtime_cache = None
    try:
        if original_model_options.get(SPECTRUM_BINDING_KEY) is not None:
            raise RuntimeError(
                "MiniMax H3 Block Cache cannot be combined with Spectrum Apply MiniMax H3; use only one acceleration node"
            )
        transformer_options = original_model_options["transformer_options"]
        if "easycache" in transformer_options:
            raise RuntimeError("MiniMax H3 Block Cache cannot be combined with EasyCache or LazyCache")

        prototype = transformer_options[CACHE_KEY]
        guider.model_options = comfy.model_patcher.create_model_options_clone(original_model_options)
        runtime_cache = prototype.clone().prepare(guider.model_patcher.model.model_sampling)
        guider.model_options["transformer_options"][CACHE_KEY] = runtime_cache
        return executor(*args, **kwargs)
    finally:
        if runtime_cache is not None:
            logging.info(
                "MiniMax H3 Block Cache - cached %d/%d model forwards, cache %.1f MiB on %s",
                runtime_cache.cache_hits,
                runtime_cache.total_forwards,
                runtime_cache.cache_bytes() / (1024 * 1024),
                runtime_cache.config.cache_device,
            )
            runtime_cache.reset()
        guider.model_options = original_model_options


def _cleanup_short_circuited_prefetch(model: MiniMaxH3Model):
    block_ids = {id(block) for block in model.blocks}
    for queue in reversed(comfy.model_prefetch.PREFETCH_QUEUES):
        belongs_to_model = False
        for entry in queue:
            module = entry[1][0] if isinstance(entry, tuple) else entry
            if id(module) in block_ids:
                belongs_to_model = True
                break
        if not belongs_to_model:
            continue
        for entry in queue:
            if not isinstance(entry, tuple):
                continue
            prefetched_module, comfy_modules = entry[1]
            if comfy_modules is not None:
                if PREFETCH_CLEANUP_TAKES_MODULE:
                    comfy.model_prefetch.cleanup_prefetched_modules(prefetched_module, comfy_modules)
                else:
                    comfy.model_prefetch.cleanup_prefetched_modules(comfy_modules)
        queue[:] = [None]
        return


def h3_block_cache_diffusion_wrapper(executor, *args, **kwargs):
    try:
        return executor(*args, **kwargs)
    except H3BlockCacheHit as hit:
        model = executor.class_obj
        _cleanup_short_circuited_prefetch(model)

        video_x, audio_x = args[0][0], args[0][1]
        timestep = args[1]
        transformer_options = args[3]
        orig_t, orig_h, orig_w = video_x.shape[2:]
        padded_video = comfy.ldm.common_dit.pad_to_patch_size(video_x, model.patch_size)
        latent_t, latent_h, latent_w = padded_video.shape[2:]

        video_rows, audio_rows = model.final_layer(hit.hidden, hit.t_emb, hit.video_segment, hit.audio_segment)
        video_out = unpatchify_video(
            video_rows,
            latent_t,
            latent_h // model.patch_size[1],
            latent_w // model.patch_size[2],
            model.latents_dim,
            model.patch_size,
        )
        video_out = video_out[:, :, :orig_t, :orig_h, :orig_w]
        audio_out = unpack_audio(audio_rows)

        sigma_video = (timestep.flatten()[0] / 1000.0).float().clamp(min=1e-6)
        shift_video = float(transformer_options.get("minimax_h3_sigma_shift_video", model.sigma_shift_video))
        shift_audio = float(transformer_options.get("minimax_h3_sigma_shift_audio", model.sigma_shift_audio))
        audio_velocity = _finalize_audio_velocity(
            audio_out,
            sigma_video,
            shift_video,
            shift_audio,
            raw_audio_velocity=H3_RETURNS_RAW_AUDIO_VELOCITY,
        )
        return [-video_out.to(video_x.dtype), audio_velocity.to(audio_x.dtype)]


class MiniMaxH3_BlockCacheT8(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MiniMaxH3_BlockCacheT8",
            display_name="H3 Block Cache (T8)",
            category="MiniMax H3/Sampling",
            description=(
                "H3-specific F1B0 residual cache with dual audio+video metric. "
                "Patches only block 0 and the last block; refuses if any dit index is already occupied. "
                "cache_device defaults to cpu for 8 GB VRAM."
            ),
            inputs=[
                io.Model.Input("model"),
                io.Float.Input("residual_diff_threshold", default=0.12, min=0.0, max=1.0, step=0.01),
                io.Float.Input("start_percent", default=0.08, min=0.0, max=1.0, step=0.01, advanced=True),
                io.Float.Input("end_percent", default=0.95, min=0.0, max=1.0, step=0.01, advanced=True),
                io.Int.Input("max_consecutive_hits", default=2, min=1, max=10, step=1, advanced=True),
                io.Combo.Input("cache_device", options=["cpu", "gpu"], default="cpu", advanced=True),
                io.Int.Input("metric_stride", default=8, min=1, max=32, step=1, advanced=True),
                io.Boolean.Input("verbose", default=False, advanced=True),
            ],
            outputs=[io.Model.Output("model")],
        )

    @classmethod
    def execute(
        cls,
        model,
        residual_diff_threshold,
        start_percent,
        end_percent,
        max_consecutive_hits,
        cache_device,
        metric_stride,
        verbose,
    ) -> io.NodeOutput:
        _require_h3_runtime()
        if start_percent >= end_percent:
            raise ValueError("MiniMax H3 Block Cache start_percent must be lower than end_percent")

        diffusion_model = model.model.diffusion_model
        if not isinstance(diffusion_model, MiniMaxH3Model):
            raise ValueError("MiniMax H3 Block Cache requires a native MiniMax H3 diffusion model")
        if model.model_options.get(SPECTRUM_BINDING_KEY) is not None:
            raise ValueError(
                "MiniMax H3 Block Cache cannot be combined with Spectrum Apply MiniMax H3; use only one acceleration node"
            )

        total_blocks = len(diffusion_model.blocks)
        if total_blocks < 2:
            raise ValueError("MiniMax H3 Block Cache requires at least two DiT blocks")

        transformer_options = model.model_options["transformer_options"]
        if "easycache" in transformer_options:
            raise ValueError("MiniMax H3 Block Cache cannot be combined with EasyCache or LazyCache")

        existing_replacements = transformer_options.get("patches_replace", {}).get("dit", {})
        install_keys = [("double_block", 0), ("double_block", total_blocks - 1)]
        check_dit_replace_conflicts_before_install(existing_replacements, install_keys)

        model = model.clone()
        config = H3BlockCacheConfig(
            residual_diff_threshold=residual_diff_threshold,
            start_percent=start_percent,
            end_percent=end_percent,
            max_consecutive_hits=max_consecutive_hits,
            cache_device=cache_device,
            metric_stride=metric_stride,
            verbose=verbose,
        )
        transformer_options = model.model_options["transformer_options"].copy()
        transformer_options[CACHE_KEY] = H3BlockCache(config, total_blocks)
        model.model_options["transformer_options"] = transformer_options

        model.set_model_patch_replace(H3BlockPatch(0), "dit", "double_block", 0)
        model.set_model_patch_replace(H3BlockPatch(total_blocks - 1), "dit", "double_block", total_blocks - 1)
        model.add_wrapper_with_key(
            comfy.patcher_extension.WrappersMP.OUTER_SAMPLE,
            WRAPPER_KEY,
            h3_block_cache_sample_wrapper,
        )
        model.add_wrapper_with_key(
            comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL,
            WRAPPER_KEY,
            h3_block_cache_diffusion_wrapper,
        )
        return io.NodeOutput(model)
