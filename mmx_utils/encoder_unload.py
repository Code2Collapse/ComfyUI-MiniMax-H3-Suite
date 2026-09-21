"""Release one named text encoder from VRAM, and nothing else.

PORTED FROM: comfyui-deno-custom-nodes (GPL-3.0). This pack is GPL-3.0, so the
licences agree; see CREDITS.md.

WHY THIS MATTERS FOR H3 SPECIFICALLY. H3's text encoder is Qwen3-VL-32B. It is
used once, at the very start, to turn the prompt into conditioning - and then
it sits in VRAM for the whole sample, which on a 16GB card is the difference
between a clip rendering and an out-of-memory. Unloading it after conditioning
and before sampling is one of the largest single wins available on H3, and it
cannot be done by a generic "free memory" button: that unloads the diffusion
model too, which is the thing about to be used.

So this frees the ONE encoder wired into it, and nothing else. It is not a
VRAM cleaner and cannot make the process use zero.

TWO THINGS THAT LOOK WRONG AND ARE NOT:

  * It sits in the CONDITIONING path and passes conditioning through
    unchanged. That is the whole mechanism - the node has to run AFTER the
    encode and BEFORE the sampler, and in a dataflow graph the only way to
    say "after" is to be on the wire.

  * IS_CHANGED returns NaN, which this pack otherwise forbids. Here it is
    correct: the unload is a SIDE EFFECT, not a value. A cache hit would skip
    it, so the second run of a workflow - or a run after another branch
    reloaded the encoder - would silently not free anything while reporting
    success. The rule exists to stop NaN being used in place of knowing
    whether state changed; this node genuinely must run every time.
"""

from __future__ import annotations

import inspect
from typing import Any

try:
    import comfy.model_management as model_management
except Exception:  # noqa: BLE001 - reported at run time, not at import
    model_management = None


class EncoderUnloadError(RuntimeError):
    pass


def check_can_leave_accelerator(clip: Any) -> None:
    """Refuse when the encoder has nowhere to go.

    With --gpu-only, load and offload device are the same accelerator, so an
    "unload" moves the weights from the GPU to the GPU and frees nothing. The
    node would report success and the next step would still run out of memory,
    which is the worst of both. Say so instead.
    """
    patcher = getattr(clip, "patcher", None)
    if patcher is None:
        return  # the real check is in unload_clip(); one message, one place

    load_device = getattr(patcher, "load_device", None)
    offload_device = getattr(patcher, "offload_device", None)
    if load_device is None or offload_device is None or load_device != offload_device:
        return

    name = str(load_device).lower()
    if name == "cpu" or name.startswith("meta"):
        return

    raise EncoderUnloadError(
        "This text encoder cannot be freed from accelerator memory: its load "
        f"and offload devices are both {load_device}, so there is nowhere to "
        "move the weights to. ComfyUI was most likely started with --gpu-only. "
        "Restart it without that flag (the default dynamic VRAM mode is what "
        "this node needs) and run the workflow again.")


def unload_clip(clip: Any, *, label: str = "text encoder") -> None:
    """Unload this CLIP's patcher and every clone of it."""
    patcher = getattr(clip, "patcher", None)
    if patcher is None:
        raise EncoderUnloadError(
            f"Cannot unload the {label}: the connected CLIP has no patcher, so "
            "there is no model to release. Connect the same CLIP output the "
            "text-encode node used, not a re-wrapped or stubbed one.")
    if model_management is None:
        raise EncoderUnloadError(
            "Cannot unload the text encoder: ComfyUI's model management could "
            "not be imported. This is a ComfyUI install problem, not a problem "
            "with the graph.")

    unload = getattr(model_management, "unload_model_and_clones", None)
    empty_cache = getattr(model_management, "soft_empty_cache", None)
    if not callable(unload) or not callable(empty_cache):
        raise EncoderUnloadError(
            "Freeing a single named encoder needs ComfyUI 0.23.0 or newer - "
            "older builds can only free everything at once, which would take "
            "the diffusion model with it. Update ComfyUI.")

    # CLONES matter. The encoder is commonly cloned by LoRA and patch nodes,
    # and unloading only the original leaves the clone holding the weights -
    # so nothing is freed and the node still reports success.
    try:
        params = inspect.signature(unload).parameters
        takes_kwargs = any(p.kind is inspect.Parameter.VAR_KEYWORD
                           for p in params.values())
        if "all_devices" in params or takes_kwargs:
            unload(patcher, all_devices=True)
        else:
            unload(patcher)
    finally:
        # In a finally: if the unload raises part-way the allocator is still
        # holding freed blocks, and not returning them makes the failure worse
        # than it needs to be.
        empty_cache(force=True)


def describe(clip: Any) -> str:
    """What was freed, in terms someone can check against nvidia-smi."""
    patcher = getattr(clip, "patcher", None)
    load_device = getattr(patcher, "load_device", None) if patcher else None
    where = f" from {load_device}" if load_device is not None else ""
    return (
        f"Text encoder released{where}, along with any clones of it.\n"
        "Only the encoder wired into this node was touched - the diffusion "
        "model, the VAEs and any ControlNets are untouched, which is the "
        "point: a general 'free memory' would unload the model that is about "
        "to sample.\n"
        "This does not take the process to zero VRAM, and it will not appear "
        "to do anything if the encoder was already unloaded.")
