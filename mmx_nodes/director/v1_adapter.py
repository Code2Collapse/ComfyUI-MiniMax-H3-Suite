"""Expose a classic (V1) node class through this pack's V3 extension.

WHY THIS EXISTS, and why the obvious alternative is dangerous.

This pack registers through `comfy_entrypoint`, the V3 path, which hands
ComfyUI a list of `io.ComfyNode` subclasses. The all-in-one Director was
written against the classic API - a plain class with `INPUT_TYPES()`,
`RETURN_TYPES` and a named function.

The obvious fix is to also export `NODE_CLASS_MAPPINGS` from the pack root.
Do not. ComfyUI's loader reads:

    if hasattr(module, "NODE_CLASS_MAPPINGS") ...:
        ...
        return True
    elif hasattr(module, "comfy_entrypoint"):

The V1 branch RETURNS. A pack that exports both gets the V1 branch and its
`comfy_entrypoint` is never called - so adding one mapping to register one
node would silently unregister the other ninety-seven, and the only symptom is
a mostly-empty menu.

So the node is adapted the other way: a generated `io.ComfyNode` subclass whose
schema is derived from the V1 `INPUT_TYPES()` at class-creation time, and whose
`execute` delegates to the original. Derived rather than transcribed, because a
hand-copied schema of twenty-odd inputs drifts from the implementation the
first time either is touched, and the drift shows up as a widget that silently
stops being read.
"""

from __future__ import annotations

from comfy_api.latest import io

# Classic type-string -> V3 input class. Only the types actually used by the
# nodes adapted here; anything else raises rather than guessing, because a
# wrong guess produces a socket of the wrong type that fails at connect time
# with no explanation.
_INPUT_TYPES = {
    "MODEL": io.Model.Input,
    "CLIP": io.Clip.Input,
    "VAE": io.Vae.Input,
    "IMAGE": io.Image.Input,
    "MASK": io.Mask.Input,
    "AUDIO": io.Audio.Input,
    "LATENT": io.Latent.Input,
    "CONDITIONING": io.Conditioning.Input,
    "INT": io.Int.Input,
    "FLOAT": io.Float.Input,
    "STRING": io.String.Input,
    "BOOLEAN": io.Boolean.Input,
}

_OUTPUT_TYPES = {
    "MODEL": io.Model.Output,
    "CLIP": io.Clip.Output,
    "VAE": io.Vae.Output,
    "IMAGE": io.Image.Output,
    "MASK": io.Mask.Output,
    "AUDIO": io.Audio.Output,
    "LATENT": io.Latent.Output,
    "CONDITIONING": io.Conditioning.Output,
    "INT": io.Int.Output,
    "FLOAT": io.Float.Output,
    "STRING": io.String.Output,
    "BOOLEAN": io.Boolean.Output,
}

# Keys the classic API accepts in an input's options dict that have no V3
# equivalent, or that V3 derives itself. Passing them through raises TypeError.
_DROP = {"forceInput", "dynamicPrompts", "advanced", "lazy", "rawLink",
         "defaultInput", "control_after_generate", "widgetType", "image_upload",
         "image_folder", "animated", "video_upload", "placeholder"}


class V1AdapterError(TypeError):
    """Raised when a classic spec cannot be represented, rather than guessed at."""


def _build_input(name: str, spec, *, optional: bool):
    """One classic input tuple -> one V3 Input."""
    if not isinstance(spec, (tuple, list)) or not spec:
        raise V1AdapterError(
            f"Input {name!r} is not a (type, options) tuple: {spec!r}")

    type_spec = spec[0]
    options = dict(spec[1]) if len(spec) > 1 and isinstance(spec[1], dict) else {}
    tooltip = options.pop("tooltip", None)
    for key in list(options):
        if key in _DROP:
            options.pop(key)

    # A list in the type slot is a combo: the list IS the options.
    if isinstance(type_spec, (list, tuple)):
        default = options.pop("default", None)
        choices = [str(c) for c in type_spec]
        return io.Combo.Input(
            name, options=choices,
            default=default if default is not None else (choices[0] if choices else None),
            optional=optional, tooltip=tooltip)

    factory = _INPUT_TYPES.get(str(type_spec))
    if factory is None:
        raise V1AdapterError(
            f"Input {name!r} has type {type_spec!r}, which this adapter does not "
            "know how to express as a V3 input. Add it to _INPUT_TYPES - do not "
            "let it fall back to a generic socket, which fails at connect time "
            "with nothing to read.")

    # multiline belongs to String only; passing it elsewhere is a TypeError.
    if str(type_spec) != "STRING":
        options.pop("multiline", None)
    return factory(name, optional=optional, tooltip=tooltip, **options)


def adapt(v1_cls, *, node_id: str, display_name: str, category: str,
          description: str = ""):
    """Return an io.ComfyNode subclass wrapping a classic node class."""
    spec = v1_cls.INPUT_TYPES()
    inputs = []
    for name, entry in (spec.get("required") or {}).items():
        inputs.append(_build_input(name, entry, optional=False))
    for name, entry in (spec.get("optional") or {}).items():
        inputs.append(_build_input(name, entry, optional=True))

    names = getattr(v1_cls, "RETURN_NAMES", None) or ()
    outputs = []
    for i, ret in enumerate(v1_cls.RETURN_TYPES):
        factory = _OUTPUT_TYPES.get(str(ret))
        if factory is None:
            raise V1AdapterError(
                f"Output {i} of {node_id} has type {ret!r}, unknown to this adapter.")
        outputs.append(factory(display_name=names[i] if i < len(names) else None))

    fn_name = getattr(v1_cls, "FUNCTION", "execute")
    doc = description or (getattr(v1_cls, "DESCRIPTION", "") or "")

    class Adapted(io.ComfyNode):
        @classmethod
        def define_schema(cls) -> io.Schema:
            return io.Schema(
                node_id=node_id,
                display_name=display_name,
                category=category,
                description=doc,
                inputs=inputs,
                outputs=outputs,
            )

        @classmethod
        def execute(cls, **kwargs) -> io.NodeOutput:
            # A fresh instance per run: the classic API allows per-call state
            # on self, and a shared instance would leak it between executions.
            result = getattr(v1_cls(), fn_name)(**kwargs)
            if isinstance(result, io.NodeOutput):
                return result
            if not isinstance(result, tuple):
                result = (result,)
            return io.NodeOutput(*result)

    Adapted.__name__ = f"{v1_cls.__name__}V3"
    Adapted.__qualname__ = Adapted.__name__
    Adapted.__doc__ = doc or v1_cls.__doc__
    return Adapted
