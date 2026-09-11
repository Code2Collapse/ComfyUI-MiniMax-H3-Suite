# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) NikoDemon80 — ComfyUI-H3-Motion-Context
# PORTED FROM: ComfyUI-H3-Motion-Context :: layout_contract.py @ third_party

"""Check that ComfyUI's H3 layout still means what this pack thinks it means.

Nothing here modifies ComfyUI. The upstream pack used to wrap
`PackedLayout.__init__` because stock rejected any keyframe anchor other
than the first or last frame. ComfyUI 0.34.0 lifted that restriction, so the
node now just builds keyframe dicts and hands them over.

What is left is a dependency on meaning rather than on structure. Stock
positions a keyframe at

    cond_t = cursor + FRAME_RESCALE * kf["resolved_frame_index"]

Two properties of that line matter to this pack and neither is guaranteed
upstream: fractional and negative anchor indices for pinned audio windows.

This runs once, before the first render, and proves the arithmetic still
holds. If it does not, the node refuses and says what moved.
"""

from __future__ import annotations

import logging


# H3's model module is imported ON DEMAND, not at module scope. The installed
# ComfyUI on some boxes has no comfy.ldm.minimax at all, and a module-scope
# import there raises inside __init__'s try/except, which silently drops all six
# Motion Context nodes from the menu with no visible error. Deferring it lets the
# nodes REGISTER and explain themselves when actually run.
class _MMProxy:
    """Resolves comfy.ldm.minimax.model on first attribute access."""

    _mod = None

    def __getattr__(self, name):
        if _MMProxy._mod is None:
            try:
                import comfy.ldm.minimax.model as _m  # type: ignore
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError(
                    "MiniMax H3 Motion Context needs ComfyUI's native H3 model code "
                    "(comfy.ldm.minimax), which this ComfyUI build does not provide: "
                    + str(exc)
                    + ". Update ComfyUI to a build that ships MiniMax H3 support."
                ) from exc
            _MMProxy._mod = _m
        return getattr(_MMProxy._mod, name)


mm = _MMProxy()

_LOG = logging.getLogger("mmx_motion_context")

_checked = None  # None not yet run, True passed, str the failure


class _Stub:
    """Stands in for a latent. The layout reads only the shape."""

    def __init__(self, shape):
        self.shape = shape


def _origin(layout):
    """The coordinate the target clip starts at, read off the built layout."""
    a, b, kind = layout.segments[-1]
    if kind != "video" or b <= a:
        raise RuntimeError(
            "expected the target video rows to be the last layout segment, "
            "found %r spanning %d rows" % (kind, b - a))
    return float(layout.position_ids[a, 0])


def _seg(layout, kind):
    return [(a, b) for a, b, k in layout.segments if k == kind]


def _check():
    text_len, latent_t, lh, lw, audio_t = 7, 7, 22, 38, 16
    fr = mm.FRAME_RESCALE

    if hasattr(mm.PackedLayout.__init__, "__wrapped__"):
        pass  # reported by ensure(); the checks below decide either way

    def build(keyframes=None, refs=None):
        return mm.PackedLayout(text_len, latent_t, lh, lw, audio_t,
                               keyframes=keyframes, refs=refs)

    video = _Stub((1, 24, 1, lh, lw))

    lay = build(keyframes=[{"resolved_frame_index": p, "latent": video}
                           for p in (0, 3)])
    o = _origin(lay)
    got = [float(lay.position_ids[a, 0]) - o for a, _ in _seg(lay, "cond")]
    want = [0.0, fr * 3]
    if len(got) != 2 or any(abs(g - w) > 1e-9 for g, w in zip(got, want)):
        raise RuntimeError(
            "keyframe anchors no longer sit at FRAME_RESCALE per pixel "
            "frame past the target origin: got %s, expected %s" % (got, want))

    lay = build(keyframes=[{"resolved_frame_index": 3, "latent": video}],
                refs=[{"kind": "audio", "ref_audio_t": 8}])
    a, _ = _seg(lay, "cond")[0]
    gap = float(lay.position_ids[a, 0]) - _origin(lay)
    if abs(gap - fr * 3) > 1e-9:
        raise RuntimeError(
            "a reference moved the anchors relative to the target: anchor "
            "sits %.6f past the origin, expected %.6f. Stock no longer "
            "compensates keyframes for reference blocks." % (gap, fr * 3))

    rt, end_coord = 40, 8
    end_frame = end_coord / fr
    idx = end_frame - rt / fr
    if idx >= 0 or idx == int(idx):
        raise RuntimeError("the audio check is not exercising a fractional "
                           "negative index; its own numbers are wrong")
    lay = build(keyframes=[{"resolved_frame_index": idx,
                            "audio_latent": _Stub((1, 32, 2, rt))}])
    spans = _seg(lay, "cond_audio")
    if len(spans) != 1:
        raise RuntimeError(
            "a keyframe carrying an audio latent produced %d cond_audio "
            "segments, expected 1" % len(spans))
    a, b = spans[0]
    if b - a != 2 * rt:
        raise RuntimeError(
            "the pinned audio window has %d rows for %d latent steps, "
            "expected %d (stereo, channel-major)" % (b - a, rt, 2 * rt))
    o = _origin(lay)
    t = lay.position_ids[a:b, 0]
    start = float(t.min()) - o
    end = start + float(rt)
    if abs(end - fr * end_frame) > 1e-9:
        raise RuntimeError(
            "the pinned audio window ends %.6f past the target origin, "
            "expected %.6f. A fractional or negative resolved_frame_index "
            "is no longer taken literally, so pinned audio would land at "
            "the wrong instant." % (end, fr * end_frame))

    if start >= 0.0:
        raise RuntimeError(
            "the pinned audio window starts %.6f past the target origin; it "
            "should start before it and end at the join" % start)


def _wrapper_origin():
    init = getattr(mm.PackedLayout, "__init__", None)
    if init is None:
        return None
    if getattr(init, "_h3_motion_context_layout_patch", False):
        return "an older copy of this pack's layout patch"
    if getattr(init, "__name__", "") == "_patched_init":
        return "a copy of this pack's layout patch (a fork, or an older version)"
    if hasattr(init, "__wrapped__"):
        return "another pack (%s)" % getattr(init, "__module__", "unknown")
    home = getattr(mm.PackedLayout, "__module__", None)
    where = getattr(init, "__module__", None)
    if home and where and where != home:
        return "another pack (%s)" % where
    return None


def ensure(context=""):
    """Run the checks once. Raises with a usable message if they fail."""
    global _checked
    if _checked is True:
        return
    if isinstance(_checked, str):
        raise RuntimeError(_checked)

    if not hasattr(mm, "PackedLayout") or not hasattr(mm, "FRAME_RESCALE"):
        _checked = (
            "MiniMaxH3 Motion Context: ComfyUI's MiniMax H3 model module is "
            "missing PackedLayout or FRAME_RESCALE. This pack cannot "
            "run against it.")
        raise RuntimeError(_checked)

    import inspect
    try:
        params = inspect.signature(mm.PackedLayout.__init__).parameters
    except (TypeError, ValueError):
        params = {}
    wrapped = _wrapper_origin()
    if wrapped:
        _LOG.warning(
            "MiniMaxH3 Motion Context: H3's layout constructor has been wrapped by "
            "%s. This pack does not patch it and does not need it patched. "
            "Checking whether anchors still land correctly; if they do, "
            "nothing needs doing, but keeping only one H3 chaining pack "
            "installed avoids surprises.", wrapped)
    elif "frame_count" in params:
        _checked = (
            "MiniMaxH3 Motion Context: this ComfyUI still has the older H3 layout, "
            "which rejects keyframe anchors other than the first and last "
            "frame. This version of the pack no longer works around that. "
            "Update ComfyUI, or use ComfyUI-H3-Motion-Context pack version 0.3.1, "
            "which runs on both layouts. (Detected from PackedLayout.__init__ still "
            "taking frame_count, not from a version number.)")
        raise RuntimeError(_checked)

    try:
        _check()
    except Exception as exc:
        _checked = (
            "MiniMaxH3 Motion Context: ComfyUI's H3 layout does not behave the way "
            "this pack needs%s. %s. Refusing to run rather than rendering a "
            "join at the wrong instant.%s"
            % (" (%s)" % context if context else "", exc,
               (" H3's layout constructor is wrapped by %s, which is the "
                "first thing to rule out: disable it and try again."
                % wrapped) if wrapped else
               " This is an upstream ComfyUI change; please open an issue "
               "with your ComfyUI version."))
        raise RuntimeError(_checked) from exc

    _checked = True
    _LOG.info(
        "MiniMaxH3 Motion Context: ComfyUI H3 layout checks passed, anchors "
        "and pinned audio will land where intended")


def is_checked():
    return _checked is True
