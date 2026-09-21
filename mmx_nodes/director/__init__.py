"""MiniMax H3 Director — storyboard conditioning for H3.

PORTED FROM: ComfyUI-MiniMaxH3-Director by CGlide (GPL-3.0). This pack is
GPL-3.0, so the licences agree; see CREDITS.md.

Five nodes:

  Director          a whole storyboard -> one conditioning + joint AV latent.
                    Shot markers ([0s-1.5s] ...) in the prompt, keyframes as
                    fl2va first/last, references as <Picture i> / <Video k> /
                    <Audio j>, and Retake Mode to regenerate a marked range of
                    an existing video between its own surrounding frames.
  PreviewOverride   swap a segment's preview without re-running the timeline.
  RetakeStitch      put a retaken range back into the base video.
  EnhancePrompt     expand a shot description into H3's prompt format.
  SaveLastFrame     the final frame, for chaining.

WHAT WAS NOT TAKEN, and why:

  * `minimax_chain.py` (MiniMaxH3DirectorChain) is present but NOT registered,
    exactly as upstream leaves it. Upstream's own note: the backend works,
    there is no usable way to give it a timeline, so it is withdrawn rather
    than shipped as a feature nobody can operate. That reasoning holds here.

  * The canvas timeline front-end (`js/`) was deliberately not copied. It is a
    fork of the LTX Director's canvas editor, and this workspace already has a
    better one - WanDirector's seven-track DOM timeline. A second canvas
    implementation would be the weaker of the two and would have to be
    maintained alongside it.

    The Director works without it: an empty `timeline_data` falls back to
    `global_prompt` and produces a valid plan, so the node ships useful as a
    prompt-driven director. The timeline is an enhancement on top, and it is
    on the UI pass to wire the existing DOM timeline to it rather than to
    reintroduce a canvas one.

Node ids are namespaced to this pack (MiniMaxH3_Director, not the upstream
MiniMaxH3DirectorCS) so that installing the original alongside this pack is a
visible duplicate in the menu rather than a silent collision where whichever
loads last wins.
"""

from __future__ import annotations

import logging

log = logging.getLogger(__name__)

_IMPORT_ERROR: Exception | None = None
try:
    from .minimax_director import MiniMaxH3Director
    from .minimax_enhance import MiniMaxH3EnhancePrompt
    from .minimax_lastframe import MiniMaxH3SaveLastFrame
    from .minimax_preview import MiniMaxH3PreviewOverride
    from .minimax_retake import MiniMaxH3RetakeStitch
except Exception as _e:  # noqa: BLE001
    # These reach into comfy_extras for ComfyUI's own H3 nodes. On a build
    # without them the import fails, and an unguarded failure here would take
    # the whole pack out of /object_info rather than just these five nodes.
    _IMPORT_ERROR = _e
    MiniMaxH3Director = None  # type: ignore[assignment]
    MiniMaxH3EnhancePrompt = None  # type: ignore[assignment]
    MiniMaxH3SaveLastFrame = None  # type: ignore[assignment]
    MiniMaxH3PreviewOverride = None  # type: ignore[assignment]
    MiniMaxH3RetakeStitch = None  # type: ignore[assignment]
    log.warning("[MiniMaxSuite] Director nodes unavailable: %s", _e)


# The all-in-one Director (MIT, separate upstream) is a CLASSIC-API node, and
# this pack registers through the V3 entrypoint. It is adapted rather than
# registered alongside: ComfyUI's loader takes the V1 branch and RETURNS if a
# pack exports NODE_CLASS_MAPPINGS, so exporting one mapping to register one
# node would silently unregister the other ninety-seven.
_ALLINONE_ERROR: Exception | None = None
try:
    from .allinone import MuseMinimaxDirector as _AllInOneV1
    from .v1_adapter import adapt as _adapt

    MiniMaxH3_DirectorAllInOne = _adapt(
        _AllInOneV1,
        node_id="MiniMaxH3_DirectorAllInOne",
        display_name="H3 Director (All-in-One, chunked)",
        category="MiniMax H3/Director",
        description=(
            "Runs the whole H3 pipeline internally and hands back finished "
            "frames and audio. The reason to have this as well as the Director "
            "next to it: H3 is only reliable to about 15s per call, and this "
            "CHUNKS - anything longer is split, with each continuation chunk "
            "seeded from the previous chunk's own last frames and last seconds "
            "of audio so it continues rather than cuts."
        ),
    )
except Exception as _ai_exc:  # noqa: BLE001
    _ALLINONE_ERROR = _ai_exc
    MiniMaxH3_DirectorAllInOne = None  # type: ignore[assignment]
    log.warning("[MiniMaxSuite] All-in-One Director unavailable: %s", _ai_exc)


def deferred_routes() -> list[str]:
    """Routes that could not be registered because the server was not up yet.

    Non-empty means this package was imported before PromptServer existed. The
    nodes still work; only the timeline front-end's endpoints are missing, and
    this pack does not ship that front-end yet.
    """
    if _IMPORT_ERROR is not None:
        return []
    from .minimax_media import _ROUTES_DEFERRED
    return list(_ROUTES_DEFERRED)


def director_nodes() -> list:
    """The five registerable classes, or an empty list if they did not import."""
    if _IMPORT_ERROR is not None:
        return []
    nodes = [MiniMaxH3Director, MiniMaxH3PreviewOverride, MiniMaxH3RetakeStitch,
             MiniMaxH3EnhancePrompt, MiniMaxH3SaveLastFrame]
    if MiniMaxH3_DirectorAllInOne is not None:
        nodes.append(MiniMaxH3_DirectorAllInOne)
    return nodes


__all__ = [
    "MiniMaxH3Director",
    "MiniMaxH3EnhancePrompt",
    "MiniMaxH3PreviewOverride",
    "MiniMaxH3RetakeStitch",
    "MiniMaxH3SaveLastFrame",
    "deferred_routes",
    "MiniMaxH3_DirectorAllInOne",
    "director_nodes",
]
