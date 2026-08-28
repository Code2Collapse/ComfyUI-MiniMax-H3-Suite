# MIT License — nuke-nodes-comfyui
# Copyright (c) 2025 Sumit Chatterjee
# PORTED FROM: nuke-nodes-comfyui :: colorspace_nodes.py @ HEAD

"""OCIO config discovery — live enumeration with documented fallback lists."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import PyOpenColorIO as OCIO

_LOG = logging.getLogger(__name__)

OCIO_AVAILABLE = False
OCIO_VERSION = "0.0.0"
OCIO_CONFIG = None

# Hardcoded ACES 2.0 Studio colorspaces (fallback when config cannot be read).
ACES_STUDIO_COLORSPACES: tuple[str, ...] = (
    "ACES2065-1",
    "ACEScc",
    "ACEScct",
    "ACEScg",
    "ADX10",
    "ADX16",
    "ARRI LogC3 (EI800)",
    "ARRI LogC4",
    "Apple Log",
    "BMDFilm WideGamut Gen5",
    "Camera Rec.709",
    "CanonLog2 CinemaGamut D55",
    "CanonLog3 CinemaGamut D55",
    "D-Log D-Gamut",
    "DaVinci Intermediate WideGamut",
    "Display P3 - Display",
    "Display P3 HDR - Display",
    "Gamma 1.8 Encoded Rec.709",
    "Gamma 2.2 Encoded AP1",
    "Gamma 2.2 Encoded AdobeRGB",
    "Gamma 2.2 Encoded Rec.709",
    "Gamma 2.2 Rec.709 - Display",
    "Gamma 2.4 Encoded Rec.709",
    "Linear ARRI Wide Gamut 3",
    "Linear ARRI Wide Gamut 4",
    "Linear AdobeRGB",
    "Linear BMD WideGamut Gen5",
    "Linear CinemaGamut D55",
    "Linear D-Gamut",
    "Linear DaVinci WideGamut",
    "Linear P3-D65",
    "Linear REDWideGamutRGB",
    "Linear Rec.2020",
    "Linear Rec.709 (sRGB)",
    "Linear S-Gamut3",
    "Linear S-Gamut3.Cine",
    "Linear V-Gamut",
    "Linear Venice S-Gamut3",
    "Linear Venice S-Gamut3.Cine",
    "Log3G10 REDWideGamutRGB",
    "P3-D65 - Display",
    "Raw",
    "Rec.1886 Rec.709 - Display",
    "Rec.2100-HLG - Display",
    "Rec.2100-PQ - Display",
    "S-Log3 S-Gamut3",
    "S-Log3 S-Gamut3.Cine",
    "S-Log3 Venice S-Gamut3",
    "S-Log3 Venice S-Gamut3.Cine",
    "ST2084-P3-D65 - Display",
    "V-Log V-Gamut",
    "sRGB - Display",
    "sRGB Encoded AP1",
    "sRGB Encoded P3-D65",
    "sRGB Encoded Rec.709 (sRGB)",
)

FALLBACK_DISPLAYS: tuple[str, ...] = (
    "sRGB - Display",
    "Rec.1886 Rec.709 - Display",
    "P3-D65 - Display",
    "Rec.2100-PQ - Display",
)

# Enumerated from the installed builtin config studio-config-v4.0.0_aces-v2.0_ocio-v2.5.
# The earlier value "ACES 2.0 - SDR Video" did NOT exist in any display and made every
# display transform fail with "the display 'sRGB - Display' does not have view ...".
FALLBACK_VIEWS: tuple[str, ...] = ("ACES 2.0 - SDR 100 nits (Rec.709)", "Un-tone-mapped", "Raw")

STUDIO_CONFIGS: tuple[str, ...] = (
    "studio-config-v4.0.0_aces-v2.0_ocio-v2.5",
    "studio-config-v2.2.0_aces-v1.3_ocio-v2.4",
    "studio-config-v2.1.0_aces-v1.3_ocio-v2.3",
    "studio-config-v1.0.0_aces-v1.3_ocio-v2.1",
)


class OCIOUnavailableError(RuntimeError):
    """PyOpenColorIO is not installed."""


class OCIOConfigUnavailableError(RuntimeError):
    """No OCIO config could be loaded."""


class OCIOColorSpaceError(ValueError):
    """Named colorspace is not in the active OCIO config."""


def _require_ocio():
    if not OCIO_AVAILABLE:
        raise OCIOUnavailableError(
            "PyOpenColorIO is not installed. Install opencolorio (2.5+ recommended)."
        )


def load_studio_config(force: bool = False):
    """Load ACES Studio Config; cache in OCIO_CONFIG."""
    global OCIO_CONFIG
    if OCIO_CONFIG is not None and not force:
        return OCIO_CONFIG
    _require_ocio()
    import PyOpenColorIO as OCIO

    for config_name in STUDIO_CONFIGS:
        try:
            config = OCIO.Config.CreateFromBuiltinConfig(config_name)
            if config:
                OCIO_CONFIG = config
                _LOG.info("MiniMax OCIO: loaded builtin config %s", config_name)
                return config
        except Exception as exc:
            _LOG.warning("MiniMax OCIO: could not load %s: %s", config_name, exc)
    OCIO_CONFIG = None
    return None


def get_ocio_config():
    if not OCIO_AVAILABLE:
        return None
    if OCIO_CONFIG is None:
        load_studio_config()
    return OCIO_CONFIG


def config_source_note() -> str:
    cfg = get_ocio_config()
    if cfg is not None:
        try:
            return f"OCIO {OCIO_VERSION} — live config ({len(list_colorspaces())} colorspaces)"
        except Exception:
            return f"OCIO {OCIO_VERSION} — live config loaded"
    return (
        f"OCIO {OCIO_VERSION if OCIO_AVAILABLE else 'unavailable'} — "
        f"FALLBACK colorspace list ({len(ACES_STUDIO_COLORSPACES)} names, no live config)"
    )


def list_colorspaces() -> list[str]:
    cfg = get_ocio_config()
    if cfg is None:
        return list(ACES_STUDIO_COLORSPACES)
    try:
        names = [cs.getName() for cs in cfg.getColorSpaces()]
        return names if names else list(ACES_STUDIO_COLORSPACES)
    except Exception:
        return list(ACES_STUDIO_COLORSPACES)


def list_displays() -> list[str]:
    cfg = get_ocio_config()
    if cfg is None:
        return list(FALLBACK_DISPLAYS)
    try:
        names = list(cfg.getDisplays())
        return names if names else list(FALLBACK_DISPLAYS)
    except Exception:
        return list(FALLBACK_DISPLAYS)


def list_views(display: str) -> list[str]:
    cfg = get_ocio_config()
    if cfg is None:
        return list(FALLBACK_VIEWS)
    try:
        names = list(cfg.getViews(display))
        return names if names else list(FALLBACK_VIEWS)
    except Exception:
        return list(FALLBACK_VIEWS)


def validate_colorspace(name: str, *, config=None) -> str:
    """Return *name* if present in config; raise OCIOColorSpaceError otherwise."""
    _require_ocio()
    cfg = config or get_ocio_config()
    if cfg is None:
        if name in ACES_STUDIO_COLORSPACES:
            return name
        raise OCIOConfigUnavailableError(
            "No OCIO config loaded — cannot validate colorspace "
            f"{name!r}. Upgrade opencolorio or install ACES builtin configs."
        )
    # PyOpenColorIO's Config.getColorSpace() RETURNS None for an unknown name — it does
    # not raise. Relying on the exception alone silently accepted any bogus string and
    # handed it straight to the transform, where it failed much later with a far less
    # useful message. Check the return value explicitly.
    try:
        cs = cfg.getColorSpace(name)
    except Exception as exc:  # a genuinely broken config, not an unknown name
        raise OCIOColorSpaceError(
            f"Unknown OCIO colorspace {name!r} — not in the active config."
        ) from exc
    if cs is None:
        raise OCIOColorSpaceError(
            f"Unknown OCIO colorspace {name!r} — not in the active config "
            f"({cfg.getName()!r}). Available names come from list_colorspaces()."
        )
    return name


try:
    import PyOpenColorIO as OCIO

    OCIO_AVAILABLE = True
    OCIO_VERSION = OCIO.GetVersion()
    load_studio_config()
except ImportError:
    OCIO = None  # type: ignore[misc, assignment]
