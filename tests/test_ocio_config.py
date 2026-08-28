import pytest

pytest.importorskip("PyOpenColorIO")

from mmx_utils.ocio_config import (
    ACES_STUDIO_COLORSPACES,
    OCIOColorSpaceError,
    config_source_note,
    list_colorspaces,
    list_displays,
    validate_colorspace,
)


def test_list_colorspaces_non_empty():
    names = list_colorspaces()
    assert len(names) >= len(ACES_STUDIO_COLORSPACES) // 2


def test_config_source_note_is_string():
    note = config_source_note()
    assert isinstance(note, str)
    assert "OCIO" in note or "FALLBACK" in note


def test_validate_unknown_colorspace_raises():
    with pytest.raises(OCIOColorSpaceError, match="Unknown OCIO colorspace"):
        validate_colorspace("__not_a_real_colorspace__")


def test_fallback_lists_when_config_missing(monkeypatch):
    monkeypatch.setattr("mmx_utils.ocio_config.OCIO_CONFIG", None)
    monkeypatch.setattr("mmx_utils.ocio_config.get_ocio_config", lambda: None)
    names = list_colorspaces()
    assert names == list(ACES_STUDIO_COLORSPACES)
    displays = list_displays()
    assert len(displays) >= 1
    note = config_source_note()
    assert "FALLBACK" in note
