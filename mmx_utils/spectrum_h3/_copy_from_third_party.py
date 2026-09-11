"""One-shot port helper — copies upstream spectrum_h3 into this package."""

from __future__ import annotations

import shutil
from pathlib import Path

_SRC = (
    Path(__file__).resolve().parents[3]
    / "third_party"
    / "ComfyUI-Spectrum-MiniMax-H3"
    / "comfyui_spectrum_h3"
)
_DST = Path(__file__).resolve().parent


def ensure_ported() -> None:
    # config.py is committed as a stub; require the full upstream tree (nodes.py).
    if (_DST / "nodes.py").is_file() and (_DST / "runtime.py").is_file():
        return
    if not _SRC.is_dir():
        # third_party/ is gitignored, so it is ABSENT on the Linux deploy and on
        # any fresh clone. The ported modules are committed, so reaching here at
        # all means the port is incomplete -- report it, never raise: this runs
        # from tests/conftest.py at collection time and an exception there aborts
        # the ENTIRE suite rather than one file.
        import logging
        logging.getLogger(__name__).warning(
            "Spectrum H3 modules are not present and the upstream copy at %s is "
            "unavailable; Spectrum nodes will not load.", _SRC)
        return
    skip = {"__init__.py", "_copy_from_third_party.py"}
    for item in _SRC.iterdir():
        if item.name in skip:
            continue
        target = _DST / item.name
        if item.is_dir():
            shutil.copytree(item, target, dirs_exist_ok=True)
        else:
            shutil.copy2(item, target)
    replacements = (
        (
            "generic_correction_worker.py",
            '_PACKAGE_NAME = "comfyui_spectrum_h3"',
            '_PACKAGE_NAME = "mmx_utils.spectrum_h3"',
        ),
    )
    for filename, old, new in replacements:
        target = _DST / filename
        if target.is_file():
            text = target.read_text(encoding="utf-8")
            if old in text:
                target.write_text(text.replace(old, new), encoding="utf-8")


if __name__ == "__main__":
    ensure_ported()
    print(f"ported spectrum_h3 -> {_DST}")
