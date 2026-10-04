"""Static rules for MiniMax H3 web widget modules (web/w*.js)."""

from __future__ import annotations

import re
from pathlib import Path

PACK = Path(__file__).resolve().parents[1]
WEB = PACK / "web"

ALLOWED_IMPORTS = frozenset({
    "./shared.js",
    "./c2c_ui/index.js",
    "../../scripts/app.js",
})

WIDGET_GLOB = "w*.js"


def _widget_files() -> list[Path]:
    return sorted(WEB.glob(WIDGET_GLOB))


def _import_specs(src: str) -> list[str]:
    specs: list[str] = []
    for m in re.finditer(
        r'(?:import\s+[^;]*?\s+from\s+|import\s*)["\']([^"\']+)["\']',
        src,
    ):
        specs.append(m.group(1))
    return specs


def test_widget_imports_only_allowed_modules():
    for path in _widget_files():
        specs = _import_specs(path.read_text(encoding="utf-8"))
        bad = [s for s in specs if s not in ALLOWED_IMPORTS]
        assert not bad, f"{path.name}: disallowed imports {bad}"


def test_widgets_use_mount_panel():
    for path in _widget_files():
        src = path.read_text(encoding="utf-8")
        assert "mountPanel(" in src, f"{path.name}: must mount via mountPanel"


def test_widgets_no_start_loading_loop():
    for path in _widget_files():
        src = path.read_text(encoding="utf-8")
        assert "startLoadingLoop(" not in src, f"{path.name}: must not use startLoadingLoop"


def test_widgets_no_self_rescheduling_raf_loops():
    loop_pat = re.compile(
        r"requestAnimationFrame\s*\(\s*(?:function\s*)?\(?\s*(\w+)",
    )
    for path in _widget_files():
        src = path.read_text(encoding="utf-8")
        for m in loop_pat.finditer(src):
            name = m.group(1)
            # A handler that re-schedules itself is a per-frame loop.
            if re.search(rf"requestAnimationFrame\s*\(\s*{name}\b", src):
                raise AssertionError(
                    f"{path.name}: self-rescheduling rAF loop via {name}",
                )


def test_widgets_no_var_in_canvas_colours():
    colour_assign = re.compile(
        r"(?:fillStyle|strokeStyle)\s*=\s*[^;]*var\s*\(",
    )
    for path in _widget_files():
        src = path.read_text(encoding="utf-8")
        assert not colour_assign.search(src), (
            f"{path.name}: canvas colours must not use var() — resolve at draw time"
        )


def test_widgets_no_text_overflow_ellipsis():
    for path in _widget_files():
        src = path.read_text(encoding="utf-8")
        assert "text-overflow" not in src, f"{path.name}: must not ellipsize text"
        assert "ellipsis" not in src.lower(), f"{path.name}: must not ellipsize text"
