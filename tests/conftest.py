import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

for candidate in (
    ROOT.parent.parent / "ComfyUI_windows_portable" / "ComfyUI",
    ROOT.parent / "third_party" / "ComfyUI",
):
    p = str(candidate)
    if (candidate / "comfy_api").is_dir() and p not in sys.path:
        sys.path.insert(0, p)

# Materialise spectrum_h3 from third_party before any test imports spectrum nodes.
try:
    from mmx_utils.spectrum_h3._copy_from_third_party import ensure_ported

    ensure_ported()
except Exception:
    pass
