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
