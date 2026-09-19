import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Most preferred FIRST in this tuple. Note the reversed() below: two
# sys.path.insert(0, ...) calls in listed order put the LAST one at index 0, so
# the plain loop this replaces made the third_party fallback win. That was not
# cosmetic - third_party/ComfyUI is 0.33.0 while the machine runs 0.36.0, so
# the whole suite was checking this pack against an API three minor versions
# behind the one it actually loads into. `MiniMaxH3FunControlPatch` does not
# exist in 0.33.0 at all.
_COMFY_CANDIDATES = (
    ROOT.parent.parent / "ComfyUI_windows_portable" / "ComfyUI",
    ROOT.parent / "third_party" / "ComfyUI",
)
for candidate in reversed(_COMFY_CANDIDATES):
    p = str(candidate)
    if (candidate / "comfy_api").is_dir() and p not in sys.path:
        sys.path.insert(0, p)

# Materialise spectrum_h3 from third_party before any test imports spectrum nodes.
try:
    from mmx_utils.spectrum_h3._copy_from_third_party import ensure_ported

    ensure_ported()
except Exception:
    pass
