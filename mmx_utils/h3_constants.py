"""MiniMax H3 constants — re-exported from ComfyUI core when available.

The fallback literals below are not guesses; each was read out of ComfyUI core:
  CANVAS_MULTIPLE / BASE_SHORT_EDGE / MAX_PIXELS / FPS / AUDIO_LATENT_FPS
      comfy_extras/nodes_minimax_h3.py:27-31
  align_frame_count   nodes_minimax_h3.py:33-35   (while n % 17 != 5: n += 1)
  video_latent_t      nodes_minimax_h3.py:40-41
  FRAME_PER_TOKEN     comfy/ldm/minimax/model.py:30

Catch `Exception`, not `ImportError`. Importing `comfy_extras.nodes_minimax_h3`
transitively imports `comfy.ldm.modules.attention`, which calls into the installed
`comfy_kitchen` build. When the checked-out core and the installed environment are
at different versions that raises `AttributeError`, not `ImportError`, and a
narrower except leaves the whole pack uncollectable. Observed on this machine:
`AttributeError: module 'comfy_kitchen' has no attribute 'int8_attention_is_available'`.
"""

from __future__ import annotations

try:
    from comfy_extras.nodes_minimax_h3 import (  # type: ignore
        AUDIO_LATENT_FPS,
        BASE_SHORT_EDGE,
        CANVAS_MULTIPLE,
        FPS,
        MAX_PIXELS,
        adapt_canvas,
        align_frame_count,
        video_latent_t,
    )
except Exception:
    import math

    CANVAS_MULTIPLE = 32
    BASE_SHORT_EDGE = 768
    MAX_PIXELS = 768 * 1344
    FPS = 24
    AUDIO_LATENT_FPS = 40

    def align_frame_count(n: int) -> int:
        while n % 17 != 5:
            n += 1
        return n

    def video_latent_t(frame_count: int) -> int:
        return 2 if frame_count <= 5 else ((frame_count - 5) // 17) * 5 + 2

    def adapt_canvas(width: int, height: int) -> tuple[int, int]:
        ratio = width / height
        if ratio >= 1.0:
            nom_w, nom_h = BASE_SHORT_EDGE * ratio, BASE_SHORT_EDGE
        else:
            nom_w, nom_h = BASE_SHORT_EDGE, BASE_SHORT_EDGE / ratio
        if nom_w * nom_h > MAX_PIXELS:
            s = math.sqrt(MAX_PIXELS / (nom_w * nom_h))
            nom_w, nom_h = nom_w * s, nom_h * s
        return (
            max(CANVAS_MULTIPLE, round(nom_w / CANVAS_MULTIPLE) * CANVAS_MULTIPLE),
            max(CANVAS_MULTIPLE, round(nom_h / CANVAS_MULTIPLE) * CANVAS_MULTIPLE),
        )

try:
    from comfy.ldm.minimax.model import FRAME_PER_TOKEN  # type: ignore
except Exception:
    FRAME_PER_TOKEN = (1, 4, 4, 4, 4)
