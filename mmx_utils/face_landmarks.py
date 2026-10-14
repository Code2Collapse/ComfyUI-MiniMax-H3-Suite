"""Per-frame face landmarks for the performance scores (C2C L8.25).

mediapipe FaceMesh with refine_landmarks (478 points incl. irises). Its models ship inside the mediapipe wheel, so
nothing is downloaded; when mediapipe is not installed the caller gets (None, reason) and reports the face scores as
not measured. Video mode (static_image_mode=False) tracks between frames like a real clip.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


def available() -> tuple[bool, str]:
    try:
        import mediapipe as mp  # noqa: F401
        _ = mp.solutions.face_mesh.FaceMesh
    except Exception as e:  # noqa: BLE001 - any import problem means "not measured", never a crash
        return False, f"mediapipe FaceMesh not available ({type(e).__name__}: {e})"
    return True, ""


def landmarks(frames: Sequence[np.ndarray], *, video: bool = True,
              min_confidence: float = 0.5) -> tuple[list[np.ndarray | None] | None, str]:
    """frames: [H,W,3] uint8 RGB (or float 0-1). Returns ([478x3 or None per frame], "") or (None, reason)."""
    ok, why = available()
    if not ok:
        return None, why
    import mediapipe as mp

    out: list[np.ndarray | None] = []
    with mp.solutions.face_mesh.FaceMesh(static_image_mode=not video, max_num_faces=1, refine_landmarks=True,
                                         min_detection_confidence=min_confidence,
                                         min_tracking_confidence=min_confidence) as fm:
        for f in frames:
            img = np.asarray(f)
            if img.dtype != np.uint8:
                img = (np.clip(img, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
            r = fm.process(np.ascontiguousarray(img[..., :3]))
            if not r.multi_face_landmarks:
                out.append(None)
                continue
            lm = r.multi_face_landmarks[0].landmark
            out.append(np.array([[p.x, p.y, p.z] for p in lm], dtype=np.float32))
    return out, ""
