# Apache-2.0 — ComfyUI-WanAnimatePreprocessV2
# PORTED FROM: ComfyUI-WanAnimatePreprocessV2 :: nodes_extras/pose_detect_vitpose.py @ HEAD

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch

_LOG = logging.getLogger(__name__)

_IMG_NORM_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMG_NORM_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class VitPoseUnavailableError(RuntimeError):
    """ViTPose-H wholebody weights or onnxruntime are not available on this machine."""


def _candidate_detection_roots() -> list[Path]:
    roots: list[Path] = []
    try:
        import folder_paths

        roots.append(Path(folder_paths.models_dir) / "detection")
    except Exception:
        pass
    # `folder_paths.models_dir` above is the ONLY correct source inside a running ComfyUI.
    # The entries below are dev-box conveniences for running the tests outside ComfyUI.
    #
    # A hardcoded "D:/PROJECT/..." used to sit here. This pack is kept in sync with a Linux
    # box, where that path does not exist — an absolute drive letter is dead weight there and
    # silently contributes nothing to the search. Keep every fallback RELATIVE to this file.
    suite = Path(__file__).resolve().parents[1]
    roots.extend(
        [
            suite.parent.parent / "ComfyUI_windows_portable" / "ComfyUI" / "models" / "detection",
            suite.parent / "third_party" / "ComfyUI" / "models" / "detection",
        ]
    )
    out: list[Path] = []
    for r in roots:
        if r.is_dir() and r not in out:
            out.append(r)
    return out


def resolve_vitpose_paths() -> tuple[str | None, str | None]:
    vit_names = ("vitpose_h_wholebody_model.onnx", "vitpose-l-wholebody.onnx", "vitpose-b-apt36k.onnx")
    yolo_names = ("yolov10m.onnx", "yolo11m.onnx")
    for root in _candidate_detection_roots():
        vit = next((str(root / n) for n in vit_names if (root / n).is_file()), None)
        yolo = next((str(root / n) for n in yolo_names if (root / n).is_file()), None)
        if vit and yolo:
            return vit, yolo
        if vit:
            return vit, yolo
    return None, None


_WAN_PKG = "_minimax_wananimate_pose"


def _import_wan_pose_stack() -> Any:
    """Import WanAnimatePreprocessV2's ONNX pose stack as a real SUBpackage.

    The obvious approach — put the pack root on sys.path and `from models.onnx_models
    import ViTPose` — does not work, and fails in a way that looks like the pack is
    missing when it is present:

        ImportError: attempted relative import beyond top-level package

    Because `models/onnx_models.py:21` does `from ..pose_utils.pose2d_utils import ...`.
    Putting the pack root on sys.path makes `models` a TOP-LEVEL package, so `..` walks
    above the root and Python refuses. The pack's own modules are written to live under a
    package, so they need one.

    So synthesise the parent: register a namespace module whose __path__ is the pack root.
    `models` then resolves as `<pkg>.models` and `..pose_utils` resolves correctly.

    Deliberately does NOT exec the pack's `__init__.py`: that registers its whole node
    list and expects a live ComfyUI. We only want two ONNX classes, and running node
    registration as a side effect of a pose call would be a nasty surprise.
    """
    wan_root = Path(__file__).resolve().parents[2] / "ComfyUI-WanAnimatePreprocessV2"
    if not wan_root.is_dir():
        return None
    import importlib
    import sys
    import types

    try:
        if _WAN_PKG not in sys.modules:
            pkg = types.ModuleType(_WAN_PKG)
            pkg.__path__ = [str(wan_root)]  # type: ignore[attr-defined]
            sys.modules[_WAN_PKG] = pkg
        onnx_models = importlib.import_module(f"{_WAN_PKG}.models.onnx_models")
        pose2d = importlib.import_module(f"{_WAN_PKG}.pose_utils.pose2d_utils")
        return (
            onnx_models.Yolo,
            onnx_models.ViTPose,
            pose2d.bbox_from_detector,
            pose2d.crop,
        )
    except Exception as exc:
        _LOG.warning("Wan ViTPose stack import failed: %s", exc)
        return None


def detect_keypoints_plate(
    images: torch.Tensor,
    *,
    confidence: float = 0.3,
    temporal_smooth: bool = True,
    gap_fill: int = 3,
) -> tuple[np.ndarray, str]:
    """
    Detect wholebody keypoints in PLATE space [T,K,3].
    Raises VitPoseUnavailableError when weights/runtime are missing.
    """
    try:
        import onnxruntime  # noqa: F401
    except ImportError as exc:
        raise VitPoseUnavailableError(
            "onnxruntime is not installed — use mode=passthrough or install onnxruntime."
        ) from exc

    vit_path, yolo_path = resolve_vitpose_paths()
    if vit_path is None:
        raise VitPoseUnavailableError(
            "ViTPose-H wholebody weights not found under ComfyUI/models/detection/ "
            "(expected vitpose_h_wholebody_model.onnx + yolov10m.onnx)."
        )

    stack = _import_wan_pose_stack()
    if stack is None:
        raise VitPoseUnavailableError(
            "ComfyUI-WanAnimatePreprocessV2 pose stack is not importable — "
            "install the pack or use mode=passthrough."
        )
    Yolo, ViTPose, bbox_from_detector, crop = stack

    if images.ndim == 3:
        images = images.unsqueeze(0)
    imgs = images.detach().cpu().numpy().astype(np.float32)
    b, h, w, _ = imgs.shape
    shape = np.array([h, w])[None]

    yolo = Yolo(yolo_path, device="CPUExecutionProvider") if yolo_path else None
    pose_model = ViTPose(vit_path, device="CPUExecutionProvider")

    kp2ds: list[np.ndarray] = []
    input_resolution = (256, 192)
    for i in range(b):
        img = imgs[i]
        bbox_use = np.array([0, 0, w, h, 1.0], dtype=np.float32)
        if yolo is not None:
            import cv2

            inp = cv2.resize(img, (640, 640)).transpose(2, 0, 1)[None]
            det = yolo(inp, shape)[0]
            if isinstance(det, list) and det and isinstance(det[0], dict):
                bb = det[0]["bbox"]
                if len(bb) >= 5 and bb[4] > 0:
                    bbox_use = bb

        center, scale = bbox_from_detector(bbox_use, input_resolution, rescale=1.25)
        img_crop = crop(img, center, scale, (input_resolution[0], input_resolution[1]))[0]
        img_norm = ((img_crop - _IMG_NORM_MEAN) / _IMG_NORM_STD).transpose(2, 0, 1).astype(np.float32)
        kp = pose_model(img_norm[None], np.array(center)[None], np.array(scale)[None])
        kps = kp[0] if isinstance(kp, (list, tuple)) else kp
        if kps.ndim == 3:
            kps = kps[0]
        # Zero sub-threshold confidences.
        kps = np.asarray(kps, dtype=np.float32)
        kps[kps[:, 2] < float(confidence), 2] = 0.0
        kp2ds.append(kps)

    keypoints = np.stack(kp2ds, axis=0)
    note = f"vitpose_h plate detect frames={b} model={Path(vit_path).name}"
    if yolo_path:
        note += f" yolo={Path(yolo_path).name}"
    return keypoints, note
