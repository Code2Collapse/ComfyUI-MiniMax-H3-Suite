import numpy as np
import pytest
import torch

from mmx_utils.per_frame_denoise import (
    face_heights_from_transform,
    merge_per_frame_video_mask,
    scale_video_noise_mask,
    strength_curve_from_faces,
)
from mmx_utils.transform_types import H3Transform


class _Nested:
    def __init__(self, members):
        self._members = members

    def unbind(self):
        return self._members


def _transform(n=5):
    boxes = tuple((0.0, 0.0, 100.0, float(30 + i * 20)) for i in range(n))
    return H3Transform(
        boxes=boxes,
        canvas=(768, 768),
        src_size=(1920, 1080),
        frames=n,
        weights=tuple(1.0 for _ in range(n)),
        detected=tuple(True for _ in range(n)),
        subject_rect=None,
        crop_factor=3.0,
        planner_mode="tv_lp",
    )


def test_face_heights_from_transform():
    t = _transform()
    face = face_heights_from_transform(t)
    assert face.shape[0] == 5
    assert face[0] == 10.0


def test_strength_curve_small_vs_large():
    face = np.array([20.0, 200.0])
    s = strength_curve_from_faces(
        face,
        strength_small_face=1.0,
        strength_large_face=0.2,
        face_px_small=30.0,
        face_px_large=120.0,
        gamma=1.0,
        smooth_frames=1,
        scale_mode="absolute_px",
    )
    assert s[0] > s[1]


def test_scale_video_noise_mask_shape():
    video = torch.zeros(1, 16, 7, 4, 4)
    strength = np.ones(5)
    vmask, _ = scale_video_noise_mask(video, None, strength)
    assert vmask.shape == video.shape


def test_merge_per_frame_video_mask_multiplies_existing():
    video = torch.ones(1, 16, 7, 2, 2)
    audio = torch.zeros(1, 8, 10)
    base_mask = torch.full((1, 16, 7, 2, 2), 0.5)
    av = {"noise_mask": _Nested([base_mask, audio])}
    vmask, _ = scale_video_noise_mask(video, base_mask, np.full(5, 0.5))
    out, _ = merge_per_frame_video_mask(av, vmask, video, audio)
    merged = out["noise_mask"].unbind()[0]
    assert float(merged.mean()) == pytest.approx(0.125, rel=1e-3)
