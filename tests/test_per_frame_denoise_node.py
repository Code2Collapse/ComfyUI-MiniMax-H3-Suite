import torch

from mmx_nodes.per_frame_denoise import MiniMaxH3_PerFrameDenoise
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


def test_per_frame_denoise_sets_noise_mask():
    video = torch.zeros(1, 16, 7, 4, 4)
    audio = torch.zeros(1, 8, 100)
    av = {
        "samples": _Nested([video, audio]),
    }
    out = MiniMaxH3_PerFrameDenoise.execute(
        av,
        _transform(),
        1.0,
        0.3,
        "absolute_px",
        30.0,
        120.0,
        1.0,
        1,
    )
    result, report = out[0], out[1]
    assert "noise_mask" in result
    assert "per-frame denoise" in report
