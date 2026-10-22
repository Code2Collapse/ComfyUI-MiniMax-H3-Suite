"""H3 Seam Correction (ported from supElement/ComfyUI_MinimaxH3_AutoContext, Apache-2.0): an exposure step between
two segments of the same shot is found and flattened; without PySceneDetect the report says real cuts are not
protected."""
from __future__ import annotations

import torch

from mmx_nodes.seam_correction import correct_seams


def _film(step=0.10, seam_at=30, n=60):
    frames = torch.full((n, 72, 128, 3), 0.40)
    for i in range(n):
        x = 8 + (i * 2) % 100
        frames[i, 20:44, x:x + 16, :] = 0.85                    # a moving subject
    frames[seam_at:] = (frames[seam_at:] + step).clamp(0, 1)    # segment 2 came out brighter
    return frames


def _step(frames, at, w=3):
    luma = (frames[..., :3] * torch.tensor([0.2126, 0.7152, 0.0722])).sum(-1).mean(dim=(1, 2))
    return float(luma[at:at + w].mean() - luma[at - w:at].mean())


def test_an_exposure_step_between_segments_is_flattened():
    film = _film()
    before = _step(film, 30)
    out, report = correct_seams(film, "medium", "off", False, 0.3, 2, False, cut_detection=False)
    after = _step(out, 30)
    assert before > 0.09 and abs(after) < 0.3 * before, (before, after, report)
    assert out.shape == film.shape and float(out.min()) >= 0.0 and float(out.max()) <= 1.0


def test_off_means_unchanged_and_the_fallback_is_reported():
    film = _film()
    out, report = correct_seams(film, "off", "off", False, cut_detection=False)
    assert torch.equal(out, film) and "unchanged" in report
    _, report2 = correct_seams(film, "medium", "off", False, use_gpu=False, cut_detection=True)
    assert "PySceneDetect" in report2 or "boundary" in report2


def test_a_clean_film_is_left_alone():
    film = _film(step=0.0)
    out, report = correct_seams(film, "medium", "off", False, use_gpu=False, cut_detection=False)
    assert float((out - film).abs().max()) < 1e-6 and "no seam found" in report
