"""MiniMaxH3_ImageAndReferenceToVideo against core's own MiniMaxH3ReferenceToVideo (A16).

Found by the A16 audit: the node compared core's CANVAS_MULTIPLE (32 on ComfyUI 0.39) with its own constant (16,
which also served as the latent stride) and refused EVERY run on current ComfyUI. These tests run the node and core's
node with the same fake CLIP / VAE and compare the reference blocks the DiT gets.
"""

from __future__ import annotations

import pytest
import torch

core_nodes = pytest.importorskip("comfy_extras.nodes_minimax_h3", reason="needs ComfyUI with MiniMax H3")

from mmx_nodes.image_and_reference import REF_VIDEO_FITS, MiniMaxH3_ImageAndReferenceToVideo  # noqa: E402


class FakeClip:
    def __init__(self):
        self.items = None

    def tokenize(self, prompt, images=None, minimax_ref_items=None):
        self.items = minimax_ref_items
        return {"prompt": prompt}

    def encode_from_tokens_scheduled(self, tokens):
        return [[torch.zeros(1, 4, 8), {}]]


class FakeVAE:
    def encode(self, images):
        n, h, w = images.shape[:3]
        t = 1 if n == 1 else max(1, (n - 5) // 17 * 5 + 2)
        return torch.full((1, 24, t, h // 16, w // 16), float(w))


def _ours(**kw):
    out = MiniMaxH3_ImageAndReferenceToVideo.execute(FakeClip(), FakeVAE(), None, "p", **kw)
    return out.args if hasattr(out, "args") else out


def _core(**kw):
    out = core_nodes.MiniMaxH3ReferenceToVideo.execute(FakeClip(), "p", vae=FakeVAE(), **kw)
    return out.args if hasattr(out, "args") else out


def _blocks(cond):
    return cond[0][1].get("minimax_refs") or []


def test_runs_on_current_comfyui():
    cond, latent, report = _ours(width=1344, height=768, length=124)
    assert latent["samples"].unbind()[0].shape[-2:] == (48, 84)
    assert "Geometry" not in report                      # nothing to say without reference videos


def test_reference_blocks_match_cores_reference_to_video():
    img = torch.rand(1, 700, 1000, 3)
    vid = torch.rand(30, 1080, 1920, 3)
    ours = _blocks(_ours(width=1344, height=768, length=124, ref_images={"ref_image_1": img},
                         ref_videos={"ref_video_1": vid})[0])
    core = _blocks(_core(width=1344, height=768, length=124, ref_images={"ref_image_0": img},
                         ref_videos={"ref_video_0": vid})[0])
    assert len(ours) == len(core) == 2
    for a, b in zip(ours, core):
        for key in ("kind", "latent_h", "latent_w", "latent_t"):
            assert a.get(key) == b.get(key), key
        assert a["latent"].shape == b["latent"].shape
        assert (a["latent_h"], a["latent_w"]) == tuple(a["latent"].shape[-2:])   # declared == encoded


def test_default_fit_is_core_canvas_and_reports_the_offset():
    vid = torch.rand(30, 1440, 1440, 3)
    _, _, report = _ours(width=1440, height=1440, length=124, ref_videos={"ref_video_1": vid})
    assert REF_VIDEO_FITS[0] == "core canvas"
    assert "-14.0 px horizontally" in report and "target grid (zero offset)" in report


def test_target_grid_puts_the_reference_video_on_the_output_grid():
    vid = torch.rand(30, 1440, 1440, 3)
    cond, _, report = _ours(width=1440, height=1440, length=124, ref_videos={"ref_video_1": vid},
                            ref_video_fit=REF_VIDEO_FITS[1])
    blk = _blocks(cond)[0]
    assert (blk["latent_h"], blk["latent_w"]) == (90, 90) and tuple(blk["latent"].shape[-2:]) == (90, 90)
    assert "offset 0" in report


def test_target_grid_cover_crops_a_different_aspect_and_says_so():
    vid = torch.rand(30, 1080, 1920, 3)
    cond, _, report = _ours(width=1344, height=1536, length=124, ref_videos={"ref_video_1": vid},
                            ref_video_fit=REF_VIDEO_FITS[1])
    blk = _blocks(cond)[0]
    assert (blk["latent_h"], blk["latent_w"]) == (96, 84)
    assert "cover-cropped" in report
