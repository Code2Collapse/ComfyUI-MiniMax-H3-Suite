"""MiniMaxH3_SourceAVLatent - the masked joint AV latent with explicit geometry (A16 Part II).

The fake VAE carries core's real H3 VAE temporal attributes (comfy/ldm/minimax/vae.py: ratio_t 4, clip 17, pre-pad 3,
chunk 5, token drop 3) so the mask reduction runs on H3's real (1,4,4,4,4) grouping. No weights.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("comfy.ldm.minimax.model", reason="needs ComfyUI with MiniMax H3")
pytest.importorskip("comfy.nested_tensor", reason="needs ComfyUI")

from mmx_nodes.source_av_latent import (  # noqa: E402
    MiniMaxH3_SourceAVLatent,
    audio_mask_from_frames,
    build_source_av_latent,
    fit_audio_ticks,
)


class FakeH3VAE:
    def __init__(self):
        self.first_stage_model = SimpleNamespace(vae_ratio_t=4, tokens_chunk_size=5, frame_pre_padding=3,
                                                 token_drop=3)
        self.downscale_ratio = (lambda a: max(1, (a - 5) // 17 * 5 + 2) if a > 1 else 1, 16, 16)
        self.upscale_ratio = (lambda a: max(1, (a - 2) // 5 * 17 + 5), 16, 16)
        self.seen = None

    def spacial_compression_encode(self):
        return 16

    def encode(self, images):
        self.seen = tuple(images.shape)
        n, h, w = images.shape[:3]
        return torch.zeros(1, 24, self.downscale_ratio[0](n), h // 16, w // 16)


def _frames(n=124, h=256, w=320):
    return torch.rand(n, h, w, 3)


def test_shapes_and_audio_length():
    lat, rep = build_source_av_latent(_frames(), FakeH3VAE())
    v, a = lat["samples"].unbind()
    assert tuple(v.shape) == (1, 24, 37, 16, 20) and tuple(a.shape) == (1, 32, 2, 207)
    vm, am = lat["noise_mask"].unbind()
    assert tuple(vm.shape) == (1, 1, 37, 16, 20) and torch.all(vm == 1)       # no mask = regenerate all
    assert tuple(am.shape) == (1, 1, 2, 207) and torch.all(am == 0)           # audio 'keep'
    assert lat["mmx_geometry"]["frames"] == 124 and "silent" in rep


def test_off_grid_frame_count_is_refused_with_the_nearest_counts():
    with pytest.raises(ValueError, match="107 or pad to 124"):
        build_source_av_latent(_frames(120), FakeH3VAE())


def test_off_grid_size_is_refused_not_cropped():
    with pytest.raises(ValueError, match="not a multiple of 16"):
        build_source_av_latent(_frames(h=260, w=320), FakeH3VAE())


def test_pad_policy_pads_right_bottom_and_records_it():
    vae = FakeH3VAE()
    m = torch.zeros(124, 260, 318)
    lat, rep = build_source_av_latent(_frames(h=260, w=318), vae, mask=m, size_policy="pad right/bottom (recorded)")
    assert vae.seen[1:3] == (272, 320)                                         # nothing cut off
    g = lat["mmx_geometry"]
    assert (g["pad_right"], g["pad_bottom"]) == (2, 12) and g["source_px"] == [318, 260]
    assert "padded 318x260 -> 320x272" in rep


def test_mask_is_reduced_on_h3s_own_frame_grouping():
    # H3 rows cover frames (1,4,4,4,4) from row 0 (core FRAME_PER_TOKEN): frames 1..4 are exactly row 1.
    m = torch.zeros(124, 256, 320)
    m[1:5, 100:140, 100:140] = 1.0
    lat, _ = build_source_av_latent(_frames(), FakeH3VAE(), mask=m)
    vm = lat["noise_mask"].unbind()[0][0, 0]
    per_row = vm.amax(dim=(1, 2))
    assert [int(r) for r in torch.nonzero(per_row).flatten()] == [1] and float(per_row[1]) == 1.0
    # what the sampler does with the PIXEL mask (comfy/utils.py reshape_mask, trilinear in time):
    # row 0 (frame 0, not masked) fully regenerated, row 1 at about half strength - one row early.
    import comfy.utils
    tri = comfy.utils.reshape_mask(m[:, None], (1, 24, 37, 16, 20))[0, 0].amax(dim=(1, 2))
    assert float(tri[0]) == pytest.approx(1.0) and float(tri[1]) < 0.6


def test_audio_follow_mask_marks_the_right_ticks():
    per_frame = torch.zeros(124)
    per_frame[24:48] = 1.0                                                     # second 1..2
    am = audio_mask_from_frames(per_frame, 207, 124)
    on = torch.nonzero(am).flatten()
    assert int(on.min()) == 40 and int(on.max()) == 79                         # 1.0 s .. 2.0 s = ticks 40..79


def test_fit_audio_ticks():
    a = torch.ones(1, 32, 2, 200)
    out, note = fit_audio_ticks(a, 207)
    assert out.shape[-1] == 207 and torch.all(out[..., 200:] == 0) and "padded" in note
    out, note = fit_audio_ticks(torch.ones(1, 32, 2, 210), 207)
    assert out.shape[-1] == 207 and "trimmed" in note


def test_follow_mask_without_a_mask_is_refused():
    with pytest.raises(ValueError, match="needs a mask"):
        build_source_av_latent(_frames(), FakeH3VAE(), audio_mode="follow mask")


def test_node_execute():
    out = MiniMaxH3_SourceAVLatent.execute(_frames(), FakeH3VAE())
    assert "latent 37x16x20" in out.args[1]
