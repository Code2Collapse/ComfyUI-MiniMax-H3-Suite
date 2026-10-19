"""mmx_utils/h3_geometry.py - H3's spatial / temporal geometry, checked against core's own code (A16 Part II).

The offset numbers are the mechanism behind the masked-replacement displacement: a reference video that shows the same
shot as the output reaches the model half a coarse token up-left of its content when core's canvas downsizes it
(docs/evidence/A16/offset/ROOT_CAUSE_AND_DESIGN.md in the workspace). CPU only, no weights.
"""

from __future__ import annotations

import pytest
import torch

core_model = pytest.importorskip("comfy.ldm.minimax.model", reason="needs ComfyUI with MiniMax H3")
core_nodes = pytest.importorskip("comfy_extras.nodes_minimax_h3", reason="needs ComfyUI with MiniMax H3")

from mmx_utils import h3_geometry as geo  # noqa: E402


def test_same_grid_is_offset_zero():
    o = geo.stream_offset_px((1344, 1536), (1344, 1536))
    assert o["same_grid"] and o["max_abs_px"] == 0.0


def test_core_canvas_reference_is_fed_half_a_coarse_token_up_left():
    # 1440x1440 target (45 tokens of 32 px) vs core's 768x768 canvas (24 tokens = 60 target px each):
    # content sits at the patch centre, the model gets the patch's left/top edge -> (32 - 60) / 2 = -14 px everywhere.
    o = geo.stream_offset_px((1440, 1440), (768, 768))
    assert o["dx_px"] == [-14.0, -14.0] and o["dy_px"] == [-14.0, -14.0]


def test_core_canvas_from_the_owners_crop_size():
    # The owner's crop is ~2 MP; core's canvas for 1344x1536 is 768x864 (aspect rounded): offset + stretch.
    cw, ch = core_nodes.adapt_canvas(1344, 1536)
    assert (cw, ch) == (768, 864)
    o = geo.stream_offset_px((1344, 1536), (cw, ch))
    assert not o["same_grid"]
    assert o["dx_px"][1] < 0 and o["dy_px"][1] < 0            # every point is fed up and left
    assert 15.0 <= o["max_abs_px"] <= 20.0


def test_vae_crop_margins_match_core_rule():
    m = geo.vae_crop_margins(1920, 1080)
    assert (m["top"], m["bottom"], m["left"], m["right"]) == (4, 4, 0, 0)
    assert (m["kept_width"], m["kept_height"]) == (1920, 1072) and m["cropped"]
    assert not geo.vae_crop_margins(1344, 768)["cropped"]


@pytest.mark.parametrize("frames", [5, 22, 39, 107, 124, 362])
def test_frame_and_audio_counts_match_core(frames):
    fc, lt, at = core_nodes.temporal_shape(frames)
    assert fc == frames
    assert geo.frame_count_for_latent_t(lt) == frames
    assert geo.expected_audio_ticks(frames) == at


def test_nearest_legal_counts():
    assert geo.nearest_legal_frame_counts(120) == (107, 124)
    assert geo.nearest_legal_frame_counts(124) == (124, 124)
    assert geo.is_legal_frame_count(5) and not geo.is_legal_frame_count(6)


def test_mask_pooling_equals_core():
    from comfy.model_base import MiniMaxH3

    class _DM:
        patch_size = (1, 2, 2)

    class _Self:
        diffusion_model = _DM()

    torch.manual_seed(0)
    m = (torch.rand(7, 13, 11) > 0.7).float() * torch.rand(7, 13, 11)
    core = MiniMaxH3._pool_masks_to_token_grid(_Self(), [m[None, None].expand(1, 24, -1, -1, -1)])[0]
    core = torch.ceil(core * 256.0) / 256.0
    ours = geo.pool_video_mask_like_core(m)
    assert torch.equal(core[0, 0], ours)


def test_rope_extent_is_cores_grid():
    g, _ = core_model._frame_grid(48, 84)
    e = geo.rope_extent(48, 84)
    assert e["w"][0] == round(float(g[:, 1].min()), 3) and e["h"][1] == round(float(g[:, 0].max()), 3)
