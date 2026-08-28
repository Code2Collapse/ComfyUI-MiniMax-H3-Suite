from mmx_utils.h3_constants import CANVAS_MULTIPLE, FPS, align_frame_count, video_latent_t


def test_align_frame_count_snaps_to_17n_plus_5():
    assert align_frame_count(5) == 5
    assert align_frame_count(6) == 22
    assert align_frame_count(22) == 22


def test_video_latent_t_matches_grid():
    """Values are ComfyUI core's, not ours.

    comfy_extras/nodes_minimax_h3.py:40-41
        return 2 if frame_count <= 5 else ((frame_count - 5) // 17) * 5 + 2

    Each legal 17n+5 frame count adds exactly 5 latent frames, starting at 2.
    Asserting the whole ladder rather than one value, so an off-by-one in the
    grid cannot slip through on a single lucky case.
    """
    assert video_latent_t(5) == 2
    assert video_latent_t(22) == 7
    assert video_latent_t(39) == 12
    assert video_latent_t(56) == 17
    assert video_latent_t(73) == 22
    # the step between adjacent legal counts is always 5
    ladder = [5, 22, 39, 56, 73, 90, 107, 124, 141, 158, 175, 192]
    lat = [video_latent_t(n) for n in ladder]
    assert all(b - a == 5 for a, b in zip(lat, lat[1:])), lat


def test_canvas_multiple_is_32():
    assert CANVAS_MULTIPLE == 32


def test_fps_is_24():
    assert FPS == 24
