import pytest

from mmx_utils.h3_grid import (
    calculate_h3_frames,
    calculate_next_h3_frames,
    is_h3_compatible,
    snap_frame_count,
)

H3_LEGAL = [5, 22, 39, 56, 73, 90, 107, 124, 141, 158, 175, 192]


@pytest.mark.parametrize("n", H3_LEGAL)
def test_is_h3_compatible_legal_counts(n):
    assert is_h3_compatible(n)


@pytest.mark.parametrize("n", [6, 21, 23, 100])
def test_is_h3_compatible_illegal(n):
    assert not is_h3_compatible(n)


@pytest.mark.parametrize("n", H3_LEGAL)
def test_calculate_h3_frames_identity_on_legal(n):
    assert calculate_h3_frames(n) == n


def test_calculate_h3_frames_rounds_up():
    assert calculate_h3_frames(6) == 22
    assert calculate_h3_frames(20) == 22


@pytest.mark.parametrize("n", H3_LEGAL)
def test_snap_matches_calculate(n):
    assert snap_frame_count(n) == n


def test_the_image_studio_decode_length_is_the_same_grid():
    """decoded_frames_for_latent_t arrived with the Image Studio port carrying
    its own closed-form copy of this grid. It is now a thin wrapper, and this
    pins that: a second implementation would drift, and the symptom would be
    an image packet that decodes to a different length than the planner
    expected - off-by-one frames at the end of every still.
    """
    from mmx_utils.h3_grid import decoded_frames_for_latent_t, frames_for_tokens

    for t in range(1, 500):
        assert decoded_frames_for_latent_t(t) == frames_for_tokens(t), t
    # zero rows is not a real latent; the wrapper clamps rather than returning 0
    assert decoded_frames_for_latent_t(0) == 1


def test_latent_t_for_frame_count_round_trips():
    """Asking for N frames must give a latent that decodes to at least N."""
    from mmx_utils.h3_grid import decoded_frames_for_latent_t, latent_t_for_frame_count

    for want in (1, 5, 9, 13, 20, 22, 39, 124):
        latent_t, _ = latent_t_for_frame_count(want)
        assert decoded_frames_for_latent_t(latent_t) >= want, want
