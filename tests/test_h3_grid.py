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
