from mmx_utils.wan_ltx_grid import (
    calculate_ltx2_frames,
    calculate_next_wan_frames,
    calculate_wan_frames,
    is_ltx2_compatible,
    is_wan_compatible,
)


def test_wan_compatible_counts():
    for n in (1, 5, 9, 13):
        assert is_wan_compatible(n)
    assert not is_wan_compatible(6)


def test_ltx2_compatible_counts():
    assert is_ltx2_compatible(9)
    assert calculate_ltx2_frames(10) == 17


def test_next_wan_when_already_compatible():
    assert calculate_next_wan_frames(9) == 9
