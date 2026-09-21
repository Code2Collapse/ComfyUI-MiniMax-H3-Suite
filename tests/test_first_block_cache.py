"""First-block residual caching.

Every failure this cache can produce is SILENT. It never crashes and never
draws an artefact; it freezes motion, or it quietly does nothing while
reporting that it is enabled. So each guard is pinned to the specific way it
fails without one:

  temporal guard      one frame freezes while the clip moves
  warmup              composition changes, not just detail
  sigma window        same, by a different route
  per-branch state    cond compared against uncond -> never fires at all
  consecutive limit   compounding error -> a still frame
  restart detection   a stale tail applied to a different image
  OOM tolerance       the run dies instead of slowing down

CPU-only, no model, no comfy.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

PACK = Path(__file__).resolve().parents[1]
if str(PACK) not in sys.path:
    sys.path.insert(0, str(PACK))

from mmx_utils.first_block_cache import (  # noqa: E402
    CUSTOM_MODE,
    PRESETS,
    CacheConfig,
    FirstBlockCache,
    FirstBlockCacheError,
    is_oom,
    relative_diff,
    temporal_diff,
    video_layout,
)


class FakeLayout:
    """The video/audio split H3 puts in its payload."""

    def __init__(self, video=(0, 40), frames=10, audio=(40, 50)):
        self.segments = [(video[0], video[1], "video"), (audio[0], audio[1], "audio")]
        self.signature = ("h3", frames)


def cache(**kw):
    config = CacheConfig(**{"warmup_steps": 0, "temporal_guard": False, **kw})
    return FirstBlockCache(config, start_sigma=1.0, end_sigma=0.0, block_count=50)


def step(c, residual, output, *, sigma=0.5, key=("a",), payload=None):
    """One model call: begin, decide, finish. Returns whether it cached."""
    c.begin_call(torch.zeros(4, 8), torch.tensor([sigma * 1000.0]),
                 {"uuids": key}, payload)
    used = c.decide(residual, output)
    if used:
        result = c.finish_cached_step(output)
    else:
        # A real full step's output is block 0's output plus whatever the other
        # 49 blocks add; a fixed tail makes the reuse checkable.
        result = output + torch.full_like(output, 3.0)
        c.finish_full_step(result)
    c.end_call()
    return used, result


# ── the metric ──────────────────────────────────────────────────────────────

def test_the_difference_is_relative_not_absolute():
    """The residual's scale changes by orders of magnitude across a denoise. A
    fixed absolute threshold would mean 'never' early and 'always' late."""
    small = torch.full((8,), 0.001)
    big = torch.full((8,), 1000.0)
    assert relative_diff(small * 1.1, small) == pytest.approx(0.1, abs=1e-5)
    assert relative_diff(big * 1.1, big) == pytest.approx(0.1, abs=1e-5)


def test_a_shape_change_is_infinitely_different():
    """Not zero, and not a crash. A changed shape means a different run."""
    assert relative_diff(torch.zeros(8), torch.zeros(4)) == float("inf")


def test_identical_residuals_are_zero_apart():
    a = torch.randn(16)
    assert relative_diff(a.clone(), a) == pytest.approx(0.0, abs=1e-6)


# ── the temporal guard, which is the one that matters for video ─────────────

FRAMES = 100          # a real clip, not a toy
ROWS = 4              # latent rows per frame
VIDEO = (0, FRAMES * ROWS)
TOTAL = VIDEO[1] + 10  # plus a short audio tail


def moving_frame(index=3, amount=0.6):
    """A clip where exactly ONE frame moves. The whole point is that the mean
    over the clip barely notices: one frame in a hundred, so 99% of the tokens
    are identical."""
    prev = torch.ones(TOTAL, 4)
    cur = prev.clone()
    lo = index * ROWS
    cur[lo:lo + ROWS] = 1.0 + amount
    return prev, cur


def test_one_frozen_frame_is_invisible_to_the_average():
    """THE failure this guard exists for. One frame in a hundred moves 60%.
    The clip average is a twentieth of a typical threshold, so a threshold on
    it caches the step and that frame stops moving while the shot carries on."""
    prev, cur = moving_frame()

    average = relative_diff(cur, prev)
    worst = temporal_diff(cur, prev, VIDEO, FRAMES)

    assert average < 0.01, (
        f"the clip average is {average:.4f} - the fixture does not reproduce "
        "the problem, because the change is too large a share of the latent")
    assert worst is not None
    assert worst == pytest.approx(0.6, abs=0.01), (
        "the per-frame metric must see the frame's OWN change, undiluted")
    assert worst > average * 50, (
        "the per-frame metric must be dramatically larger, or it adds nothing")


def test_the_guard_blocks_a_step_the_average_would_cache():
    """The same fixture, run through the cache both ways. This is the whole
    argument for the guard: identical inputs, opposite decisions."""
    prev, cur = moving_frame()
    payload = {"layout": FakeLayout(video=VIDEO, frames=FRAMES,
                                    audio=(VIDEO[1], TOTAL))}
    out = torch.ones(TOTAL, 4)

    without = cache(threshold=0.12, temporal_guard=False)
    step(without, prev, out, payload=payload)
    used_without, _ = step(without, cur, out, payload=payload)

    with_guard = cache(threshold=0.12, temporal_guard=True)
    step(with_guard, prev, out, payload=payload)
    used_with, _ = step(with_guard, cur, out, payload=payload)

    assert used_without, "the fixture no longer fools the plain average"
    assert not used_with, "the temporal guard let a frozen frame through"


def test_the_guard_has_no_opinion_when_the_layout_is_unknown():
    """None, not 0.0. Treating 'cannot tell' as 'safe' is how the guard would
    silently switch itself off on an unfamiliar payload."""
    assert temporal_diff(torch.ones(50, 4), torch.ones(50, 4), None, 10) is None
    assert temporal_diff(torch.ones(50, 4), torch.ones(50, 4), (0, 40), None) is None
    # rows that do not divide into frames: a reshape would lie about the layout
    assert temporal_diff(torch.ones(50, 4), torch.ones(50, 4), (0, 37), 10) is None


def test_the_report_admits_when_the_guard_never_ran():
    c = cache(threshold=0.9, temporal_guard=True)
    for _ in range(3):
        step(c, torch.ones(50, 4), torch.ones(50, 4))     # no payload
    text = c.summary()
    assert "never ran" in text and "average" in text


def test_the_layout_reader_finds_the_video_span():
    span, frames = video_layout({"layout": FakeLayout(video=(0, 40), frames=10)})
    assert span == (0, 40) and frames == 10
    assert video_layout(None) == (None, None)
    assert video_layout({}) == (None, None)
    assert video_layout({"layout": None}) == (None, None)


# ── warmup and window ───────────────────────────────────────────────────────

def test_warmup_steps_are_never_cached():
    """Early steps set composition. Caching there changes what the shot is
    OF, which no amount of later refinement undoes."""
    c = cache(threshold=0.9, warmup_steps=3)
    same = torch.ones(8, 4)
    used = [step(c, same, same)[0] for _ in range(6)]
    assert used[:3] == [False, False, False], "a warmup step was cached"
    assert any(used[3:]), "caching never started after the warmup"


def test_nothing_is_cached_outside_the_sigma_window():
    c = FirstBlockCache(CacheConfig(threshold=0.9, warmup_steps=0,
                                    temporal_guard=False),
                        start_sigma=0.8, end_sigma=0.2, block_count=50)
    same = torch.ones(8, 4)
    step(c, same, same, sigma=0.9)
    assert not step(c, same, same, sigma=0.9)[0], "cached above the window"
    step(c, same, same, sigma=0.5)
    assert step(c, same, same, sigma=0.5)[0], "never cached inside the window"
    step(c, same, same, sigma=0.1)
    assert not step(c, same, same, sigma=0.1)[0], "cached below the window"


def test_a_window_that_runs_backwards_is_refused():
    with pytest.raises(FirstBlockCacheError, match="smaller than"):
        CacheConfig(start_percent=0.9, end_percent=0.1).validate()


# ── branch separation ───────────────────────────────────────────────────────

def test_cond_and_uncond_do_not_share_a_cache():
    """Sharing one state compares cond against uncond every step. The diff is
    always huge, so the cache never fires - it reports success and buys
    nothing, which is the hardest failure to notice."""
    c = cache(threshold=0.1)
    cond = torch.ones(8, 4)
    uncond = torch.full((8, 4), 50.0)

    step(c, cond, cond, key=("cond",))
    step(c, uncond, uncond, key=("uncond",))
    assert step(c, cond, cond, key=("cond",))[0], "cond did not reuse its own state"
    assert step(c, uncond, uncond, key=("uncond",))[0], "uncond did not either"
    assert len(c.branches) == 2


def test_the_branch_key_falls_back_to_cond_or_uncond():
    assert FirstBlockCache.branch_key({"uuids": ["x", "y"]}) == ("x", "y")
    assert FirstBlockCache.branch_key({"cond_or_uncond": [0]}) == ("cu", 0)
    assert FirstBlockCache.branch_key({}) == ("shared",)


# ── the compounding limit ───────────────────────────────────────────────────

def test_consecutive_skips_are_capped():
    """Error compounds across reuses. Without a cap an identical-looking run
    caches to the end and the result is a still frame."""
    c = cache(threshold=0.9, max_consecutive_hits=2)
    same = torch.ones(8, 4)
    used = [step(c, same, same)[0] for _ in range(9)]
    run = longest = 0
    for u in used:
        run = run + 1 if u else 0
        longest = max(longest, run)
    assert longest <= 2, f"skipped {longest} steps in a row"


def test_a_cap_of_zero_is_refused_rather_than_silently_disabling_the_cache():
    with pytest.raises(FirstBlockCacheError, match="at least 1"):
        CacheConfig(max_consecutive_hits=0).validate()


# ── what a cached step actually produces ────────────────────────────────────

def test_a_cached_step_adds_back_the_tail_from_the_last_full_step():
    c = cache(threshold=0.9)
    same = torch.ones(8, 4)
    _, full = step(c, same, same)
    used, cached = step(c, same, same)
    assert used
    # the tail recorded above was a constant 3.0 on top of block 0's output
    assert torch.allclose(cached, same + 3.0)
    assert torch.allclose(cached, full)


def test_the_first_step_can_never_cache_because_there_is_no_tail_yet():
    c = cache(threshold=1.0)
    assert not step(c, torch.ones(8, 4), torch.ones(8, 4))[0]


# ── restart detection ───────────────────────────────────────────────────────

def test_a_rising_sigma_clears_the_cache():
    """Sigma going up means a new run on the same object. A tail residual from
    the previous image applied to this one is a ghost of the old picture."""
    c = cache(threshold=0.9)
    same = torch.ones(8, 4)
    step(c, same, same, sigma=0.5)
    step(c, same, same, sigma=0.4)
    assert not step(c, same, same, sigma=0.9)[0], "a stale tail survived a restart"


def test_a_changed_latent_shape_clears_the_cache():
    c = cache(threshold=0.9)
    step(c, torch.ones(8, 4), torch.ones(8, 4))
    c.begin_call(torch.zeros(9, 9), torch.tensor([500.0]), {"uuids": ("a",)})
    assert c.current is not None and c.current.tail_residual is None


# ── OOM tolerance ───────────────────────────────────────────────────────────

def test_allocation_failures_are_recognised():
    assert is_oom(RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB"))
    assert is_oom(RuntimeError("DefaultCPUAllocator: can't allocate memory"))
    assert not is_oom(RuntimeError("shape mismatch"))
    assert not is_oom(ValueError("out of memory"))   # wrong type, not ours


class OOMOnSubtract(torch.Tensor):
    """A tensor whose subtraction fails the way an allocator does.

    Subclassing Tensor rather than standing in for one, because the cache
    stores this value first and only subtracts later - a bare object fails the
    earlier step and the test would pass for the wrong reason.
    """

    def __sub__(self, other):
        raise RuntimeError("CUDA out of memory. Tried to allocate 8.00 GiB")


def test_running_out_of_memory_slows_the_run_down_instead_of_killing_it():
    """The tail residual is the size of the hidden state. If THAT is the
    allocation that fails, dropping it and running uncached is strictly better
    than ending the render."""
    same = torch.ones(8, 4)
    moved = torch.full((8, 4), 5.0)
    # threshold 0 so the second step is a FULL step: only a full step builds
    # the tail residual, which is the allocation being made to fail.
    c = cache(threshold=0.0)
    step(c, same, same)

    c.begin_call(torch.zeros(4, 8), torch.tensor([500.0]), {"uuids": ("a",)})
    assert not c.decide(moved, moved), "this step must run in full"
    c.finish_full_step(torch.ones(8, 4).as_subclass(OOMOnSubtract))
    c.end_call()

    assert c.stats.oom_recoveries == 1
    assert c.stats.full_steps == 2, "the step must still count as completed"
    assert c.current is None
    assert "out-of-memory" in c.summary()


def test_a_real_error_during_the_tail_is_not_swallowed_as_an_oom():
    """Dropping the cache is the right answer for an allocation failure and
    the WRONG answer for a shape bug, which would then be invisible."""

    class Broken(torch.Tensor):
        def __sub__(self, other):
            raise RuntimeError("The size of tensor a (8) must match tensor b (4)")

    c = cache(threshold=0.9)
    same = torch.ones(8, 4)
    c.begin_call(torch.zeros(4, 8), torch.tensor([500.0]), {"uuids": ("a",)})
    c.decide(same, same)
    with pytest.raises(RuntimeError, match="must match"):
        c.finish_full_step(torch.ones(8, 4).as_subclass(Broken))


# ── honesty in the report ───────────────────────────────────────────────────

def test_the_report_says_plainly_when_the_cache_never_fired():
    """A cache that never fires looks exactly like one that is working."""
    c = cache(threshold=0.0001)
    step(c, torch.ones(8, 4), torch.ones(8, 4))
    step(c, torch.full((8, 4), 9.0), torch.ones(8, 4))
    text = c.summary()
    assert "never fired" in text
    assert "nothing was saved" in text


def test_the_report_counts_what_it_reused():
    c = cache(threshold=0.9, max_consecutive_hits=10)
    same = torch.ones(8, 4)
    for _ in range(5):
        step(c, same, same)
    text = c.summary()
    assert "reused 4 of 5 steps" in text
    assert "fewer block evaluations" in text


def test_no_steps_at_all_is_stated_rather_than_divided_by_zero():
    assert "no model steps" in cache().summary()


# ── misuse is named, not guessed at ─────────────────────────────────────────

def test_deciding_outside_a_model_call_names_the_missing_wrapper():
    c = cache()
    with pytest.raises(FirstBlockCacheError, match="wrapper"):
        c.decide(torch.ones(4), torch.ones(4))


def test_finishing_a_step_that_never_began_names_block_zero():
    c = cache()
    c.begin_call(torch.zeros(4), torch.tensor([500.0]), {})
    with pytest.raises(FirstBlockCacheError, match="block 0"):
        c.finish_full_step(torch.ones(4))


def test_a_cached_step_with_no_tail_is_refused():
    c = cache()
    c.begin_call(torch.zeros(4), torch.tensor([500.0]), {})
    with pytest.raises(FirstBlockCacheError, match="does not exist"):
        c.finish_cached_step(torch.ones(4))


# ── presets ─────────────────────────────────────────────────────────────────

def test_every_preset_is_valid_and_named_with_its_numbers():
    for name, config in PRESETS.items():
        config.validate()
        assert f"{config.threshold:.2f}" in name, (
            f"preset {name!r} does not name its own threshold")
        assert str(config.warmup_steps) in name


def test_the_presets_are_ordered_from_safe_to_fast():
    thresholds = [c.threshold for c in PRESETS.values()]
    assert thresholds == sorted(thresholds), "the preset list is not ordered"


def test_the_default_config_has_the_video_guards_on():
    """A user who wires this up and leaves it alone must not get frozen
    motion. The unsafe settings have to be chosen deliberately."""
    default = CacheConfig()
    assert default.temporal_guard is True
    assert default.warmup_steps >= 1
    assert default.max_consecutive_hits <= 3


def test_custom_mode_is_not_a_preset_name():
    assert CUSTOM_MODE not in PRESETS
