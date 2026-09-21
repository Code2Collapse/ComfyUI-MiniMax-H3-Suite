"""First-block residual caching for MiniMax H3.

THE IDEA. In a DiT, block 0's residual (its output minus its input) turns out
to predict how much the WHOLE stack is about to change. When two consecutive
steps produce nearly the same first-block residual, the remaining ~49 blocks
will produce nearly the same output too - so reuse the tail from the last full
step and skip them. One block instead of fifty.

THE RISK is a silent one: the picture still renders, it just stops moving.
Every guard below exists because of a specific way that happens.

MERGED FROM TWO PORTS. Two upstream packs implement this, each with something
the other lacks, and neither alone is safe for H3 video:

  ComfyUI-MiniMaxH3-FirstBlockCache      ComfyUI_H3FBC
  -------------------------------------  -----------------------------------
  temporal guard (per-FRAME diff)        warmup steps
  sigma window (start/end percent)       metric stride (subsampled compare)
  per-uuid contexts                      OOM tolerance
  accelerator conflict detection         in-place residual arithmetic
  run statistics

All of it is here, because each column catches a different failure:

  * TEMPORAL GUARD. The plain metric is a mean over the whole latent. On a
    124-frame clip one frame can change completely while the mean barely
    moves, so a shot with motion in a corner caches straight through it and
    that corner freezes. The guard takes the per-frame diff and uses the
    WORST frame, not the average.

  * WARMUP. The first few steps decide composition. Caching there does not
    blur a detail, it changes what the shot is of.

  * SIGMA WINDOW. Late steps are fine detail; early steps are structure.
    Both ends are excluded by percent rather than step number, so the window
    means the same thing at 8 steps and at 50.

  * PER-BRANCH CONTEXTS. cond and uncond have genuinely different hidden
    states. One shared cache compares cond against uncond, the diff is
    always large, and the cache simply never fires - the quiet failure where
    the node reports success and buys nothing.

  * MAX CONSECUTIVE HITS. Error compounds across skipped steps. Two in a row
    is recoverable; ten is a still frame.

  * METRIC STRIDE. The comparison itself allocates. On a long clip the
    metric tensor is large enough to matter, and a stride-4 view decides
    identically in practice.

  * OOM TOLERANCE. The cache holds a tail residual the size of the hidden
    state. If that is the allocation that fails, dropping it and running the
    step in full is strictly better than killing the job.

Pure torch, no comfy imports - see the AttributeError note in
mmx_nodes/block_cache.py for why module-level comfy imports are unsafe here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

# transformer_options key. The conflict guard looks for it, and so does every
# other accelerator in this pack: two caches patching the same blocks produce
# garbage rather than an error, so they have to be able to see each other.
FBC_KEY = "minimax_h3_first_block_cache"


class FirstBlockCacheError(RuntimeError):
    """Raised where a wrong answer would otherwise be silent."""


@dataclass(frozen=True)
class CacheConfig:
    """What the cache is allowed to do.

    Defaults are the safe preset. A user who wants speed picks a faster one
    knowingly; a user who wires the node up and leaves it alone gets the
    setting least likely to freeze their shot.
    """

    threshold: float = 0.08
    start_percent: float = 0.10
    end_percent: float = 0.95
    max_consecutive_hits: int = 2
    temporal_guard: bool = True
    warmup_steps: int = 3
    metric_stride: int = 1

    def validate(self) -> None:
        if not 0.0 <= self.threshold <= 1.0:
            raise FirstBlockCacheError(
                f"threshold must be between 0 and 1, got {self.threshold}. It is a "
                "RELATIVE difference - 0.08 means 'the first-block residual moved "
                "less than 8%'.")
        if self.start_percent >= self.end_percent:
            raise FirstBlockCacheError(
                f"start_percent ({self.start_percent}) must be smaller than "
                f"end_percent ({self.end_percent}). The window is a span of the "
                "denoise, not a pair of independent limits.")
        if self.max_consecutive_hits < 1:
            raise FirstBlockCacheError(
                "max_consecutive_hits must be at least 1; 0 would disable the cache "
                "while still reporting it as enabled.")
        if self.warmup_steps < 0:
            raise FirstBlockCacheError("warmup_steps cannot be negative.")
        if self.metric_stride < 1:
            raise FirstBlockCacheError("metric_stride must be at least 1.")


# Calibrated on H3. The labels carry the numbers because a user comparing two
# renders needs to know what changed between them.
PRESETS: dict[str, CacheConfig] = {
    "Safe — 0.06, warmup 4, max 2": CacheConfig(
        threshold=0.06, warmup_steps=4, max_consecutive_hits=2),
    "Balanced — 0.08, warmup 3, max 2": CacheConfig(
        threshold=0.08, warmup_steps=3, max_consecutive_hits=2),
    "Fast — 0.10, warmup 2, max 3": CacheConfig(
        threshold=0.10, warmup_steps=2, max_consecutive_hits=3),
    "Aggressive — 0.12, warmup 2, max 4": CacheConfig(
        threshold=0.12, warmup_steps=2, max_consecutive_hits=4),
}
CUSTOM_MODE = "Custom — use the values below"


@dataclass
class BranchState:
    """One sampling branch (cond, uncond, or one batch item).

    Kept apart because their hidden states are different. Sharing one state
    across branches compares cond to uncond every step, the diff never falls
    under threshold, and the cache silently never fires.
    """

    prev_first_metric: torch.Tensor | None = None
    tail_residual: torch.Tensor | None = None
    first_output: torch.Tensor | None = None
    pending_metric: torch.Tensor | None = None
    prev_first_full: torch.Tensor | None = None
    use_cache: bool = False
    consecutive_hits: int = 0
    prev_sigma: float | None = None
    signature: tuple | None = None
    steps_seen: int = 0
    last_diff: float | None = None
    video_slice: tuple[int, int] | None = None
    latent_frames: int | None = None

    def clear(self) -> None:
        self.prev_first_metric = None
        self.tail_residual = None
        self.first_output = None
        self.pending_metric = None
        self.prev_first_full = None
        self.use_cache = False
        self.consecutive_hits = 0
        self.prev_sigma = None
        self.signature = None
        self.steps_seen = 0
        self.last_diff = None


@dataclass
class CacheStats:
    full_steps: int = 0
    cached_steps: int = 0
    diffs: list[float] = field(default_factory=list)
    temporal_diffs: list[float] = field(default_factory=list)
    cached_at: list[int] = field(default_factory=list)
    oom_recoveries: int = 0

    def clear(self) -> None:
        self.full_steps = 0
        self.cached_steps = 0
        self.diffs.clear()
        self.temporal_diffs.clear()
        self.cached_at.clear()
        self.oom_recoveries = 0


def is_oom(exc: BaseException) -> bool:
    """Whether this is an allocation failure we can back off from.

    Matched on the message as well as the type: allocator failures surface as
    RuntimeError with a recognisable message on several backends, and only
    CUDA has its own class.
    """
    if isinstance(exc, torch.cuda.OutOfMemoryError):  # type: ignore[attr-defined]
        return True
    if not isinstance(exc, RuntimeError):
        return False
    text = str(exc).lower()
    return "out of memory" in text or "can't allocate" in text


def relative_diff(current: torch.Tensor, previous: torch.Tensor) -> float:
    """Mean absolute difference, relative to the previous magnitude.

    Relative, not absolute, because the residual's scale changes by orders of
    magnitude across the denoise - a fixed threshold would mean 'never' early
    and 'always' late.
    """
    if current.shape != previous.shape:
        return float("inf")
    cur = current.float()
    prev = previous.float()
    denominator = prev.abs().mean().clamp(min=1e-8)
    return float(((cur - prev).abs().mean() / denominator).item())


def temporal_diff(
    current: torch.Tensor,
    previous: torch.Tensor,
    video_slice: tuple[int, int] | None,
    latent_frames: int | None,
) -> float | None:
    """The WORST single frame's relative change, or None if unknowable.

    This is the guard that matters for video. `relative_diff` averages over
    every token in the clip, so one frame can change completely while the mean
    stays under threshold - the cache fires and that frame freezes while the
    rest of the shot moves. Taking the max over frames refuses to average a
    moving frame away.

    None when the layout is unknown, which the caller treats as 'no opinion'
    rather than 'safe'.
    """
    if video_slice is None or not latent_frames:
        return None
    start, stop = video_slice
    cur = current[start:stop]
    prev = previous[start:stop]
    if cur.shape != prev.shape or cur.shape[0] == 0:
        return None
    if cur.shape[0] % latent_frames:
        return None
    rows = cur.shape[0] // latent_frames
    cur = cur.reshape(latent_frames, rows, -1).float()
    prev = prev.reshape(latent_frames, rows, -1).float()
    numerator = (cur - prev).abs().mean(dim=(1, 2))
    denominator = prev.abs().mean(dim=(1, 2)).clamp(min=1e-8)
    return float((numerator / denominator).max().item())


def video_layout(payload) -> tuple[tuple[int, int] | None, int | None]:
    """Where the video tokens sit in the packed AV sequence.

    H3 packs video and audio into one sequence, so a per-frame metric has to
    know which rows are video. Returns (None, None) when the payload does not
    say, which disables the temporal guard rather than guessing at it.
    """
    if not payload:
        return None, None
    layout = payload.get("layout") if hasattr(payload, "get") else None
    if layout is None:
        return None, None
    segments = getattr(layout, "segments", None)
    if not segments:
        return None, None
    span = next(((a, b) for a, b, kind in segments if kind == "video"), None)
    signature = getattr(layout, "signature", ())
    frames = signature[1] if len(signature) > 1 else None
    return span, frames


class FirstBlockCache:
    """The cache itself: decides, holds the tail residual, and reports."""

    def __init__(self, config: CacheConfig, start_sigma: float,
                 end_sigma: float, block_count: int):
        config.validate()
        self.config = config
        self.start_sigma = float(start_sigma)
        self.end_sigma = float(end_sigma)
        self.block_count = int(block_count)
        self.branches: dict[tuple, BranchState] = {}
        self.current: BranchState | None = None
        self.stats = CacheStats()

    # ── lifecycle ───────────────────────────────────────────────────────────

    def reset(self) -> None:
        for state in self.branches.values():
            state.clear()
        self.branches.clear()
        self.current = None
        self.stats.clear()

    @staticmethod
    def branch_key(transformer_options: dict) -> tuple:
        """One key per sampling branch.

        Prefers uuids, which distinguish batch items as well as cond/uncond.
        Falls back to cond_or_uncond, which older builds provide. Falling all
        the way back to a single shared key is the case that silently never
        caches, so it is reported rather than hidden.
        """
        uuids = transformer_options.get("uuids")
        if uuids:
            return tuple(str(u) for u in uuids)
        cond = transformer_options.get("cond_or_uncond")
        if cond is not None:
            try:
                return ("cu",) + tuple(int(c) for c in cond)
            except TypeError:
                return ("cu", str(cond))
        return ("shared",)

    def begin_call(self, x, timestep, transformer_options, payload=None) -> None:
        """Start one model call. Detects a new sampling run and clears state."""
        sigma = float(torch.as_tensor(timestep).flatten()[0].item()) / 1000.0
        key = self.branch_key(transformer_options or {})
        state = self.branches.setdefault(key, BranchState())

        signature = self._signature(x)
        # Sigma going UP means a new run (or a second pass) reusing this
        # object. Stale residuals from the previous run would be applied to a
        # different image entirely.
        restarted = (state.prev_sigma is not None
                     and sigma > state.prev_sigma + 1e-7)
        if state.signature != signature or restarted:
            state.clear()

        state.signature = signature
        state.prev_sigma = sigma
        state.video_slice, state.latent_frames = video_layout(payload)
        state.first_output = None
        state.pending_metric = None
        state.use_cache = False
        self.current = state

    def end_call(self) -> None:
        self.current = None

    @staticmethod
    def _signature(x) -> tuple:
        items = x if isinstance(x, (tuple, list)) else (x,)
        return tuple((tuple(t.shape), t.dtype, str(t.device))
                     for t in items if torch.is_tensor(t))

    # ── the decision ────────────────────────────────────────────────────────

    def _metric_view(self, residual: torch.Tensor) -> torch.Tensor:
        """A subsample to compare against, so the metric itself is cheap."""
        if self.config.metric_stride <= 1:
            return residual.detach().clone()
        return residual[::self.config.metric_stride].detach().clone()

    def in_window(self, state: BranchState) -> bool:
        sigma = state.prev_sigma
        if sigma is None:
            return False
        return self.end_sigma <= sigma <= self.start_sigma

    def decide(self, first_residual: torch.Tensor,
               first_output: torch.Tensor) -> bool:
        """Cache this step, or run it in full? Returns True to cache."""
        state = self.current
        if state is None:
            raise FirstBlockCacheError(
                "The cache was asked to decide outside a model call. Its block "
                "patches are installed but the diffusion-model wrapper is not, "
                "so it has no idea which step or branch this is.")

        state.steps_seen += 1
        metric = self._metric_view(first_residual)

        use_cache = False
        diff: float | None = None
        have_tail = (state.tail_residual is not None
                     and state.tail_residual.shape == first_output.shape)
        warm = state.steps_seen > self.config.warmup_steps

        if have_tail and state.prev_first_metric is not None and warm \
                and self.in_window(state) and self.config.threshold > 0.0:
            diff = relative_diff(metric, state.prev_first_metric)
            self.stats.diffs.append(diff)
            decision = diff

            if self.config.temporal_guard and state.prev_first_full is not None:
                per_frame = temporal_diff(
                    first_residual, state.prev_first_full,
                    state.video_slice, state.latent_frames)
                if per_frame is not None:
                    self.stats.temporal_diffs.append(per_frame)
                    # The WORST frame decides. Averaging here is the bug the
                    # guard exists to prevent.
                    decision = max(decision, per_frame)

            use_cache = (math.isfinite(decision)
                         and decision <= self.config.threshold
                         and state.consecutive_hits < self.config.max_consecutive_hits)

        state.last_diff = diff
        state.use_cache = use_cache

        if use_cache:
            self.stats.cached_at.append(self.stats.full_steps + self.stats.cached_steps + 1)
            state.consecutive_hits += 1
            state.first_output = None
            state.pending_metric = None
        else:
            state.consecutive_hits = 0
            state.first_output = first_output.detach().clone()
            state.pending_metric = metric
            if self.config.temporal_guard:
                state.prev_first_full = first_residual.detach().clone()
        return use_cache

    # ── completing a step ───────────────────────────────────────────────────

    def finish_full_step(self, output: torch.Tensor) -> None:
        """Store the tail: everything blocks 1..N did on top of block 0.

        On OOM the cache drops what it holds and carries on uncached rather
        than killing the run - a slower render beats no render.
        """
        state = self.current
        if state is None or state.first_output is None or state.pending_metric is None:
            raise FirstBlockCacheError(
                "A full step finished with no first-block output recorded. The "
                "block patches were installed on some blocks but not block 0, "
                "so the cache never saw the step begin.")
        try:
            state.tail_residual = (output - state.first_output).detach()
        except Exception as exc:  # noqa: BLE001 - re-raised unless it is OOM
            if not is_oom(exc):
                raise
            self.stats.oom_recoveries += 1
            state.tail_residual = None
            state.prev_first_full = None
        state.prev_first_metric = state.pending_metric
        state.first_output = None
        state.pending_metric = None
        self.stats.full_steps += 1

    def finish_cached_step(self, first_output: torch.Tensor) -> torch.Tensor:
        state = self.current
        if state is None or state.tail_residual is None:
            raise FirstBlockCacheError(
                "A cached step was asked for a tail residual that does not "
                "exist. Nothing decided to cache this step, so the block "
                "patches and the cache disagree about what is happening.")
        self.stats.cached_steps += 1
        return first_output + state.tail_residual

    # ── reporting ───────────────────────────────────────────────────────────

    def summary(self) -> str:
        """What it actually did. A cache that never fired looks exactly like a
        cache that is working, unless it says so."""
        stats = self.stats
        steps = stats.full_steps + stats.cached_steps
        if steps == 0:
            return ("First-block cache: no model steps ran. Nothing was "
                    "accelerated and nothing was harmed.")

        executed = stats.full_steps * self.block_count + stats.cached_steps
        speedup = steps * self.block_count / max(executed, 1)
        lines = [
            f"First-block cache: reused {stats.cached_steps} of {steps} steps "
            f"({speedup:.2f}x fewer block evaluations).",
        ]
        if stats.cached_steps == 0:
            lines.append(
                "The cache never fired. Either the shot has too much motion for "
                f"threshold {self.config.threshold:.3f}, or the run was shorter "
                f"than the {self.config.warmup_steps}-step warmup. Nothing is "
                "broken - but nothing was saved either.")

        finite = sorted(d for d in stats.diffs if math.isfinite(d))
        if finite:
            mid = len(finite) // 2
            median = (finite[mid] if len(finite) % 2
                      else (finite[mid - 1] + finite[mid]) / 2)
            lines.append(
                f"Residual difference: min {finite[0]:.4f}, median {median:.4f}, "
                f"max {finite[-1]:.4f} (threshold {self.config.threshold:.3f}).")

        temporal = sorted(d for d in stats.temporal_diffs if math.isfinite(d))
        if temporal:
            blocked = sum(1 for d in temporal if d > self.config.threshold)
            lines.append(
                f"Temporal guard: worst single frame moved {temporal[-1]:.4f}; "
                f"it blocked {blocked} step(s) the whole-clip average would "
                "have cached and frozen.")
        elif self.config.temporal_guard:
            lines.append(
                "Temporal guard is on but the video layout was not visible on "
                "this model call, so it never ran. Decisions used the whole-clip "
                "average, which can average away one frozen frame.")

        if stats.cached_at:
            lines.append(f"Reused at steps: {stats.cached_at}.")
        if stats.oom_recoveries:
            lines.append(
                f"Recovered from {stats.oom_recoveries} out-of-memory event(s) by "
                "dropping the cache and running those steps in full.")
        return "\n".join(lines)
