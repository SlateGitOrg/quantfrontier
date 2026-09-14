"""Frontier analysis: cliff detection, Pareto dominance, and eval-set sizing.

This is the differentiator module. Everything here operates on `Measurement`
records and answers the three questions a quantisation decision actually turns
on:

  1. Does this metric degrade smoothly, or does it fall off a cliff? (And where?)
  2. Which configs are Pareto-optimal on (quality, latency, memory) FOR THIS
     TASK -- because the answer differs by task.
  3. How many eval items do I need before a difference between two adjacent
     configs is anything other than noise?
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import NormalDist
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from .backend import Measurement

# Two-sided significance and conventional power. Both are reported alongside
# every sizing result rather than buried, because the required n is meaningless
# without them.
DEFAULT_ALPHA = 0.05
DEFAULT_POWER = 0.80

# A latency difference is only believed if it clears this many standard errors
# of the difference. 1.96 is the two-sided 5% normal critical value; with it,
# configs whose timings overlap are treated as tied on latency rather than one
# spuriously dominating the other.
LATENCY_Z = 1.96


# --------------------------------------------------------------------------
# Cliff detection
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class CliffReport:
    """Shape of one metric's degradation curve across an ordered sweep."""

    metric: str
    labels: List[str]
    values: List[float]
    total_drop: float
    max_step_drop: float
    max_step_index: int
    monotone_non_increasing: bool

    @property
    def cliff_share(self) -> float:
        """Fraction of the entire degradation concentrated in one step.

        With S steps, a perfectly smooth curve gives 1/S. A true cliff gives
        something close to 1.0. The share is the statistic the cliff test keys
        on, because it is invariant to the units of the metric and to how much
        total degradation happened.
        """
        if self.total_drop <= 0:
            return 0.0
        return self.max_step_drop / self.total_drop

    @property
    def smooth_share(self) -> float:
        """The cliff_share a perfectly even degradation would produce."""
        steps = len(self.values) - 1
        return 1.0 / steps if steps > 0 else 0.0

    def steps_to_share(self, share: float = 0.90) -> int:
        """Fewest sweep steps that together account for `share` of the total drop.

        A threshold-free way to say "cliff". If ONE step out of five explains
        90% of everything the metric lost, the metric has a cliff and averaging
        across bit widths is meaningless. If it takes most of the steps, the
        metric degraded gradually. The answer is an integer, so the test that
        uses it does not need an invented float tolerance.
        """
        if not (0.0 < share <= 1.0):
            raise ValueError("share must be in (0, 1]")
        if self.total_drop <= 0:
            return 0
        target = share * self.total_drop
        running = 0.0
        drops = sorted((self.values[i] - self.values[i + 1] for i in range(len(self.values) - 1)), reverse=True)
        for k, d in enumerate(drops, start=1):
            running += d
            if running >= target:
                return k
        return len(drops)

    @property
    def cliff_at(self) -> str:
        """Label of the config the largest single drop lands ON."""
        return self.labels[self.max_step_index + 1]

    @property
    def cliff_from(self) -> str:
        return self.labels[self.max_step_index]


def degradation_profile(labels: Sequence[str], values: Sequence[float], metric: str) -> CliffReport:
    """Profile a metric across a sweep already ordered worst-quantisation-last."""
    if len(labels) != len(values):
        raise ValueError("labels and values must be the same length")
    if len(values) < 3:
        raise ValueError("need at least 3 points to talk about a cliff")
    drops = [values[i] - values[i + 1] for i in range(len(values) - 1)]
    # Total drop is measured end-to-end, not as the sum of positive steps: a
    # metric that wobbles upward should not get credit for the wobble.
    total = values[0] - values[-1]
    max_drop = max(drops)
    return CliffReport(
        metric=metric,
        labels=list(labels),
        values=list(values),
        total_drop=total,
        max_step_drop=max_drop,
        max_step_index=drops.index(max_drop),
        monotone_non_increasing=all(d >= -1e-12 for d in drops),
    )


def profile_sweep(measurements: Sequence[Measurement]) -> Dict[str, CliffReport]:
    """Profile the aggregate metric and every task over one ordered sweep.

    Key "aggregate" holds the perplexity-derived score; task names hold theirs.
    """
    labels = [m.config.name for m in measurements]
    out: Dict[str, CliffReport] = {
        "aggregate": degradation_profile(labels, [m.aggregate_quality for m in measurements], "aggregate")
    }
    for task in measurements[0].task_accuracy:
        out[task] = degradation_profile(labels, [m.task_accuracy[task] for m in measurements], task)
    return out


def select_on_metric(
    measurements: Sequence[Measurement],
    score: Callable[[Measurement], float],
    min_score: float,
) -> Optional[Measurement]:
    """The config a single-metric decision picks: cheapest memory among those
    whose score clears the bar. This is exactly the procedure the project argues
    against when `score` is the aggregate metric."""
    eligible = [m for m in measurements if score(m) >= min_score]
    if not eligible:
        return None
    return min(eligible, key=lambda m: (m.memory_mb, m.latency_ms_mean))


# --------------------------------------------------------------------------
# Pareto frontier
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ParetoResult:
    task: str
    frontier: List[Measurement]
    dominated: List[Tuple[Measurement, Measurement]]  # (loser, a config that dominates it)

    @property
    def frontier_names(self) -> List[str]:
        return [m.config.name for m in self.frontier]

    @property
    def dominated_names(self) -> List[str]:
        return [loser.config.name for loser, _ in self.dominated]


def _dominates(a: Measurement, b: Measurement, quality: Callable[[Measurement], float]) -> bool:
    """True if a is at least as good as b on all three objectives and strictly
    better on one.

    Latency is compared with its measurement error: a is only counted as faster
    when the gap clears LATENCY_Z standard errors of the difference. Without
    that guard, 2%-noise timings manufacture dominance relations that vanish on
    the next run.
    """
    qa, qb = quality(a), quality(b)
    if qa < qb:
        return False
    if a.memory_mb > b.memory_mb:
        return False

    sem = math.sqrt(a.latency_sem ** 2 + b.latency_sem ** 2)
    gap = b.latency_ms_mean - a.latency_ms_mean  # positive => a is faster
    a_slower = gap < -LATENCY_Z * sem
    a_faster = gap > LATENCY_Z * sem
    if a_slower:
        return False

    strictly_better = (qa > qb) or (a.memory_mb < b.memory_mb) or a_faster
    return strictly_better


def pareto_frontier(measurements: Sequence[Measurement], task: str) -> ParetoResult:
    """Pareto frontier on (task quality up, latency down, memory down).

    `task` may be a task name or "aggregate".
    """
    for m in measurements:
        if m.replications < 2:
            # The README promises the report refuses to draw a frontier from
            # single-run measurements; this is that refusal, not a warning.
            raise ValueError(
                "config %s has %d replication(s); a frontier needs >= 2 so latency "
                "variance is known" % (m.config.name, m.replications)
            )

    if task == "aggregate":
        quality: Callable[[Measurement], float] = lambda m: m.aggregate_quality
    else:
        if task not in measurements[0].task_accuracy:
            raise KeyError("unknown task %r" % (task,))
        quality = lambda m: m.task_accuracy[task]

    frontier: List[Measurement] = []
    dominated: List[Tuple[Measurement, Measurement]] = []
    for cand in measurements:
        winner = next((o for o in measurements if o is not cand and _dominates(o, cand, quality)), None)
        if winner is None:
            frontier.append(cand)
        else:
            dominated.append((cand, winner))
    return ParetoResult(task=task, frontier=frontier, dominated=dominated)


def frontier_by_task(measurements: Sequence[Measurement]) -> Dict[str, ParetoResult]:
    tasks = ["aggregate"] + list(measurements[0].task_accuracy)
    return {t: pareto_frontier(measurements, t) for t in tasks}


def recommend(
    measurements: Sequence[Measurement],
    task: str,
    min_quality: float,
) -> Optional[Measurement]:
    """Smallest-memory frontier config for `task` that still meets a quality bar."""
    res = pareto_frontier(measurements, task)
    if task == "aggregate":
        q: Callable[[Measurement], float] = lambda m: m.aggregate_quality
    else:
        q = lambda m: m.task_accuracy[task]
    ok = [m for m in res.frontier if q(m) >= min_quality]
    if not ok:
        return None
    return min(ok, key=lambda m: (m.memory_mb, m.latency_ms_mean))


# --------------------------------------------------------------------------
# Eval-set sizing
# --------------------------------------------------------------------------


def min_eval_size(
    p1: float,
    p2: float,
    alpha: float = DEFAULT_ALPHA,
    power: float = DEFAULT_POWER,
) -> int:
    """Items per config needed to call a p1-vs-p2 accuracy gap real.

    Standard two-proportion normal approximation with a pooled null variance:

        n = ( z_{1-a/2} sqrt(2 p_bar q_bar) + z_{power} sqrt(p1 q1 + p2 q2) )^2
            / (p1 - p2)^2

    Returned as items per arm. Infinite (represented as a very large int) when
    the two proportions are equal, which is the honest answer: no sample size
    distinguishes configs that do not differ.
    """
    if not (0.0 <= p1 <= 1.0 and 0.0 <= p2 <= 1.0):
        raise ValueError("proportions must be in [0, 1]")
    delta = abs(p1 - p2)
    if delta == 0.0:
        return 10 ** 9
    nd = NormalDist()
    z_alpha = nd.inv_cdf(1.0 - alpha / 2.0)
    z_power = nd.inv_cdf(power)
    pbar = (p1 + p2) / 2.0
    null_term = z_alpha * math.sqrt(2.0 * pbar * (1.0 - pbar))
    alt_term = z_power * math.sqrt(p1 * (1.0 - p1) + p2 * (1.0 - p2))
    return max(1, math.ceil(((null_term + alt_term) ** 2) / (delta ** 2)))


@dataclass(frozen=True)
class SizingRow:
    task: str
    lower: str
    upper: str
    p_lower: float
    p_upper: float
    n_required: int


def adjacent_sizing(measurements: Sequence[Measurement], task: str) -> List[SizingRow]:
    """Required eval-set size for every adjacent pair in an ordered sweep."""
    rows: List[SizingRow] = []
    for a, b in zip(measurements, measurements[1:]):
        if task == "aggregate":
            pa, pb = a.aggregate_quality, b.aggregate_quality
        else:
            pa, pb = a.task_accuracy[task], b.task_accuracy[task]
        rows.append(
            SizingRow(
                task=task,
                lower=a.config.name,
                upper=b.config.name,
                p_lower=pa,
                p_upper=pb,
                n_required=min_eval_size(pa, pb),
            )
        )
    return rows
