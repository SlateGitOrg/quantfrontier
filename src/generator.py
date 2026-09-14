"""Synthetic data with planted ground truth.

Two generators live here:

  * make_weight_tensor -- a float matrix whose channel scales are deliberately
    heavy-tailed, with a known set of planted outlier channels. Per-channel
    quantisation should recover those channels well and per-tensor should not,
    and the planted set is what the test checks against.

  * make_task_population -- an eval-item population for a downstream task. Each
    item carries a planted "robustness margin": the reconstruction error at
    which that item stops being answered correctly. The shape of the margin
    distribution is the ground truth a cliff-detector has to recover.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import List, Sequence

from .quantise import Matrix


def make_weight_tensor(
    channels: int,
    width: int,
    seed: int,
    outlier_channels: int = 0,
    outlier_gain: float = 20.0,
) -> tuple[Matrix, List[int]]:
    """Gaussian weights with a few channels scaled up by outlier_gain.

    Returns (tensor, planted_outlier_channel_indices).

    The gain of 20x is not arbitrary: measured activation/weight outliers in
    transformer feed-forward layers sit roughly one to two orders of magnitude
    above the bulk, and 20x is the low end of that band. Anything much smaller
    and per-tensor quantisation would not visibly suffer, which would make the
    per-channel comparison vacuous.
    """
    rng = random.Random(seed)
    planted = sorted(rng.sample(range(channels), outlier_channels)) if outlier_channels else []
    planted_set = set(planted)
    tensor: Matrix = []
    for c in range(channels):
        # Channel-level scale heterogeneity exists even without planted
        # outliers; a lognormal spread of about 1.3x keeps the bulk realistic.
        base = math.exp(rng.gauss(0.0, 0.25))
        gain = outlier_gain if c in planted_set else 1.0
        tensor.append([rng.gauss(0.0, base * gain) for _ in range(width)])
    return tensor, planted


@dataclass(frozen=True)
class TaskSpec:
    """A downstream task defined by how its items respond to weight error.

    margin_median  -- relative reconstruction error at which half the items fail.
    margin_log_sd  -- spread of per-item margins on a log scale. SMALL spread is
                      what produces a cliff: every item fails at nearly the same
                      error, so accuracy drops from ~1.0 to ~0.0 across a narrow
                      band of bit-width. LARGE spread produces the smooth,
                      forgiving curve people expect from "quantisation degrades
                      gracefully".
    floor          -- accuracy of random guessing / trivially-correct items.
    """

    name: str
    margin_median: float
    margin_log_sd: float
    floor: float = 0.0

    def __post_init__(self) -> None:
        if self.margin_median <= 0:
            raise ValueError("margin_median must be positive")
        if self.margin_log_sd <= 0:
            raise ValueError("margin_log_sd must be positive")


# The four tasks are chosen to span the real spread of quantisation
# sensitivity reported in the literature: structured/JSON generation and
# multi-step arithmetic collapse abruptly because a single mis-sampled token
# invalidates the whole answer, while summarisation-style scoring degrades
# smoothly because partial credit exists.
TASK_BATTERY: tuple[TaskSpec, ...] = (
    TaskSpec("json_schema_valid", margin_median=0.085, margin_log_sd=0.12, floor=0.02),
    TaskSpec("multistep_arithmetic", margin_median=0.110, margin_log_sd=0.30, floor=0.05),
    TaskSpec("longctx_retrieval", margin_median=0.160, margin_log_sd=0.50, floor=0.10),
    TaskSpec("summarisation_rouge", margin_median=0.300, margin_log_sd=0.90, floor=0.25),
)


def make_task_population(spec: TaskSpec, size: int, seed: int) -> List[float]:
    """Draw `size` per-item robustness margins for a task.

    The same population is reused across every quantisation config, which makes
    every comparison paired: differences between configs are not contaminated
    by having sampled different items.
    """
    rng = random.Random(seed)
    mu = math.log(spec.margin_median)
    return [math.exp(rng.gauss(mu, spec.margin_log_sd)) for _ in range(size)]


def score_population(margins: Sequence[float], error: float, floor: float) -> float:
    """Accuracy of a task population at a given weight reconstruction error.

    An item is answered correctly if the error stayed below its margin. Items
    below their margin also include a `floor` fraction that come out right by
    luck even when the capability is gone -- without it, a broken structured-
    output task would score an implausible exact zero.
    """
    if not margins:
        return 0.0
    survived = sum(1 for m in margins if error < m)
    frac = survived / len(margins)
    return frac + (1.0 - frac) * floor
