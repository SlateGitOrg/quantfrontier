"""Deterministic simulated inference backend.

THERE IS NO MODEL HERE. No weights, no runtime, no GPU, no network. What this
module provides is a seeded, reproducible stand-in that maps a quantisation
config to (perplexity, per-task accuracy, latency, memory) with the error
characteristics real quantisation has.

The one thing that is NOT simulated is the input to that map: the weight
reconstruction error is measured by actually quantising real float matrices in
`quantise.py`. So the chain is

    real float tensors -> real integer round-trip -> MEASURED relative error
        -> simulated task response -> quality/latency/memory

and the part under test in `tests/` is the measurement methodology built on top:
cliff detection, per-task Pareto frontiers, replication variance, and the eval
set size needed to tell two configs apart. Swapping this backend for llama.cpp
means replacing `ModelBackend` and changing nothing else.
"""

from __future__ import annotations

import math
import random
import statistics
import zlib
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

from .generator import (
    TASK_BATTERY,
    TaskSpec,
    make_task_population,
    make_weight_tensor,
    score_population,
)
from .quantise import (
    Matrix,
    QuantConfig,
    bits_per_weight,
    per_channel_errors,
    quantise_matrix,
)

# Full-precision reference perplexity. Any positive constant works; it only sets
# the units of the aggregate metric, and every claim made about it is relative.
FP_PERPLEXITY = 8.00

# Sensitivity of perplexity to weight error. Perplexity is an average over a
# huge number of next-token predictions, so it is dominated by easy,
# high-margin tokens and moves very little: a ~4% relative-error reconstruction
# (4-bit) costs single-digit percent perplexity in published sweeps. k=1.5
# reproduces that magnitude. This gentleness is exactly the trap the project
# exists to expose, so it is a modelling choice stated out loud, not a fudge.
PERPLEXITY_SENSITIVITY = 1.5

# Decode is memory-bandwidth bound: roughly 400 GB/s of usable bandwidth on the
# kind of on-premise card this project targets.
BANDWIDTH_GB_PER_S = 400.0
# Fixed per-token cost quantisation cannot remove (attention, sampling, launch).
FIXED_OVERHEAD_MS = 1.20
# Dequantisation cost per bit width. 8-bit is a plain byte load. 4-bit is two
# nibbles per byte -- one shift and mask. 6- and 3-bit do not divide a byte, so
# the kernel unpacks across byte boundaries and pays materially more. This is
# why latency is NOT monotone in bit width, and it is what makes the frontier
# non-trivial.
DEQUANT_MS: Dict[int, float] = {16: 0.00, 8: 0.05, 6: 0.38, 5: 0.34, 4: 0.11, 3: 0.41, 2: 0.14}
# Timing noise as a fraction of the mean. Latency timing is noisy at roughly the
# 2% level even with pinned threads; unreplicated numbers at that noise level
# cannot resolve the sub-percent gaps between neighbouring configs, which is why
# the frontier refuses single-run input.
LATENCY_NOISE_FRAC = 0.02


def _stable_seed(*parts: object) -> int:
    """Process-stable seed.

    Python randomises hash() for strings per process, so deriving a seed from
    hash(cfg.name) would make a "deterministic" run differ between processes.
    crc32 does not have that problem.
    """
    return zlib.crc32("|".join(str(p) for p in parts).encode("ascii"))


@dataclass(frozen=True)
class Measurement:
    """One config, fully measured."""

    config: QuantConfig
    weight_error: float
    perplexity: float
    task_accuracy: Dict[str, float]
    latency_ms_mean: float
    latency_ms_sd: float
    replications: int
    memory_mb: float
    # Perplexity of the full-precision model the aggregate is normalised to.
    reference_perplexity: float = FP_PERPLEXITY
    # Mean KL(P_fp || P_quant) in nats; only measured by real forward passes.
    kl_divergence: float = float("nan")

    @property
    def latency_sem(self) -> float:
        if self.replications < 2:
            return float("nan")
        return self.latency_ms_sd / math.sqrt(self.replications)

    @property
    def aggregate_quality(self) -> float:
        """Perplexity on a 0-1 higher-is-better scale, so the aggregate metric
        can be compared on the same axis as task accuracy. FP maps to 1.0."""
        return self.reference_perplexity / self.perplexity


class ModelBackend:
    """Simulated quantisable model with real weight tensors underneath."""

    def __init__(
        self,
        seed: int = 20240,
        layers: int = 6,
        channels: int = 96,
        width: int = 128,
        outlier_channels: int = 4,
        eval_size: int = 2000,
        tasks: Sequence[TaskSpec] = TASK_BATTERY,
        deployed_params: float = 7.0e9,
    ) -> None:
        self.seed = seed
        self.tasks = tuple(tasks)
        self.group_size = width
        self.tensors: List[Matrix] = []
        self.planted_outliers: List[List[int]] = []
        for layer in range(layers):
            tensor, planted = make_weight_tensor(
                channels, width, seed=seed + 1000 * layer, outlier_channels=outlier_channels
            )
            self.tensors.append(tensor)
            self.planted_outliers.append(planted)
        self.param_count = layers * channels * width
        # Reconstruction error converges quickly in the number of weights, so a
        # few tens of thousands of them measure it accurately. Memory and
        # latency do NOT converge -- they scale linearly -- so they are reported
        # for a deployment-sized model (7B by default). Measuring error on a
        # small tensor and projecting cost to the real parameter count is the
        # honest split; pretending to hold 7B floats in a Python list is not.
        self.deployed_params = deployed_params
        # Populations are drawn once and reused for every config, so every
        # config-to-config comparison is paired on identical items.
        self.populations: Dict[str, List[float]] = {
            t.name: make_task_population(t, eval_size, seed=seed + 77 * i)
            for i, t in enumerate(self.tasks)
        }

    def measure_weight_error(self, cfg: QuantConfig) -> float:
        """Weight reconstruction error over all layers -- measured, not modelled.

        Deliberately the EQUAL-WEIGHTED mean of per-channel relative errors, not
        the tensor-wide Frobenius relative error. Under per-tensor quantisation
        a handful of high-magnitude outlier channels carry most of the Frobenius
        norm and reconstruct perfectly well, so a norm-weighted figure reports
        ~0.25 while 95% of the channels have been rounded to literal zero
        (relative error 1.0). That is the same aggregation mistake perplexity
        makes, one level down, and this project should not make it twice.
        """
        errs: List[float] = []
        for tensor in self.tensors:
            errs.extend(per_channel_errors(tensor, quantise_matrix(tensor, cfg)))
        return statistics.fmean(errs) if errs else 0.0

    def memory_mb(self, cfg: QuantConfig) -> float:
        # Per-tensor keeps one scale for an entire matrix, per-channel one per
        # row. Charging both the same overhead would hide the (small) real cost
        # of per-channel scaling and make the comparison flattering.
        group = self.group_size if cfg.per_channel else self.group_size * len(self.tensors[0])
        return self.deployed_params * bits_per_weight(cfg, group) / 8.0 / 1e6

    def _latency_mean(self, cfg: QuantConfig) -> float:
        bytes_moved = self.memory_mb(cfg) * 1e6
        transfer_ms = bytes_moved / (BANDWIDTH_GB_PER_S * 1e9) * 1e3
        dequant = DEQUANT_MS.get(cfg.bits, 0.30)
        # Sparse fp16 outliers need a gather/scatter pass: expensive relative to
        # how few values it touches.
        outlier_ms = cfg.outlier_fraction * 45.0
        return FIXED_OVERHEAD_MS + transfer_ms + dequant + outlier_ms

    def measure_latency(self, cfg: QuantConfig, replications: int) -> tuple[float, float]:
        """Replicated timing. Returns (mean, sample sd)."""
        if replications < 1:
            raise ValueError("replications must be >= 1")
        rng = random.Random(_stable_seed(self.seed, cfg.name, replications))
        true_mean = self._latency_mean(cfg)
        samples = [true_mean * (1.0 + rng.gauss(0.0, LATENCY_NOISE_FRAC)) for _ in range(replications)]
        sd = statistics.stdev(samples) if replications > 1 else 0.0
        return statistics.fmean(samples), sd

    def perplexity(self, weight_error: float) -> float:
        return FP_PERPLEXITY * math.exp(PERPLEXITY_SENSITIVITY * weight_error)

    def task_accuracies(self, weight_error: float, eval_size: int | None = None) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for spec in self.tasks:
            pop = self.populations[spec.name]
            if eval_size is not None:
                pop = pop[:eval_size]
            out[spec.name] = score_population(pop, weight_error, spec.floor)
        return out

    def measure(self, cfg: QuantConfig, replications: int = 12) -> Measurement:
        err = self.measure_weight_error(cfg)
        mean, sd = self.measure_latency(cfg, replications)
        return Measurement(
            config=cfg,
            weight_error=err,
            perplexity=self.perplexity(err),
            task_accuracy=self.task_accuracies(err),
            latency_ms_mean=mean,
            latency_ms_sd=sd,
            replications=replications,
            memory_mb=self.memory_mb(cfg),
        )

    def sweep(self, configs: Iterable[QuantConfig], replications: int = 12) -> List[Measurement]:
        return [self.measure(c, replications) for c in configs]


def default_sweep_configs() -> List[QuantConfig]:
    """The bit widths a llama.cpp user actually chooses between, plus the
    per-tensor variants that show what the cheap implementation costs."""
    configs: List[QuantConfig] = [QuantConfig(bits=16, per_channel=True, symmetric=True)]
    for bits in (8, 6, 5, 4, 3):
        configs.append(QuantConfig(bits=bits, per_channel=True, symmetric=True))
    for bits in (8, 4, 3):
        configs.append(QuantConfig(bits=bits, per_channel=False, symmetric=True))
    for bits in (4, 3):
        configs.append(QuantConfig(bits=bits, per_channel=True, symmetric=True, outlier_fraction=0.01))
    return configs
