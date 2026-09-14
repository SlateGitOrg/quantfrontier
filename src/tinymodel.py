"""A real, runnable quantisation target: a tiny MLP with planted ground truth.

Unlike `backend.ModelBackend` (which simulates the error->accuracy map), every
number here is MEASURED by running the quantised network forward:

  * weights are quantised with `quantise.quantise_matrix` (real integer grid),
  * task accuracy is argmax agreement with labels on a held-out eval set,
  * KL divergence is computed between full-precision and quantised softmaxes,
  * perplexity is exp(mean NLL) on the same labels,
  * latency is wall-clock, replicated, and reported only (never asserted).

Planted ground truth:
  * The labels are the teacher's argmax with a KNOWN flip rate `label_noise`,
    so the full-precision accuracy has a closed-form expectation (1 - noise)
    and the structured task (all `record_len` fields right) has (1 - noise)^k.
    This is the "validate the harness against a published baseline" step.
  * Hidden layer rows carry planted 20x outlier channels, compensated by 1/20 in
    the next layer so the network FUNCTION is unchanged -- the same function-
    preserving rescaling that makes real LLM weights heavy-tailed. Per-tensor
    quantisation is fooled by it; per-channel is not.

How a real model plugs in: implement `forward(x) -> logits` over quantised
weights (e.g. llama.cpp logits for GGUF q8_0/q5_K/q4_K) and feed the same
`evaluate_config` bookkeeping. Nothing downstream of `Measurement` changes.
"""

from __future__ import annotations

import math
import random
import statistics
import time
from typing import Dict, List, Sequence

from .backend import Measurement
from .quantise import Matrix, QuantConfig, bits_per_weight, quantise_matrix

OUTLIER_GAIN = 20.0


def _matvec(w: Matrix, x: Sequence[float]) -> List[float]:
    return [sum(a * b for a, b in zip(row, x)) for row in w]


def _log_softmax(z: Sequence[float]) -> List[float]:
    m = max(z)
    lse = m + math.log(sum(math.exp(v - m) for v in z))
    return [v - lse for v in z]


class TinyMLP:
    def __init__(self, seed: int = 11, d_in: int = 16, hidden: int = 48, classes: int = 10,
                 outlier_rows: int = 3, eval_size: int = 1600, label_noise: float = 0.05,
                 record_len: int = 8) -> None:
        rng = random.Random(seed)
        self.label_noise = label_noise
        self.record_len = record_len
        self.planted_outliers = sorted(rng.sample(range(hidden), outlier_rows))
        w1 = [[rng.gauss(0, 1) for _ in range(d_in)] for _ in range(hidden)]
        w2 = [[rng.gauss(0, 1) for _ in range(hidden)] for _ in range(classes)]
        for r in self.planted_outliers:
            w1[r] = [v * OUTLIER_GAIN for v in w1[r]]
            for row in w2:
                row[r] /= OUTLIER_GAIN
        self.w1, self.w2 = w1, w2
        self.inputs = [[rng.gauss(0, 1) for _ in range(d_in)] for _ in range(eval_size)]
        fp_logits = [self._forward(w1, w2, x) for x in self.inputs]
        labels = []
        for z in fp_logits:
            y = max(range(classes), key=z.__getitem__)
            if rng.random() < label_noise:
                y = rng.choice([c for c in range(classes) if c != y])
            labels.append(y)
        self.labels = labels
        self.fp_logprobs = [_log_softmax(z) for z in fp_logits]
        self.n_params = hidden * d_in + classes * hidden
        self.fp_perplexity = self._perplexity(self.fp_logprobs)

    @staticmethod
    def _forward(w1: Matrix, w2: Matrix, x: Sequence[float]) -> List[float]:
        h = [v if v > 0 else 0.0 for v in _matvec(w1, x)]
        return _matvec(w2, h)

    def _perplexity(self, logprobs: Sequence[Sequence[float]]) -> float:
        nll = -statistics.fmean(lp[y] for lp, y in zip(logprobs, self.labels))
        return math.exp(nll)

    def task_scores(self, logprobs: Sequence[Sequence[float]]) -> Dict[str, float]:
        correct = [max(range(len(lp)), key=lp.__getitem__) == y for lp, y in zip(logprobs, self.labels)]
        k = self.record_len
        records = [all(correct[i:i + k]) for i in range(0, len(correct) - k + 1, k)]
        return {
            "classification": sum(correct) / len(correct),
            "structured_record": sum(records) / len(records),
        }

    def quantised_weights(self, cfg: QuantConfig):
        if cfg.bits >= 16:
            return self.w1, self.w2
        return quantise_matrix(self.w1, cfg), quantise_matrix(self.w2, cfg)

    def mean_kl(self, logprobs: Sequence[Sequence[float]]) -> float:
        """Mean KL(P_fp || P_q) in nats over the eval set."""
        tot = 0.0
        for p, q in zip(self.fp_logprobs, logprobs):
            tot += sum(math.exp(a) * (a - b) for a, b in zip(p, q))
        return tot / len(logprobs)

    def evaluate(self, cfg: QuantConfig, replications: int = 3, timing_items: int = 100) -> Measurement:
        w1, w2 = self.quantised_weights(cfg)
        logprobs = [_log_softmax(self._forward(w1, w2, x)) for x in self.inputs]
        times = []
        for _ in range(replications):
            t0 = time.perf_counter()
            for x in self.inputs[:timing_items]:
                self._forward(w1, w2, x)
            times.append((time.perf_counter() - t0) * 1e3 / timing_items)
        group = len(self.w1[0]) if cfg.per_channel else len(self.w1[0]) * len(self.w1)
        bpw = 16.0 if cfg.bits >= 16 else bits_per_weight(cfg, group)
        return Measurement(
            config=cfg,
            weight_error=0.0,
            perplexity=self._perplexity(logprobs),
            task_accuracy=self.task_scores(logprobs),
            latency_ms_mean=statistics.fmean(times),
            latency_ms_sd=statistics.stdev(times) if replications > 1 else 0.0,
            replications=replications,
            memory_mb=self.n_params * bpw / 8.0 / 1e6,
            reference_perplexity=self.fp_perplexity,
            kl_divergence=self.mean_kl(logprobs),
        )
