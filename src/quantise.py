"""Real affine quantise/dequantise arithmetic on plain Python float matrices.

Nothing here is simulated. A tensor is a list of rows (output channels), each a
list of floats, and every number this module reports is a measured property of
an actual round-trip through an integer grid.

Two axes of choice are implemented because they are the two that decide whether
a 4-bit weight matrix is usable:

  * granularity  -- one scale for the whole tensor, or one scale per channel.
  * scheme       -- symmetric (zero-point pinned to 0) or asymmetric affine.

plus optional outlier retention, where the largest-magnitude entries are kept at
full precision and excluded from the range used to pick the scale.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Sequence, Tuple

Matrix = List[List[float]]

# A quantisation grid needs at least two representable levels, and below 2 bits
# "round to nearest grid point" is not a meaningful operation.
MIN_BITS = 2
MAX_BITS = 16


@dataclass(frozen=True)
class QuantConfig:
    """One point in the quantisation design space."""

    bits: int
    per_channel: bool = True
    symmetric: bool = True
    # Fraction of weights (per scale group) kept in fp16 rather than quantised.
    # 0.0 disables outlier handling. Real schemes (LLM.int8, AWQ) keep a
    # comparable sliver; 0.1% - 1% is the range that appears in practice.
    outlier_fraction: float = 0.0

    def __post_init__(self) -> None:
        if not (MIN_BITS <= self.bits <= MAX_BITS):
            raise ValueError("bits must be in [%d, %d], got %r" % (MIN_BITS, MAX_BITS, self.bits))
        if not (0.0 <= self.outlier_fraction < 0.5):
            raise ValueError("outlier_fraction must be in [0, 0.5), got %r" % (self.outlier_fraction,))

    @property
    def name(self) -> str:
        gran = "chan" if self.per_channel else "tens"
        scheme = "sym" if self.symmetric else "asym"
        tag = "q%d-%s-%s" % (self.bits, gran, scheme)
        if self.outlier_fraction > 0:
            tag += "-out%.1f%%" % (self.outlier_fraction * 100.0)
        return tag


def _round_half_away(x: float) -> float:
    """Round-half-away-from-zero.

    Python's built-in round() is banker's rounding, which biases a quantiser:
    on a symmetric grid it pulls exact .5 values toward even codes and leaves a
    small but systematic DC offset in the reconstruction. Quantisation kernels
    round half away from zero, so do the same.
    """
    return math.floor(x + 0.5) if x >= 0 else math.ceil(x - 0.5)


def _select_outliers(values: Sequence[float], fraction: float) -> List[int]:
    """Indices of the largest-magnitude entries, as a sorted list."""
    if fraction <= 0.0:
        return []
    k = int(len(values) * fraction)
    if k == 0:
        return []
    order = sorted(range(len(values)), key=lambda i: -abs(values[i]))
    return sorted(order[:k])


def quantise_group(values: Sequence[float], cfg: QuantConfig) -> Tuple[List[float], float, int]:
    """Quantise-dequantise one scale group. Returns (reconstruction, scale, zero_point).

    The group is whatever shares a scale: one channel under per-channel
    granularity, the flattened tensor under per-tensor.
    """
    n = len(values)
    if n == 0:
        return [], 1.0, 0

    outliers = set(_select_outliers(values, cfg.outlier_fraction))
    # The whole point of keeping outliers in fp is that they no longer stretch
    # the range the remaining weights have to share. Excluding them from the
    # min/max is what buys the accuracy, not the fp storage itself.
    body = [values[i] for i in range(n) if i not in outliers]
    if not body:
        body = list(values)

    levels = (1 << cfg.bits) - 1

    if cfg.symmetric:
        qmax = (1 << (cfg.bits - 1)) - 1
        qmin = -(1 << (cfg.bits - 1))
        amax = max(abs(v) for v in body)
        scale = amax / qmax if amax > 0 else 1.0
        zero_point = 0
    else:
        lo = min(body)
        hi = max(body)
        qmin, qmax = 0, levels
        span = hi - lo
        scale = span / levels if span > 0 else 1.0
        # The zero-point is an integer OFFSET, not itself a code, so it must not
        # be clamped into [qmin, qmax]. Clamping it (a common copy-paste bug,
        # and the first one this suite caught) is harmless only when the value
        # range straddles zero; on an all-positive group it collapses the whole
        # group onto the wrong part of the grid and asymmetric quantisation
        # comes out an order of magnitude WORSE than symmetric.
        zero_point = int(_round_half_away(qmin - lo / scale))

    out: List[float] = []
    for i, v in enumerate(values):
        if i in outliers:
            out.append(v)
            continue
        q = _round_half_away(v / scale) + zero_point
        q = max(qmin, min(qmax, q))
        out.append((q - zero_point) * scale)
    return out, scale, zero_point


def quantise_matrix(tensor: Matrix, cfg: QuantConfig) -> Matrix:
    """Quantise-dequantise a whole matrix under cfg."""
    if cfg.per_channel:
        return [quantise_group(row, cfg)[0] for row in tensor]
    flat = [v for row in tensor for v in row]
    recon, _, _ = quantise_group(flat, cfg)
    out: Matrix = []
    pos = 0
    for row in tensor:
        out.append(recon[pos:pos + len(row)])
        pos += len(row)
    return out


def rmse(a: Matrix, b: Matrix) -> float:
    total = 0.0
    count = 0
    for ra, rb in zip(a, b):
        for x, y in zip(ra, rb):
            d = x - y
            total += d * d
            count += 1
    return math.sqrt(total / count) if count else 0.0


def frobenius(tensor: Matrix) -> float:
    total = 0.0
    for row in tensor:
        for v in row:
            total += v * v
    return math.sqrt(total)


def relative_error(original: Matrix, recon: Matrix) -> float:
    """||W - W_hat||_F / ||W||_F -- the scale-free reconstruction error.

    Scale-free matters: the downstream quality model consumes this number, and
    it must not change if somebody rescales the weights.
    """
    denom = frobenius(original)
    if denom == 0.0:
        return 0.0
    diff = [[x - y for x, y in zip(ra, rb)] for ra, rb in zip(original, recon)]
    return frobenius(diff) / denom


def per_channel_errors(original: Matrix, recon: Matrix) -> List[float]:
    """Relative error per row, for attributing tensor-wide error to channels."""
    out = []
    for ra, rb in zip(original, recon):
        denom = math.sqrt(sum(v * v for v in ra))
        if denom == 0.0:
            out.append(0.0)
            continue
        num = math.sqrt(sum((x - y) ** 2 for x, y in zip(ra, rb)))
        out.append(num / denom)
    return out


def bits_per_weight(cfg: QuantConfig, group_size: int) -> float:
    """Effective storage cost including scale/zero-point and fp16 outliers.

    A per-channel scheme stores one fp16 scale (and an int zero-point for the
    asymmetric variant) per group; that overhead is negligible for a 4096-wide
    channel and material for a 64-wide one, so it is charged explicitly rather
    than assumed away.
    """
    if group_size <= 0:
        raise ValueError("group_size must be positive")
    overhead_bits = 16.0  # fp16 scale
    if not cfg.symmetric:
        overhead_bits += 16.0  # zero-point stored alongside
    per_weight = cfg.bits + overhead_bits / group_size
    # Outliers cost fp16 for the value plus a 16-bit index to locate it.
    per_weight += cfg.outlier_fraction * 32.0
    return per_weight
