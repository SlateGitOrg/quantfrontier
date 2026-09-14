import math
import random
import statistics
import unittest

from src.generator import make_weight_tensor
from src.quantise import (
    QuantConfig, _round_half_away, bits_per_weight, per_channel_errors,
    quantise_group, quantise_matrix,
)


class TestQuantiserArithmetic(unittest.TestCase):
    def test_rms_error_matches_uniform_quantiser_closed_form(self):
        # For data spread uniformly across many steps, RMS error = step/sqrt(12).
        # With 40k samples the sampling error of the ratio is well under 1%;
        # 3% tolerance is ~3x that, not tuned.
        rng = random.Random(7)
        vals = [rng.uniform(-1, 1) for _ in range(40000)]
        for bits in (4, 5, 6, 8):
            rec, scale, _ = quantise_group(vals, QuantConfig(bits, False, True))
            err = math.sqrt(statistics.fmean((a - b) ** 2 for a, b in zip(vals, rec)))
            self.assertAlmostEqual(err / (scale / math.sqrt(12)), 1.0, delta=0.03)

    def test_values_on_the_grid_reconstruct_exactly(self):
        rng = random.Random(3)
        grid = [0.25 * rng.randint(-7, 7) for _ in range(300)] + [1.75]
        rec, scale, _ = quantise_group(grid, QuantConfig(4, False, True))
        self.assertAlmostEqual(scale, 0.25)
        self.assertLess(max(abs(a - b) for a, b in zip(grid, rec)), 1e-12)

    def test_round_half_away_from_zero_not_bankers(self):
        self.assertEqual(_round_half_away(2.5), 3)
        self.assertEqual(_round_half_away(-2.5), -3)
        self.assertEqual(round(2.5), 2)  # the built-in would bias the grid

    def test_asymmetric_beats_symmetric_on_all_positive_data(self):
        # Fails against the zero-point-clamping bug, where asym comes out worse.
        rng = random.Random(1)
        pos = [rng.uniform(3.0, 4.0) for _ in range(5000)]
        def rms(cfg):
            rec, _, _ = quantise_group(pos, cfg)
            return math.sqrt(statistics.fmean((a - b) ** 2 for a, b in zip(pos, rec)))
        sym, asym = rms(QuantConfig(4, False, True)), rms(QuantConfig(4, False, False))
        # asym uses the full 15 steps on a width-1 range, sym ~1.9 steps of it.
        self.assertLess(asym, sym / 4)

    def test_error_shrinks_with_bits(self):
        t, _ = make_weight_tensor(32, 64, seed=5, outlier_channels=2)
        errs = [statistics.fmean(per_channel_errors(t, quantise_matrix(t, QuantConfig(b)))) for b in (8, 6, 4, 3, 2)]
        self.assertEqual(errs, sorted(errs))

    def test_config_validation(self):
        with self.assertRaises(ValueError):
            QuantConfig(bits=1)
        with self.assertRaises(ValueError):
            QuantConfig(bits=4, outlier_fraction=0.6)

    def test_storage_overhead_is_charged(self):
        self.assertAlmostEqual(bits_per_weight(QuantConfig(4, True, True), 128), 4 + 16 / 128)
        self.assertAlmostEqual(bits_per_weight(QuantConfig(4, True, False), 128), 4 + 32 / 128)
        self.assertAlmostEqual(bits_per_weight(QuantConfig(4, True, True, 0.01), 128), 4 + 16 / 128 + 0.32)


class TestPlantedOutliers(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.t, cls.planted = make_weight_tensor(96, 128, seed=1, outlier_channels=4)

    def test_per_tensor_zeroes_the_bulk_but_per_channel_does_not(self):
        pt = per_channel_errors(self.t, quantise_matrix(self.t, QuantConfig(4, False, True)))
        pc = per_channel_errors(self.t, quantise_matrix(self.t, QuantConfig(4, True, True)))
        bulk = [i for i in range(96) if i not in self.planted]
        # Bulk sigma ~1/20 of the global max: with 7 positive codes the step is
        # ~3x a bulk weight's size, so nearly every bulk weight rounds to 0.
        self.assertGreater(statistics.fmean(pt[i] for i in bulk), 0.9)
        self.assertLess(statistics.fmean(pc[i] for i in bulk), 0.2)

    def test_planted_channels_recovered_as_best_reconstructed_under_per_tensor(self):
        pt = per_channel_errors(self.t, quantise_matrix(self.t, QuantConfig(4, False, True)))
        best = sorted(range(96), key=pt.__getitem__)[:4]
        self.assertEqual(sorted(best), self.planted)

    def test_outlier_retention_reduces_error(self):
        base = statistics.fmean(per_channel_errors(self.t, quantise_matrix(self.t, QuantConfig(3))))
        kept = statistics.fmean(per_channel_errors(self.t, quantise_matrix(self.t, QuantConfig(3, outlier_fraction=0.01))))
        self.assertLess(kept, base)


if __name__ == "__main__":
    unittest.main()
