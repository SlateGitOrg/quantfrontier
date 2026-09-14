import statistics
import unittest

from src.backend import LATENCY_NOISE_FRAC, Measurement, ModelBackend, default_sweep_configs
from src.frontier import (
    degradation_profile, min_eval_size, pareto_frontier, profile_sweep, recommend, select_on_metric,
)
from src.generator import TASK_BATTERY
from src.quantise import QuantConfig


def _m(bits, q, lat, mem, sd=0.01, reps=10):
    return Measurement(QuantConfig(bits), 0.0, 1.0, {"t": q}, lat, sd, reps, mem)


class TestCliffStatistic(unittest.TestCase):
    def test_step_vs_linear(self):
        labels = list("abcdef")
        step = degradation_profile(labels, [1, 1, 1, 1, 0, 0], "s")
        lin = degradation_profile(labels, [1, .8, .6, .4, .2, 0], "l")
        self.assertEqual(step.steps_to_share(0.9), 1)
        self.assertEqual(step.cliff_at, "e")
        self.assertEqual(lin.steps_to_share(0.9), 5)
        self.assertAlmostEqual(lin.cliff_share, lin.smooth_share)


class TestSimulatedSweep(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        b = ModelBackend()
        cls.backend = b
        cls.all = b.sweep(default_sweep_configs())
        cls.ladder = [m for m in cls.all if m.config.per_channel and m.config.outlier_fraction == 0]
        cls.prof = profile_sweep(cls.ladder)

    def test_structured_task_has_cliff_that_aggregate_hides(self):
        self.assertEqual(self.prof["json_schema_valid"].steps_to_share(0.9), 1)
        self.assertGreaterEqual(self.prof["aggregate"].steps_to_share(0.9), 3)
        self.assertTrue(self.prof["aggregate"].monotone_non_increasing)

    def test_cliff_location_matches_planted_margin(self):
        # Planted margin median must fall between q5 and q4 measured error,
        # so the cliff must land on q4 -- derived, not tuned.
        spec = next(t for t in TASK_BATTERY if t.name == "json_schema_valid")
        by = {m.config.bits: m for m in self.ladder}
        self.assertLess(by[5].weight_error, spec.margin_median)
        self.assertGreater(by[4].weight_error, spec.margin_median)
        self.assertEqual(self.prof["json_schema_valid"].cliff_at, "q4-chan-sym")

    def test_aggregate_selection_ships_broken_structured_output(self):
        pick = select_on_metric(self.all, lambda m: m.aggregate_quality, 0.80)
        self.assertEqual(pick.config.name, "q4-chan-sym")
        self.assertLess(pick.task_accuracy["json_schema_valid"], 0.5)
        safe = recommend(self.all, "json_schema_valid", 0.95)
        self.assertGreater(safe.config.bits, pick.config.bits)

    def test_frontier_differs_by_task(self):
        sets = {t: frozenset(pareto_frontier(self.all, t).frontier_names)
                for t in ["aggregate", "json_schema_valid", "summarisation_rouge"]}
        self.assertEqual(len(set(sets.values())), 3)
        for t in sets:
            self.assertNotIn("q4-tens-sym", sets[t])  # q3-chan: smaller, faster, better

    def test_frontier_refuses_single_run(self):
        single = self.backend.sweep(default_sweep_configs()[:3], replications=1)
        with self.assertRaises(ValueError):
            pareto_frontier(single, "aggregate")

    def test_latency_replication_variance_matches_stated_noise(self):
        # Pooled CV over 11 configs x 12 reps; relative SE of a pooled sd with
        # 121 dof is ~6.5%, so +-20% is ~3 SE around the stated 2%.
        cvs = [m.latency_ms_sd / m.latency_ms_mean for m in self.all]
        pooled = statistics.fmean(c * c for c in cvs) ** 0.5
        self.assertAlmostEqual(pooled, LATENCY_NOISE_FRAC, delta=0.2 * LATENCY_NOISE_FRAC)

    def test_deterministic_across_runs(self):
        again = ModelBackend().sweep(default_sweep_configs())
        self.assertEqual([m.latency_ms_mean for m in again], [m.latency_ms_mean for m in self.all])


class TestDominance(unittest.TestCase):
    def test_latency_tie_within_noise_does_not_dominate(self):
        a = _m(4, 0.9, 10.00, 3.0, sd=0.5)
        b = _m(5, 0.9, 10.05, 3.0, sd=0.5)
        self.assertEqual(len(pareto_frontier([a, b], "t").frontier), 2)

    def test_clear_dominance(self):
        a = _m(4, 0.9, 10.0, 3.0)
        b = _m(5, 0.8, 12.0, 4.0)
        res = pareto_frontier([a, b], "t")
        self.assertEqual(len(res.frontier), 1)
        self.assertIs(res.frontier[0], a)


class TestSizing(unittest.TestCase):
    def test_textbook_value(self):
        # Standard pooled two-proportion formula, alpha .05 two-sided, power .8.
        self.assertEqual(min_eval_size(0.90, 0.85), 686)

    def test_equal_proportions_unresolvable(self):
        self.assertGreater(min_eval_size(0.5, 0.5), 10 ** 8)

    def test_smaller_gap_needs_more_items(self):
        self.assertGreater(min_eval_size(0.90, 0.88), min_eval_size(0.90, 0.80))


if __name__ == "__main__":
    unittest.main()
