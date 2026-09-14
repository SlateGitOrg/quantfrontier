import math
import unittest

from src.quantise import QuantConfig
from src.tinymodel import TinyMLP


class TestTinyModelRealMeasurement(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = TinyMLP()
        cls.m = {b: cls.model.evaluate(QuantConfig(b), replications=2) for b in (16, 8, 4, 2)}
        cls.q4_tensor = cls.model.evaluate(QuantConfig(4, per_channel=False), replications=2)

    def test_full_precision_baseline_matches_planted_value(self):
        # Harness validation: FP scores must match closed-form expectations
        # from the planted label-noise rate within 3 binomial standard errors.
        acc = self.m[16].task_accuracy
        p, n, k = 1 - self.model.label_noise, len(self.model.labels), self.model.record_len
        self.assertAlmostEqual(acc["classification"], p, delta=3 * math.sqrt(p * (1 - p) / n))
        pr, nr = p ** k, n // k
        self.assertAlmostEqual(acc["structured_record"], pr, delta=3 * math.sqrt(pr * (1 - pr) / nr))

    def test_kl_zero_at_full_precision_and_grows(self):
        self.assertAlmostEqual(self.m[16].kl_divergence, 0.0, places=12)
        self.assertAlmostEqual(self.m[16].aggregate_quality, 1.0)
        self.assertLess(self.m[8].kl_divergence, self.m[4].kl_divergence)
        self.assertLess(self.m[4].kl_divergence, self.m[2].kl_divergence)

    def test_per_tensor_collapses_on_planted_outliers(self):
        pc = self.m[4].task_accuracy["classification"]
        pt = self.q4_tensor.task_accuracy["classification"]
        # Chance is 1/10; per-tensor should fall near it, per-channel well above.
        self.assertLess(pt, 0.3)
        self.assertGreater(pc, 0.6)
        self.assertGreater(self.q4_tensor.kl_divergence, 5 * self.m[4].kl_divergence)

    def test_tasks_degrade_differently(self):
        # Retention = quantised / FP score. A single aggregate would assign
        # both tasks the same fate; the structured task must lose far more.
        fp, q4 = self.m[16].task_accuracy, self.m[4].task_accuracy
        r_cls = q4["classification"] / fp["classification"]
        r_rec = q4["structured_record"] / fp["structured_record"]
        self.assertLess(r_rec, r_cls)
        # Independence approximation: record retention ~ r_cls ** k. Within 2x.
        pred = r_cls ** self.model.record_len
        self.assertLess(abs(math.log(r_rec / pred)), math.log(2))

    def test_memory_is_exact(self):
        n = self.model.n_params
        self.assertAlmostEqual(self.m[16].memory_mb, n * 16 / 8 / 1e6)
        self.assertLess(self.m[4].memory_mb, self.m[8].memory_mb)

    def test_latency_is_reported_with_replications(self):
        m = self.m[4]
        self.assertEqual(m.replications, 2)
        self.assertGreater(m.latency_ms_mean, 0)
        self.assertFalse(math.isnan(m.latency_sem))


if __name__ == "__main__":
    unittest.main()
