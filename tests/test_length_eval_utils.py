import unittest

import numpy as np

from route_rangers.eval.length_utils import (
    aggregate_length_metrics,
    bin_name_for_length,
    gap_decision_from_seed_values,
    parse_bins,
)


class LengthEvalUtilsTests(unittest.TestCase):
    def test_bin_assignment_boundaries(self):
        bins = np.asarray([50, 200], dtype=np.int64)
        self.assertEqual(bin_name_for_length(10, bins), "short")
        self.assertEqual(bin_name_for_length(50, bins), "short")
        self.assertEqual(bin_name_for_length(51, bins), "medium")
        self.assertEqual(bin_name_for_length(200, bins), "medium")
        self.assertEqual(bin_name_for_length(201, bins), "long")

    def test_parse_bins_fixed_and_quantile(self):
        fixed, strategy = parse_bins("40,120", np.asarray([10, 20, 30], dtype=np.int64))
        self.assertEqual(strategy, "fixed")
        self.assertTrue(np.array_equal(fixed, np.asarray([40, 120], dtype=np.int64)))

        qbins, qstrategy = parse_bins("", np.asarray([10, 20, 30, 40, 50], dtype=np.int64))
        self.assertEqual(qstrategy, "quantile")
        self.assertEqual(qbins.shape[0], 2)
        self.assertLessEqual(qbins[0], qbins[1])

    def test_aggregate_metrics_and_gap_decision(self):
        seed_metrics = [
            {"short": {"dest_top1": 0.60}, "long": {"dest_top1": 0.62}},
            {"short": {"dest_top1": 0.58}, "long": {"dest_top1": 0.61}},
            {"short": {"dest_top1": 0.59}, "long": {"dest_top1": 0.60}},
        ]
        agg = aggregate_length_metrics(seed_metrics, ci_method="seed")
        self.assertIn("short", agg)
        self.assertIn("dest_top1", agg["short"])
        self.assertGreater(agg["short"]["dest_top1"]["mean"], 0.0)

        short_vals = [0.60, 0.58, 0.59]
        long_vals = [0.62, 0.61, 0.60]
        decision = gap_decision_from_seed_values(
            short_vals, long_vals, variability_k=1.0, tolerance=0.0
        )
        self.assertIn("gap_mean", decision)
        self.assertIn("pass", decision)
        self.assertEqual(decision["n"], 3)


if __name__ == "__main__":
    unittest.main()
