import unittest

import numpy as np

from route_rangers.eval.od_utils import compute_od_tensor, metric_mae_rmse_mape


class ODUtilsTests(unittest.TestCase):
    def test_compute_od_tensor_counts(self):
        origin_idx = np.asarray([0, 0, 1, 1], dtype=np.int64)
        dest_idx = np.asarray([1, 1, 0, 1], dtype=np.int64)
        time_idx = np.asarray([0, 0, 1, 1], dtype=np.int64)
        od = compute_od_tensor(origin_idx, dest_idx, time_idx, num_times=3, num_zones=3)
        self.assertEqual(od.shape, (3, 3, 3))
        self.assertEqual(float(od[0, 0, 1]), 2.0)
        self.assertEqual(float(od[1, 1, 0]), 1.0)
        self.assertEqual(float(od[1, 1, 1]), 1.0)

    def test_mae_rmse_mape(self):
        target = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
        pred = np.asarray([[1.0, 1.0], [5.0, 4.0]], dtype=np.float64)
        m = metric_mae_rmse_mape(pred, target, eps=1e-6)
        self.assertIn("mae", m)
        self.assertIn("rmse", m)
        self.assertIn("mape", m)
        self.assertGreater(m["rmse"], 0.0)


if __name__ == "__main__":
    unittest.main()
