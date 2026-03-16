import unittest

import numpy as np
try:
    import torch

    from route_rangers.cli.run_proposal_baselines import evaluate_mean_displacement

    TORCH_AVAILABLE = True
except Exception:  # pragma: no cover
    torch = None
    TORCH_AVAILABLE = False


class _ToyDataset:
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def _make_linear_sample(length: int, dlat: float, dlon: float):
    coords = np.zeros((32, 2), dtype=np.float32)
    ts = np.zeros((32,), dtype=np.float32)
    attn = np.zeros((32,), dtype=np.float32)
    lat0, lon0 = 40.0, -73.0
    for i in range(length):
        coords[i, 0] = lat0 + i * dlat
        coords[i, 1] = lon0 + i * dlon
        ts[i] = float(i)
        attn[i] = 1.0
    return {
        "coords": torch.from_numpy(coords),
        "timestamps": torch.from_numpy(ts),
        "attention_mask": torch.from_numpy(attn),
        "raw_length": length,
        "effective_length": length,
    }


class ProposalBaselineTests(unittest.TestCase):
    @unittest.skipUnless(TORCH_AVAILABLE, "torch not installed")
    def test_mean_displacement_perfect_on_linear_motion(self):
        ds = _ToyDataset(
            [
                _make_linear_sample(8, 1e-4, 2e-4),
                _make_linear_sample(10, 2e-4, 1e-4),
            ]
        )
        bins = np.asarray([8, 9], dtype=np.int64)
        out = evaluate_mean_displacement(
            ds,
            indices=[0, 1],
            bins=bins,
            dest_prefix_ratio=0.5,
        )
        next_mae = out["next_location_regression_probe"]["test"]["mae_m"]
        dest_mae = out["destination_regression_probe"]["test"]["mae_m"]
        self.assertLess(next_mae, 1.0)
        self.assertLess(dest_mae, 1.0)


if __name__ == "__main__":
    unittest.main()
