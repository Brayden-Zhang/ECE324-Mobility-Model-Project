import tempfile
import unittest
from pathlib import Path

from route_rangers.reporting import build_paper_metrics, write_paper_artifacts


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class PaperArtifactsTests(unittest.TestCase):
    def test_build_paper_metrics_exposes_core_sections(self) -> None:
        metrics = build_paper_metrics()

        self.assertIn("length", metrics)
        self.assertIn("macro", metrics)
        self.assertIn("unitraj_clean", metrics)
        self.assertIn("compute_resources", metrics)
        self.assertGreater(len(metrics["compute_resources"]), 0)
        self.assertLess(metrics["length"]["gap_dest_top1"], 0.0)
        self.assertEqual(
            metrics["compute_resources"][0]["stage"],
            "HMT full pretraining",
        )

    def test_write_paper_artifacts_emits_expected_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            write_paper_artifacts(output_dir)

            expected = {
                "paper_metrics.json",
                "paper_length_table.tex",
                "paper_robustness_table.tex",
                "paper_unitraj_main.tex",
                "paper_unitraj_clean.tex",
                "paper_macro_table.tex",
                "paper_compute_table.tex",
            }
            self.assertEqual({path.name for path in output_dir.iterdir()}, expected)


if __name__ == "__main__":
    unittest.main()
