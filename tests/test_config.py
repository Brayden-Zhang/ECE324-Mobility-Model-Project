"""Basic checks for the repository path configuration."""

from pathlib import Path
import sys
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from route_rangers import config


class ConfigTests(unittest.TestCase):
    def test_project_root_points_to_repository_root(self) -> None:
        self.assertEqual(config.PROJECT_ROOT, PROJECT_ROOT)

    def test_standard_directories_exist_or_can_be_created(self) -> None:
        created_directories = config.ensure_project_directories()
        self.assertIn(config.RAW_DATA_DIR, created_directories)
        self.assertIn(config.PROCESSED_DATA_DIR, created_directories)
        self.assertTrue(config.CHECKPOINTS_DIR.exists())
        self.assertTrue(config.PAPER_FIGURES_DIR.exists())


if __name__ == "__main__":
    unittest.main()
