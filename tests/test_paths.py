"""
Unit tests for Route2Read path resolution.
"""

import unittest
from pathlib import Path
from src.utils.paths import get_project_root, DATA_DIR, RESULTS_DIR


class TestPaths(unittest.TestCase):
    def test_root_resolution(self):
        root = get_project_root()
        self.assertTrue(root.exists())
        self.assertTrue((root / "README.md").exists())

    def test_subdirs(self):
        self.assertEqual(DATA_DIR.name, "data")
        self.assertEqual(RESULTS_DIR.name, "results")


if __name__ == "__main__":
    unittest.main()
