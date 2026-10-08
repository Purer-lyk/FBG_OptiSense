"""Run PyTorch CNN tests before Qt DLLs load by using an isolated process."""

import subprocess
import sys
import unittest
from pathlib import Path


class SparseTouchCnnProcessTests(unittest.TestCase):
    def test_sparse_cnn_core_in_clean_process(self):
        result = subprocess.run(
            [sys.executable, "manual_touch_cnn_subprocess_tests.py"],
            cwd=Path(__file__).resolve().parent,
            text=True,
            capture_output=True,
            timeout=120,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("OK", result.stderr)


if __name__ == "__main__":
    unittest.main()
