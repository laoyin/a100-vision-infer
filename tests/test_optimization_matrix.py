import importlib.util
from pathlib import Path
import unittest
from unittest.mock import Mock, patch
import tempfile
import subprocess

spec = importlib.util.spec_from_file_location("matrix", Path(__file__).resolve().parents[1] / "tools/optimization_matrix.py")
matrix = importlib.util.module_from_spec(spec)
spec.loader.exec_module(matrix)

class MatrixTests(unittest.TestCase):
    def test_failed_results_cannot_win(self):
        rows = [dict(name="failed", status="failed", successful=3, output_tokens_per_second=999),
                dict(name="empty", status="passed", successful=0, output_tokens_per_second=999),
                dict(name="slow", status="passed", successful=3, output_tokens_per_second=5),
                dict(name="fast", status="passed", successful=3, output_tokens_per_second=10)]
        self.assertEqual(matrix.ranking(rows), ["fast", "slow"])

    def test_timeout_reaps_ranks_after_leader_exits(self):
        process = Mock(pid=1234)
        process.wait.side_effect = [subprocess.TimeoutExpired("worker", 1), 0, 0]
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(matrix.signal, "SIGKILL", 9, create=True), patch.object(matrix.subprocess, "Popen", return_value=process), patch.object(matrix.os, "killpg", create=True) as kill:
                with self.assertRaises(subprocess.TimeoutExpired):
                    matrix.execute(["worker"], Path(directory) / "run.log", 1)
                self.assertEqual(kill.call_count, 2)
                self.assertEqual(process.wait.call_count, 3)

    def test_matrix_has_isolated_and_combined_paths(self):
        profiles = matrix.profiles()
        self.assertEqual(len({p[0] for p in profiles}), len(profiles))
        self.assertEqual(profiles[0][0], "baseline")
        self.assertTrue(any(p[1] == ["--extra-fusions"] for p in profiles))
        self.assertTrue(any(p[1] == ["--cublas-prefill"] for p in profiles))
        self.assertTrue(any(p[2] == "graph" for p in profiles))
        self.assertTrue(any(p[3] == 2 for p in profiles))

if __name__ == "__main__":
    unittest.main()
