import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np


class NativeComparisonTests(unittest.TestCase):
    def test_failure_is_structured_and_keeps_later_checks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ('baseline', 'candidate'):
                path = root / (name + '.json')
                path.write_text(json.dumps({'generated_ids': [1, 2, 3]}))
                for suffix in ('.prefill_logits.f32', '.decode_1.f32', '.decode_2.f32'):
                    values = [-1, -2] if name == 'candidate' and suffix == '.decode_1.f32' else [1, 2]
                    np.array(values, '<f4').tofile(str(path) + suffix)
            script = Path(__file__).resolve().parents[1] / 'tools/compare_native.py'
            result = subprocess.run([sys.executable, str(script), '--baseline', str(root/'baseline.json'), '--candidate', str(root/'candidate.json')], capture_output=True, text=True)
            self.assertEqual(result.returncode, 1)
            report = json.loads(result.stdout)
            self.assertFalse(report['passed'])
            self.assertEqual(report['failed_checks'], ['.decode_1.f32'])
            self.assertIn('.decode_2.f32', report['checks'])
