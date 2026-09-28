import importlib.util
from pathlib import Path
import tempfile
import unittest
import json
import numpy as np

spec = importlib.util.spec_from_file_location('layers', Path(__file__).resolve().parents[1] / 'tools/compare_layers.py')
layers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(layers)


class LayerComparisonTests(unittest.TestCase):
    def test_zero_and_nonfinite(self):
        self.assertEqual(layers.metrics([0, 0], [0, 0])['cosine'], 1)
        self.assertEqual(layers.metrics([0, 0], [1, 0])['cosine'], 0)
        with self.assertRaises(ValueError):
            layers.metrics([1], [np.nan])

    def test_sort_and_skip_incomparable_decode(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            b, c = root / 'baseline.json', root / 'candidate.json'
            b.write_text(json.dumps({'generated_ids': [1]*100}))
            c.write_text(json.dumps({'generated_ids': [2]*100}))
            for layer in (12, 2):
                for path in (b, c):
                    np.array([1, 2, 3], '<f4').tofile(str(path)+f'.prefill_layers.rank0.layer{layer}.hidden.f32')
            report = layers.compare(b, c)
            self.assertEqual([r['layer'] for r in report['records']], [2, 12])
            self.assertTrue(report['skipped'])
            Path(str(c)+'.prefill_layers.rank0.layer2.hidden.f32').unlink()
            with self.assertRaises(ValueError):
                layers.compare(b, c)
