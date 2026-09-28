import json
from pathlib import Path
import tempfile
import unittest
from tools.inspect_checkpoint import inspect_checkpoint


class InspectionTests(unittest.TestCase):
    def test_missing_quantization_is_not_compatible_claim(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "config.json").write_text('{}', encoding="utf-8")
            result = inspect_checkpoint(path)
            self.assertFalse(result["runtime_ready"])
            self.assertIsNone(result["quantization_config"])
            self.assertTrue(any("FP8 format unknown" in w for w in result["warnings"]))

    def test_adapter_and_partition_metadata_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "config.json").write_text(json.dumps({"text_config": {
                "num_attention_heads": 24, "num_key_value_heads": 3}}), encoding="utf-8")
            adapter = {"r": 128, "lora_alpha": 128, "modules_to_save": ["visual.merger"]}
            (path / "adapter_config.json").write_text(json.dumps(adapter), encoding="utf-8")
            result = inspect_checkpoint(path, path)
            self.assertEqual(result["adapter_config"], adapter)
            self.assertTrue(result["partition_checks"]["num_attention_heads"]["divisible"])
            self.assertFalse(result["partition_checks"]["num_key_value_heads"]["divisible"])

    def test_invalid_config_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "config.json").write_text('[]', encoding="utf-8")
            with self.assertRaises(ValueError):
                inspect_checkpoint(path)


if __name__ == "__main__":
    unittest.main()
