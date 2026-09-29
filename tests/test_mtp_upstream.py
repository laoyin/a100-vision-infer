import json
from pathlib import Path
import struct
import tempfile
import unittest
from tools.inspect_mtp import inspect, headers
from tools.benchmark_mtp_upstream import engine_options


class MTPTests(unittest.TestCase):
    def fixture(self, root, include=True):
        config = {'model_type': 'qwen3_5', 'text_config': {'hidden_size': 4, 'head_dim': 2,
                  'num_attention_heads': 2, 'num_key_value_heads': 1, 'intermediate_size': 8,
                  'mtp_num_hidden_layers': 1, 'mtp_use_dedicated_embeddings': False}}
        (root/'config.json').write_text(json.dumps(config))
        shapes = {'mtp.fc.weight': [4, 8], 'mtp.norm.weight': [4],
                  'mtp.pre_fc_norm_hidden.weight': [4], 'mtp.pre_fc_norm_embedding.weight': [4]}
        layer = {'input_layernorm': [4], 'post_attention_layernorm': [4], 'self_attn.q_norm': [2],
                 'self_attn.k_norm': [2], 'self_attn.q_proj': [8, 4], 'self_attn.k_proj': [2, 4],
                 'self_attn.v_proj': [2, 4], 'self_attn.o_proj': [4, 4],
                 'mlp.gate_proj': [8, 4], 'mlp.up_proj': [8, 4], 'mlp.down_proj': [4, 8]}
        shapes.update({'mtp.layers.0.'+k+'.weight': v for k, v in layer.items()})
        values, offset = {}, 0
        for key, shape in (shapes.items() if include else []):
            count = 2
            for dim in shape:
                count *= dim
            values[key] = {'shape': shape, 'dtype': 'BF16', 'data_offsets': [offset, offset+count]}
            offset += count
        header = json.dumps(values).encode()
        (root/'model.safetensors').write_bytes(struct.pack('<Q', len(header))+header+bytes(offset))

    def test_config_alone_is_not_mtp(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root, include=False)
            self.assertFalse(inspect(root)['eligible_for_trial'])

    def test_complete_shapes_and_corrupt_extent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root)
            self.assertTrue(inspect(root)['eligible_for_trial'])
            path = root/'model.safetensors'
            path.write_bytes(path.read_bytes()[:-1])
            with self.assertRaises(ValueError):
                headers(root)

    def test_native_artifact_is_not_hf_mtp(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'manifest.json').write_text(json.dumps({'skipped': ['mtp.fc.weight']}))
            report = inspect(root)
            self.assertFalse(report['eligible_for_trial'])
            self.assertEqual(report['skipped_mtp_tensors'], ['mtp.fc.weight'])

    def test_shard_cannot_escape(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'model.safetensors.index.json').write_text(json.dumps({'weight_map': {'x': '../outside.safetensors'}}))
            with self.assertRaises(ValueError):
                headers(root)

    def test_baseline_and_mtp_share_precision_and_preprocessing(self):
        base = engine_options('model', 0, 20480, .8)
        candidate = engine_options('model', 3, 20480, .8)
        self.assertEqual(candidate.pop('speculative_config'), {'method': 'mtp', 'num_speculative_tokens': 3})
        self.assertEqual(base, candidate)
        self.assertFalse(base['enable_prefix_caching'])
