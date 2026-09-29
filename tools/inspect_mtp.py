"""Read checkpoint headers to check for dense Qwen3.5 MTP weights; no Torch or GPU required."""
import argparse
import json
from pathlib import Path
import struct
import math


def headers(model):
    model = Path(model).resolve()
    index = model / 'model.safetensors.index.json'
    files = sorted(set(json.loads(index.read_text(encoding='utf-8'))['weight_map'].values())) if index.exists() else ['model.safetensors']
    result = {}
    for relative in files:
        path = (model / relative).resolve()
        if not path.is_relative_to(model):
            raise ValueError('Shard escapes model directory')
        size = path.stat().st_size
        with path.open('rb') as stream:
            prefix = stream.read(8)
            if len(prefix) != 8:
                raise ValueError('Truncated safetensors header: ' + relative)
            length = struct.unpack('<Q', prefix)[0]
            if not 2 <= length <= min(size-8, 32 << 20):
                raise ValueError('Invalid safetensors header length: ' + relative)
            header = json.loads(stream.read(length))
        for name, entry in header.items():
            if name == '__metadata__':
                continue
            if name in result:
                raise ValueError('Duplicate tensor: ' + name)
            start, end = entry['data_offsets']
            if not 0 <= start <= end <= size-8-length:
                raise ValueError('Invalid tensor extent: ' + name)
            shape = entry['shape']
            if not isinstance(shape, list) or any(type(d) is not int or d < 0 for d in shape):
                raise ValueError('Invalid shape: ' + name)
            widths = {'BF16': 2, 'F16': 2, 'F32': 4, 'F64': 8, 'F8_E4M3': 1, 'F8_E5M2': 1,
                      'I64': 8, 'I32': 4, 'I16': 2, 'I8': 1, 'U8': 1, 'BOOL': 1}
            width = widths.get(entry['dtype'])
            if width is not None and math.prod(shape)*width != end-start:
                raise ValueError('Shape/dtype byte count mismatch: ' + name)
            result[name] = {'shape': entry['shape'], 'dtype': entry['dtype'], 'bytes': end-start}
    return result


def inspect(model):
    model = Path(model)
    if (model/'manifest.json').exists():
        manifest = json.loads((model/'manifest.json').read_text(encoding='utf-8'))
        return {'eligible_for_trial': False, 'native_artifact': True,
                'skipped_mtp_tensors': [n for n in manifest.get('skipped', []) if n.startswith(('mtp.', 'model.mtp.'))],
                'reason': 'Native AVI import omits MTP. Use the original merged HF FP8 checkpoint for upstream runtime trials.'}
    config = json.loads((model/'config.json').read_text(encoding='utf-8'))
    text = config.get('text_config', config)
    count = text.get('mtp_num_hidden_layers', 0)
    tensors = headers(model)
    mtp = {}
    for name, entry in tensors.items():
        if name.startswith(('mtp.', 'model.mtp.')):
            key = name.removeprefix('model.')
            if key in mtp:
                raise ValueError('Ambiguous MTP tensor aliases: ' + key)
            mtp[key] = entry
    dense = config.get('model_type') == 'qwen3_5' and not text.get('num_experts', 0)
    required = {}
    if dense and isinstance(count, int) and 0 < count <= 8:
        h, d = text['hidden_size'], text['head_dim']
        q, k, inter = text['num_attention_heads']*d, text['num_key_value_heads']*d, text['intermediate_size']
        required = {'mtp.fc.weight': [h, 2*h]}
        for norm in ('norm', 'pre_fc_norm_embedding', 'pre_fc_norm_hidden'):
            required[f'mtp.{norm}.weight'] = [h]
        for layer in range(count):
            shapes = {'input_layernorm': [h], 'post_attention_layernorm': [h],
                      'self_attn.q_norm': [d], 'self_attn.k_norm': [d],
                      'self_attn.q_proj': [2*q, h], 'self_attn.k_proj': [k, h],
                      'self_attn.v_proj': [k, h], 'self_attn.o_proj': [h, q],
                      'mlp.gate_proj': [inter, h], 'mlp.up_proj': [inter, h], 'mlp.down_proj': [h, inter]}
            required.update({f'mtp.layers.{layer}.{name}.weight': shape for name, shape in shapes.items()})
    missing = sorted(set(required)-set(mtp))
    wrong = {n: {'expected': shape, 'actual': mtp[n]['shape']} for n, shape in required.items() if n in mtp and mtp[n]['shape'] != shape}
    format_errors = []
    for name in required.keys() & mtp.keys():
        dtype = mtp[name]['dtype']
        if dtype not in ('BF16', 'F16', 'F32', 'F8_E4M3'):
            format_errors.append('Unsupported MTP dtype: ' + name + ': ' + dtype)
        if dtype == 'F8_E4M3':
            shape = mtp[name]['shape']
            scale = mtp.get(name.removesuffix('.weight')+'.weight_scale_inv')
            if len(shape) != 2 or scale is None or scale['shape'] != [(d+127)//128 for d in shape]:
                format_errors.append('Missing/invalid block128 scale: ' + name)
    shared = not text.get('mtp_use_dedicated_embeddings', False)
    return {'eligible_for_trial': bool(required) and shared and not missing and not wrong and not format_errors,
            'native_artifact': False, 'model_type': config.get('model_type'), 'mtp_num_hidden_layers': count,
            'shared_embeddings': shared, 'mtp_tensor_count': len(mtp), 'mtp_bytes': sum(e['bytes'] for e in mtp.values()),
            'missing': missing, 'wrong_shapes': wrong, 'format_errors': format_errors, 'tensors': mtp,
            'quantization': config.get('quantization_config', text.get('quantization_config')),
            'note': 'Header/shape eligibility only, not runtime compatibility or tensor numerical validation. Fine-tuning may reduce draft acceptance. Dedicated MTP embeddings need separate review.'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True, type=Path)
    p.add_argument('--require-mtp', action='store_true')
    a = p.parse_args()
    report = inspect(a.model)
    print(json.dumps(report, indent=2))
    if a.require_mtp and not report['eligible_for_trial']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
