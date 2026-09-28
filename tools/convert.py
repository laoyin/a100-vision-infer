"""Convert a merged BF16 checkpoint to rank-specific AVI files. CPU/offline only."""
import argparse
import json
from pathlib import Path
import numpy as np
from export_vocabulary import export_vocabulary
from format_utils import bf16_bytes, partition, quantize_fp8, expected_shapes


def validate(config, tp):
    text, vision = config['text_config'], config['vision_config']
    if config.get('model_type') != 'qwen3_5' or config.get('quantization_config') or text.get('quantization_config'):
        raise ValueError('Expected unquantized merged qwen3_5 checkpoint; existing FP8 formats cannot be imported')
    if config.get('tie_word_embeddings') or text.get('attention_bias'):
        raise ValueError('Tied embeddings and attention bias are not supported')
    if text.get('hidden_act') != 'silu' or vision.get('hidden_act') != 'gelu_pytorch_tanh':
        raise ValueError('Unsupported activation')
    rope = text['rope_parameters']
    if rope.get('rope_type') != 'default' or not rope.get('mrope_interleaved', False):
        raise ValueError('Only default interleaved mRoPE is implemented')
    if vision.get('deepstack_visual_indexes') or vision['hidden_size'] % (4 * vision['num_heads']):
        raise ValueError('Unsupported vision geometry')
    if len(text['layer_types']) != text['num_hidden_layers'] or set(text['layer_types']) - {'full_attention', 'linear_attention'}:
        raise ValueError('Unsupported layer layout')
    for field in ['num_attention_heads', 'num_key_value_heads', 'linear_num_key_heads', 'linear_num_value_heads', 'intermediate_size']:
        if text[field] % tp:
            raise ValueError(f'{field} not divisible by TP={tp}')
    if text['num_attention_heads'] % text['num_key_value_heads'] or text['linear_num_value_heads'] % text['linear_num_key_heads']:
        raise ValueError('Invalid grouped head ratio')
    if text['linear_value_head_dim'] > 1024:
        raise ValueError('GDN value head exceeds kernel limit')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--tp', type=int, choices=[1, 2, 4], default=2)
    parser.add_argument('--precision', choices=['bf16', 'fp8'], default='fp8')
    args = parser.parse_args()
    if (args.model / 'adapter_config.json').exists():
        parser.error('Adapter-only input: merge it with the exact base checkpoint first')
    config = json.loads((args.model / 'config.json').read_text())
    validate(config, args.tp)
    from safetensors import safe_open
    index = args.model / 'model.safetensors.index.json'
    if index.exists():
        shard_names = sorted(set(json.loads(index.read_text())['weight_map'].values()))
        shards = [args.model / name for name in shard_names]
    else:
        shards = [args.model / 'model.safetensors']
    for shard in shards:
        if not shard.resolve().is_relative_to(args.model.resolve()) or not shard.is_file():
            raise ValueError(f'Invalid/missing shard: {shard}')
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / 'INCOMPLETE').write_text('Do not use until manifest.json exists and this marker is removed.\n')
    manifest = {'format': 'avi-v1', 'tp': args.tp, 'precision': args.precision, 'config': config,
                'quantization': 'E4M3FN per-row FP32 multiplier; text matrices only; visual/norm/gates BF16 or FP32',
                'ranks': [{'tensors': {}} for _ in range(args.tp)], 'skipped': []}
    for rank in range(args.tp):
        (args.out / f'rank{rank}').mkdir()
    seen = set()
    expected = expected_shapes(config)
    for shard in shards:
        with safe_open(shard, framework='pt', device='cpu') as source:
            for name in source.keys():
                if name.startswith('mtp.') or name.startswith('model.mtp.'):
                    manifest['skipped'].append(name)
                    continue
                if not name.startswith(('model.language_model.', 'model.visual.')) and name != 'lm_head.weight':
                    raise ValueError(f'Unknown checkpoint tensor: {name}')
                if name in seen:
                    raise ValueError(f'Duplicate tensor: {name}')
                seen.add(name)
                tensor = source.get_tensor(name)
                if name not in expected or tuple(tensor.shape) != expected[name]:
                    raise ValueError(f"Unsupported tensor or shape: {name}: {tuple(tensor.shape)}")
                # Float32 materialization is bounded to one source tensor.
                values = tensor.float().numpy()
                if not np.isfinite(values).all():
                    raise ValueError(f'Nonfinite tensor: {name}')
                for rank in range(args.tp):
                    piece = np.ascontiguousarray(partition(name, values, config['text_config'], rank, args.tp))
                    filename = f'rank{rank}/{name}.bin'
                    desc = {'file': filename, 'shape': list(piece.shape) or [1]}
                    quant = args.precision == 'fp8' and piece.ndim == 2 and name.endswith('.weight') and not name.startswith('model.visual.')
                    if quant:
                        codes, scales = quantize_fp8(piece)
                        (args.out / filename).write_bytes(codes.tobytes())
                        scale_file = filename + '.scale'
                        (args.out / scale_file).write_bytes(scales.tobytes())
                        desc.update(dtype='U8', scale={'file': scale_file, 'dtype': 'F32', 'shape': [len(scales)]})
                    else:
                        keep_fp32 = name.endswith(('.A_log', '.dt_bias'))
                        (args.out / filename).write_bytes(piece.astype('<f4').tobytes() if keep_fp32 else bf16_bytes(piece))
                        desc['dtype'] = 'F32' if keep_fp32 else 'BF16'
                    manifest['ranks'][rank]['tensors'][name] = desc
                print(f'Converted {name}', flush=True)
                del tensor, values
    required = {'lm_head.weight', 'model.language_model.embed_tokens.weight', 'model.language_model.norm.weight',
                'model.visual.patch_embed.proj.weight', 'model.visual.pos_embed.weight'}
    required = set(expected)
    if not required <= seen:
        raise ValueError(f'Missing required weights: {required-seen}')
    manifest['json_grammar_supported'] = export_vocabulary(args.model, args.out, config['text_config']['vocab_size'])
    (args.out / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    (args.out / 'INCOMPLETE').unlink()
    print('Converted. This does not certify model accuracy; run BF16 reference comparison first.')


if __name__ == '__main__':
    main()