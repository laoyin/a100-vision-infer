"""Offline tokenization/image preprocessing only; no model forward or image encoder."""
import argparse
import json
from pathlib import Path
import numpy as np
from format_utils import image_geometry, rope_positions


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, required=True, help='Merged HF checkpoint with original processor/tokenizer')
    p.add_argument('--image', type=Path, action='append', default=[])
    p.add_argument('--prompt', required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--max-pixels', type=int, default=4000000)
    p.add_argument('--max-context', type=int, default=20480)
    p.add_argument('--max-new-tokens', type=int, default=256)
    p.add_argument('--thinking', action='store_true', help='Default disables thinking for controlled baseline')
    a = p.parse_args()
    if a.max_pixels <= 0 or a.max_new_tokens <= 0:
        p.error('Budgets must be positive')
    from transformers import AutoProcessor
    from PIL import Image, ImageOps
    config = json.loads((a.model / 'config.json').read_text())
    processor = AutoProcessor.from_pretrained(a.model, local_files_only=True, trust_remote_code=False)
    images = []
    for path in a.image:
        with Image.open(path) as image:
            images.append(ImageOps.exif_transpose(image).convert('RGB'))
    content = [{'type': 'image'} for _ in images] + [{'type': 'text', 'text': a.prompt}]
    text = processor.apply_chat_template([{'role': 'user', 'content': content}], tokenize=False,
                                         add_generation_prompt=True, enable_thinking=a.thinking)
    # Explicit processor image kwargs preserve the official resize/patch rules.
    size = processor.image_processor.size
    minimum = size['shortest_edge'] if isinstance(size, dict) else size.shortest_edge
    if a.max_pixels < minimum:
        raise ValueError(f'max_pixels must be >= processor minimum {minimum}')
    kwargs = {'images': images, 'images_kwargs': {'min_pixels': minimum, 'max_pixels': a.max_pixels}} if images else {}
    inputs = processor(text=[text], return_tensors='pt', **kwargs)
    ids = inputs['input_ids'][0].tolist()
    if len(ids) + a.max_new_tokens > a.max_context:
        raise ValueError(f'Input {len(ids)} + output {a.max_new_tokens} exceeds {a.max_context}')
    grids = inputs['image_grid_thw'].tolist() if images else []
    patch_size = config['vision_config']['patch_size']
    if any(h * w * patch_size**2 > a.max_pixels for _, h, w in grids):
        raise ValueError('Processor did not honor max_pixels')
    merge = config['vision_config']['spatial_merge_size']
    positions, next_position = rope_positions(ids, grids, config['image_token_id'], merge)
    a.out.mkdir(parents=True, exist_ok=False)
    def write(name, array, dtype):
        array = np.ascontiguousarray(array)
        (a.out / (name + '.bin')).write_bytes(array.tobytes())
        return {'file': name + '.bin', 'shape': list(array.shape), 'dtype': dtype}
    generation_file = a.model / 'generation_config.json'
    generation = json.loads(generation_file.read_text()) if generation_file.exists() else {}
    eos = generation.get('eos_token_id', config['text_config']['eos_token_id'])
    eos = eos if isinstance(eos, list) else [eos]
    request = {'format': 'avi-request-v1', 'max_context': a.max_context, 'max_new_tokens': a.max_new_tokens,
               'next_position': next_position, 'eos_token_ids': eos, 'image_token_id': config['image_token_id'],
               'input_ids': write('input_ids', np.array(ids, dtype='<i8'), 'I64'),
               'positions': write('positions', positions, 'I64'), 'images': [],
               'prompt': a.prompt, 'thinking': a.thinking, 'max_pixels': a.max_pixels,
               'source_model': str(a.model.resolve())}
    offset = 0
    side = int(config['vision_config']['num_position_embeddings'] ** 0.5)
    for i, (t, h, w) in enumerate(grids):
        if t != 1:
            raise ValueError('Only still images are supported')
        patches = inputs['pixel_values'][offset:offset+h*w].float().numpy().astype('<f4')
        offset += h*w
        coords, indices, factors = image_geometry(h, w, merge, side)
        request['images'].append({'grid': [t, h, w], 'original_size': list(images[i].size),
                                 'patches': write(f'image{i}_patches', patches, 'F32'),
                                 'coords': write(f'image{i}_coords', coords, 'I64'),
                                 'position_indices': write(f'image{i}_indices', indices, 'I64'),
                                 'position_weights': write(f'image{i}_factors', factors, 'F32')})
    (a.out / 'request.json').write_text(json.dumps(request, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Prepared {len(ids)} tokens, {len(images)} images. Image encoding runs in C++, not here.')


if __name__ == '__main__':
    main()