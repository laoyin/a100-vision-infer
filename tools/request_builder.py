"""Shared frontend: tokenizer and image preprocessing only, never model inference."""
import json
from pathlib import Path
import numpy as np
from format_utils import image_geometry, rope_positions


def build_request(processor, config, generation, messages, images, out, *, max_pixels=4000000,
                  max_context=20480, max_new_tokens=256, thinking=False,content_format='hf',bf16_patches=False):
    if not 0 < max_new_tokens < max_context or max_pixels <= 0:
        raise ValueError('Invalid request budget')
    if content_format=='vllm-string':
        from chat_format import vllm_string_messages
        marker=''.join(processor.tokenizer.convert_ids_to_tokens(config[key])
                       for key in ('vision_start_token_id','image_token_id','vision_end_token_id'))
        messages=vllm_string_messages(messages,marker)
    elif content_format!='hf':
        raise ValueError('Unknown content format')
    text=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=thinking)
    size=processor.image_processor.size
    minimum=size['shortest_edge'] if isinstance(size,dict) else size.shortest_edge
    if max_pixels<minimum: raise ValueError('max_pixels is below processor minimum')
    kwargs={'images':images,'images_kwargs':{'min_pixels':minimum,'max_pixels':max_pixels}} if images else {}
    encoded=processor(text=[text],return_tensors='pt',**kwargs)
    ids=encoded['input_ids'][0].tolist()
    if len(ids)+max_new_tokens>max_context: raise ValueError(f'Input {len(ids)} + output {max_new_tokens} exceeds {max_context}')
    grids=encoded['image_grid_thw'].tolist() if images else []
    vc=config['vision_config'];patch=vc['patch_size'];merge=vc['spatial_merge_size']
    if any(t!=1 or h*w*patch*patch>max_pixels for t,h,w in grids):raise ValueError('Unsupported grid or pixel limit was not honored')
    positions,next_position=rope_positions(ids,grids,config['image_token_id'],merge)
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    def write(name,array,dtype):
        array=np.ascontiguousarray(array);(out/(name+'.bin')).write_bytes(array.tobytes())
        return {'file':name+'.bin','shape':list(array.shape),'dtype':dtype}
    eos=generation.get('eos_token_id',config['text_config']['eos_token_id']);eos=eos if isinstance(eos,list) else [eos]
    req={'format':'avi-request-v1','max_context':max_context,'max_new_tokens':max_new_tokens,'next_position':next_position,
         'eos_token_ids':eos,'image_token_id':config['image_token_id'],'input_ids':write('input_ids',np.array(ids,dtype='<i8'),'I64'),
         'positions':write('positions',positions,'I64'),'images':[],'thinking':thinking,'max_pixels':max_pixels}
    offset=0;side=int(vc['num_position_embeddings']**0.5)
    for i,(t,h,w) in enumerate(grids):
        pixels=encoded['pixel_values'][offset:offset+h*w];offset+=h*w
        if bf16_patches:
            import torch
            # Identical rounding to the worker's F32 -> BF16 conversion, half the
            # serialization/read/H2D bytes. Values are not requantized to FP8.
            patches=pixels.to(torch.bfloat16).contiguous().view(torch.int16).numpy().astype('<i2',copy=False)
        else:
            patches=pixels.float().numpy().astype('<f4')
        coords,indices,factors=image_geometry(h,w,merge,side)
        req['images'].append({'grid':[t,h,w],'patches':write(f'image{i}_patches',patches,'BF16' if bf16_patches else 'F32'),
             'coords':write(f'image{i}_coords',coords,'I64'),'position_indices':write(f'image{i}_indices',indices,'I64'),
             'position_weights':write(f'image{i}_factors',factors,'F32')})
    (out/'request.json').write_text(json.dumps(req,indent=2),encoding='utf-8')
    return req
