"""Import existing HF block-FP8 weights without dequantizing/requantizing the checkpoint."""
import argparse,copy,json
from pathlib import Path
import numpy as np
from fp8_import_utils import validate_quantization,partition_fp8
from format_utils import expected_shapes,partition,bf16_bytes,mtp_shapes
from export_vocabulary import export_vocabulary
from convert import validate

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--tp',type=int,choices=[1,2,4],default=2);p.add_argument('--include-mtp',action='store_true');a=p.parse_args()
 import torch
 from safetensors import safe_open
 config=json.loads((a.model/'config.json').read_text(encoding='utf-8'));quant=validate_quantization(config)
 native=copy.deepcopy(config);native.pop('quantization_config',None);native['text_config'].pop('quantization_config',None);validate(native,a.tp)
 index=a.model/'model.safetensors.index.json'
 shards=sorted(set(json.loads(index.read_text())['weight_map'].values())) if index.exists() else ['model.safetensors']
 locations={}
 if a.include_mtp:
  from inspect_mtp import inspect
  audit=inspect(a.model)
  if not audit['eligible_for_trial']:raise ValueError('MTP audit failed: '+json.dumps(audit.get('missing',audit)))
  if (a.model/'mtp.safetensors').is_file() and 'mtp.safetensors' not in shards:shards.append('mtp.safetensors')
 for relative in shards:
  path=(a.model/relative).resolve()
  if not path.is_relative_to(a.model.resolve()) or not path.is_file():raise ValueError('Invalid shard path')
  with safe_open(path,framework='pt',device='cpu') as f:
   for key in f.keys():
    if key in locations:raise ValueError(f'Duplicate tensor {key}')
    locations[key]=path
 aliases={}
 if a.include_mtp:
  for key in list(locations):
   if key.startswith('model.mtp.'):
    target=key.removeprefix('model.')
    if target in locations:raise ValueError('Duplicate MTP alias '+target)
    locations[target]=locations.pop(key);aliases[target]=key
 expected=expected_shapes(native)
 if a.include_mtp:expected.update(mtp_shapes(native))
 skipped=[] if a.include_mtp else [k for k in locations if k.startswith(('mtp.','model.mtp.'))]
 scale_names={name[:-7]+'.weight_scale_inv' for name in expected if name.endswith('.weight')}
 unknown=set(locations)-set(expected)-scale_names-set(skipped)
 if unknown:raise ValueError(f'Unsupported checkpoint tensor names: {sorted(unknown)[:20]}')
 if set(expected)-set(locations):raise ValueError(f'Missing weights: {sorted(set(expected)-set(locations))[:20]}')
 def read(name):
  if name not in locations:raise ValueError(f'Missing scale tensor {name}; expected HF weight_scale_inv layout')
  with safe_open(locations[name],framework='pt',device='cpu') as f:return f.get_tensor(aliases.get(name,name))
 a.out.mkdir(parents=True,exist_ok=False);(a.out/'INCOMPLETE').write_text('Import in progress')
 manifest={'format':'avi-v1','tp':a.tp,'precision':'fp8','config':native,'source_quantization_config':quant,'quantization':'E4M3FN original codes, row-expanded block128 multipliers; BF16 activation compute','ranks':[{'tensors':{}} for _ in range(a.tp)],'skipped':skipped}
 used=set();quantized=0
 manifest['native_mtp']=a.include_mtp
 for rank in range(a.tp):(a.out/f'rank{rank}').mkdir()
 for name,shape in expected.items():
  tensor=read(name)
  if tuple(tensor.shape)!=shape:raise ValueError(f'Unexpected shape {name}: {tensor.shape}')
  scale_name=name[:-7]+'.weight_scale_inv' if name.endswith('.weight') else None
  is_fp8=tensor.dtype==torch.float8_e4m3fn
  if is_fp8:
   if tensor.ndim!=2 or not name.endswith('.weight') or name.startswith('model.visual.'):raise ValueError(f'Unsupported FP8 placement: {name}')
   codes=tensor.contiguous().view(torch.uint8).numpy();scales=read(scale_name).float().numpy();used.add(scale_name);quantized+=1
  else:
   if 'float8' in str(tensor.dtype):raise ValueError(f'Unsupported FP8 dtype: {tensor.dtype}')
   if scale_name in locations:raise ValueError(f'Non-FP8 weight has a scale: {name}')
   if tensor.dtype not in (torch.bfloat16,torch.float16,torch.float32):raise ValueError(f'Unsupported dtype: {name}: {tensor.dtype}')
   values=tensor.float().numpy()
   if not np.isfinite(values).all():raise ValueError(f'Nonfinite weight: {name}')
  for rank in range(a.tp):
   filename=f'rank{rank}/{name}.bin'
   if is_fp8:
    piece,scale=partition_fp8(name,codes,scales,native['text_config'],rank,a.tp)
    (a.out/filename).write_bytes(piece.tobytes());(a.out/(filename+'.scale')).write_bytes(scale.tobytes())
    desc={'file':filename,'shape':list(piece.shape),'dtype':'U8','scale':{'file':filename+'.scale','dtype':'F32','shape':list(scale.shape)}}
   else:
    piece=np.ascontiguousarray(partition(name,values,native['text_config'],rank,a.tp));fp32=name.endswith(('.A_log','.dt_bias'))
    (a.out/filename).write_bytes(piece.astype('<f4').tobytes() if fp32 else bf16_bytes(piece))
    desc={'file':filename,'shape':list(piece.shape) or [1],'dtype':'F32' if fp32 else 'BF16'}
   manifest['ranks'][rank]['tensors'][name]=desc
  print('Imported',name,flush=True)
 if not quantized:raise ValueError('No actual FP8 tensors found')
 if (set(locations)&scale_names)-used:raise ValueError('Unconsumed scale tensors')
 manifest['quantized_tensor_count']=quantized
 manifest['json_grammar_supported']=export_vocabulary(a.model,a.out,native['text_config']['vocab_size'])
 (a.out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8');(a.out/'INCOMPLETE').unlink()
 print('Imported original FP8 weights. A100 execution uses BF16 activations, not the source dynamic W8A8 arithmetic.')
if __name__=='__main__':main()
