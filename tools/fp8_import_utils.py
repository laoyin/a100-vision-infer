"""Preserve E4M3FN codes and expand only the block-scale row axis."""
import numpy as np
from format_utils import partition

def validate_quantization(config):
 q=config.get('quantization_config') or config.get('text_config',{}).get('quantization_config')
 if not isinstance(q,dict) or q.get('quant_method')!='fp8' or q.get('fmt','e4m3')!='e4m3' or q.get('weight_block_size')!=[128,128] or q.get('activation_scheme')!='dynamic':
  raise ValueError('Expected FP8 E4M3 dynamic activation with weight_block_size [128,128]')
 return q

def partition_fp8(name,codes,scales,text,rank,tp):
 codes=np.asarray(codes);scales=np.asarray(scales,dtype=np.float32)
 if codes.dtype!=np.uint8 or codes.ndim!=2:raise ValueError('Expected 2D raw FP8 bytes')
 n,k=codes.shape
 if scales.shape!=((n+127)//128,(k+127)//128):raise ValueError(f'Invalid block scale shape for {name}: {scales.shape}')
 if not np.isfinite(scales).all() or (scales<=0).any():raise ValueError('Block scales must be finite and positive')
 if ((codes&127)==127).any():raise ValueError('Nonfinite E4M3FN weight')
 row_parallel=name.startswith('model.language_model.layers.') and name.endswith(('.mlp.down_proj.weight','.self_attn.o_proj.weight','.linear_attn.out_proj.weight'))
 if row_parallel and tp>1 and k%(128*tp):raise ValueError('TP column split must align with 128-element blocks')
 expanded=np.repeat(scales,128,axis=0)[:n]
 output=partition(name,codes,text,rank,tp)
 scale_output=partition(name,expanded,text,rank,tp)
 if scale_output.shape!=(output.shape[0],(output.shape[1]+127)//128):raise ValueError('Partitioned block scale mismatch')
 return np.ascontiguousarray(output),np.ascontiguousarray(scale_output,dtype='<f4')
