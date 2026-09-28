"""Generate a small random HF checkpoint + prepared image request for GPU smoke tests."""
import argparse
import json
from pathlib import Path
import numpy as np
from format_utils import image_geometry, rope_positions

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--out',type=Path,required=True); a=p.parse_args()
    import torch
    from transformers import Qwen3_5Config,Qwen3_5ForConditionalGeneration
    torch.manual_seed(42)
    cfg=Qwen3_5Config(text_config={
        'vocab_size':256,'hidden_size':64,'intermediate_size':128,'num_hidden_layers':2,
        'layer_types':['linear_attention','full_attention'],'num_attention_heads':4,'num_key_value_heads':2,
        'head_dim':16,'linear_num_key_heads':2,'linear_num_value_heads':4,'linear_key_head_dim':16,
        'linear_value_head_dim':16,'linear_conv_kernel_dim':4,'hidden_act':'silu','attention_bias':False,
        'rms_norm_eps':1e-6,'tie_word_embeddings':False,'eos_token_id':2,
        'rope_parameters':{'rope_type':'default','rope_theta':10000000.,'partial_rotary_factor':0.5,
                           'mrope_interleaved':True,'mrope_section':[2,1,1]}},
        vision_config={'depth':2,'hidden_size':64,'intermediate_size':96,'num_heads':4,
                       'in_channels':3,'patch_size':16,'temporal_patch_size':2,'spatial_merge_size':2,
                       'out_hidden_size':64,'num_position_embeddings':16,'hidden_act':'gelu_pytorch_tanh',
                       'deepstack_visual_indexes':[]},
        image_token_id=250,video_token_id=249,vision_start_token_id=251,vision_end_token_id=252,
        tie_word_embeddings=False)
    a.out.mkdir(parents=True,exist_ok=False)
    model=Qwen3_5ForConditionalGeneration(cfg).to(torch.bfloat16).eval()
    model.save_pretrained(a.out/'model',safe_serialization=True)
    request_dir=a.out/'request'; request_dir.mkdir()
    def write(name,x,dtype):
        (request_dir/(name+'.bin')).write_bytes(x.tobytes())
        return {'file':name+'.bin','shape':list(x.shape),'dtype':dtype}
    ids=np.array([1,251,250,250,250,250,252,3,4],dtype='<i8')
    pos,nxt=rope_positions(ids,[[1,4,4]],250,2)
    coords,indices,factors=image_geometry(4,4,2,4)
    patches=np.random.default_rng(42).normal(size=(16,1536)).astype('<f4')
    req={'format':'avi-request-v1','max_context':32,'max_new_tokens':3,'next_position':nxt,
         'eos_token_ids':[2],'image_token_id':250,'input_ids':write('ids',ids,'I64'),'positions':write('positions',pos,'I64'),
         'images':[{'grid':[1,4,4],'patches':write('patches',patches,'F32'),
                    'coords':write('coords',coords,'I64'),'position_indices':write('indices',indices,'I64'),
                    'position_weights':write('weights',factors,'F32')}]}
    (request_dir/'request.json').write_text(json.dumps(req,indent=2))
    print(a.out)

if __name__=='__main__': main()