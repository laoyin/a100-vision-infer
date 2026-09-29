"""Generate a small random HF checkpoint + prepared image request for GPU smoke tests."""
import argparse
import json
from pathlib import Path
import numpy as np
from format_utils import image_geometry, rope_positions

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--out',type=Path,required=True); p.add_argument('--block-fp8',action='store_true'); p.add_argument('--mtp',action='store_true'); a=p.parse_args()
    if a.mtp and not a.block_fp8:p.error('--mtp requires --block-fp8')
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
    if a.block_fp8:
        cfg.text_config.hidden_size=256;cfg.text_config.intermediate_size=512
        cfg.text_config.head_dim=128;cfg.text_config.linear_key_head_dim=128;cfg.text_config.linear_value_head_dim=128
        cfg.vision_config.out_hidden_size=256
    a.out.mkdir(parents=True,exist_ok=False)
    if a.mtp:
        cfg.text_config.mtp_num_hidden_layers=1
        cfg.text_config.mtp_use_dedicated_embeddings=False
    model=Qwen3_5ForConditionalGeneration(cfg).to(torch.bfloat16).eval()
    model.save_pretrained(a.out/'model',safe_serialization=True)
    if a.block_fp8:
        from safetensors.torch import save_file
        state=model.state_dict();converted={};excluded=[]
        if a.mtp:
            h=cfg.text_config.hidden_size
            state['mtp.fc.weight']=(torch.randn(h,2*h)*.02).to(torch.bfloat16)
            for name in ('norm','pre_fc_norm_embedding','pre_fc_norm_hidden'):
                state[f'mtp.{name}.weight']=torch.zeros(h,dtype=torch.bfloat16)
            for name,tensor in list(state.items()):
                if name.startswith('model.language_model.layers.1.'):
                    state[name.replace('model.language_model.layers.1.','mtp.layers.0.')]=tensor.clone()
        for name,tensor in state.items():
            quantize=name.startswith(('model.language_model.layers.','mtp.layers.')) and tensor.ndim==2 and name.endswith('.weight') and not name.endswith(('.in_proj_a.weight','.in_proj_b.weight'))
            if not quantize:
                converted[name]=tensor.contiguous()
                if name.endswith('.weight'):excluded.append(name[:-7])
                continue
            n,k=tensor.shape;scales=torch.empty(((n+127)//128,(k+127)//128),dtype=torch.float32);codes=torch.empty_like(tensor,dtype=torch.float8_e4m3fn)
            for row in range(0,n,128):
                for col in range(0,k,128):
                    block=tensor[row:row+128,col:col+128].float();scale=block.abs().max().clamp_min(1e-12)/448
                    scales[row//128,col//128]=scale;codes[row:row+128,col:col+128]=(block/scale).to(torch.float8_e4m3fn)
            converted[name]=codes;converted[name[:-7]+'.weight_scale_inv']=scales
        save_file(converted,a.out/'model/model.safetensors',metadata={'format':'pt'})
        cp=a.out/'model/config.json';config=json.loads(cp.read_text());config['quantization_config']={'quant_method':'fp8','fmt':'e4m3','weight_block_size':[128,128],'activation_scheme':'dynamic','modules_to_not_convert':excluded};cp.write_text(json.dumps(config,indent=2))
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
    if a.mtp:
        req['max_new_tokens']=12
        (request_dir/'request.json').write_text(json.dumps(req,indent=2))
    print(a.out)

if __name__=='__main__': main()
