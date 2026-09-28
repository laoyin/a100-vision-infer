"""Validation only: compare native prefill trace to a Transformers BF16 reference."""
import argparse
import json
from pathlib import Path
import numpy as np

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--request',type=Path,required=True)
    p.add_argument('--native-output',type=Path,required=True)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--decode-check',type=int,default=8,help='Check up to N native continuation logits with teacher-forced full reference forwards')
    p.add_argument('--min-cosine',type=float,default=0.99)
    p.add_argument('--max-relative-rmse',type=float,default=0.15)
    a=p.parse_args()
    import torch
    from transformers import Qwen3_5ForConditionalGeneration
    req=json.loads((a.request/'request.json').read_text())
    def read(d):
        dt={'F32':'<f4','I64':'<i8'}[d['dtype']]
        array=np.fromfile(a.request/d['file'],dtype=dt).reshape(d['shape'])
        return torch.from_numpy(array.copy()).to(a.device)
    model=Qwen3_5ForConditionalGeneration.from_pretrained(a.model,dtype=torch.bfloat16,
                device_map=a.device,attn_implementation='sdpa',local_files_only=True).eval()
    ids=read(req['input_ids']).unsqueeze(0)
    kwargs={'input_ids':ids,'attention_mask':torch.ones_like(ids),
            'position_ids':read(req['positions']).unsqueeze(1),'use_cache':False,'logits_to_keep':1}
    if req['images']:
        kwargs['pixel_values']=torch.cat([read(image['patches']) for image in req['images']]).to(torch.bfloat16)
        kwargs['image_grid_thw']=torch.tensor([image['grid'] for image in req['images']],device=a.device)
    with torch.inference_mode():
        reference=model(**kwargs).logits[0,-1].float().cpu().numpy()
        native=np.fromfile(str(a.native_output)+'.prefill_logits.f32',dtype='<f4')
        if reference.shape!=native.shape: raise ValueError('Logit trace shape mismatch')
        def metrics(x,y):
            x,y=x.astype(np.float64).reshape(-1),y.astype(np.float64).reshape(-1)
            if not np.isfinite(x).all() or not np.isfinite(y).all(): raise ValueError('Nonfinite trace')
            return {'cosine':float(np.dot(x,y)/max(np.linalg.norm(x)*np.linalg.norm(y),1e-30)),
                    'relative_rmse':float(np.linalg.norm(x-y)/max(np.linalg.norm(x),1e-30)),
                    'max_abs_error':float(np.max(np.abs(x-y)))}
        report={'logits':metrics(reference,native),'reference_top1':int(reference.argmax()),'native_top1':int(native.argmax())}
        if req['images']:
            visual=model.model.visual(kwargs['pixel_values'],grid_thw=kwargs['image_grid_thw']).pooler_output.float().cpu().numpy()
            native_visual=np.fromfile(str(a.native_output)+'.vision.f32',dtype='<f4')
            if visual.size!=native_visual.size: raise ValueError('Vision trace shape mismatch')
            report['vision']=metrics(visual,native_visual)
        generated=json.loads(a.native_output.read_text())['generated_ids']
        report['decode']=[]
        original_positions=kwargs['position_ids']
        for step in range(1,min(len(generated),a.decode_check+1)):
            trace=Path(str(a.native_output)+f'.decode_{step}.f32')
            if not trace.exists():raise ValueError(f'Missing continuation trace: {trace}; rerun native with --trace')
            extension=torch.tensor([generated[:step]],device=a.device,dtype=torch.long)
            kwargs['input_ids']=torch.cat([ids,extension],dim=1)
            kwargs['attention_mask']=torch.ones_like(kwargs['input_ids'])
            tail=torch.arange(req['next_position'],req['next_position']+step,device=a.device).view(1,1,-1).expand(3,1,-1)
            kwargs['position_ids']=torch.cat([original_positions,tail],dim=-1)
            reference_step=model(**kwargs).logits[0,-1].float().cpu().numpy()
            native_step=np.fromfile(trace,dtype='<f4')
            if native_step.shape!=reference_step.shape:raise ValueError('Decode trace shape mismatch')
            report['decode'].append({'step':step,**metrics(reference_step,native_step)})
    print(json.dumps(report,indent=2))
    if any(x['cosine']<a.min_cosine or x['relative_rmse']>a.max_relative_rmse for x in report['decode']):
        raise SystemExit('Continuation numerical smoke check failed')
    # Broad smoke threshold; passing is not a business-quality acceptance result.
    if any(report[key]['cosine']<a.min_cosine or report[key]['relative_rmse']>a.max_relative_rmse for key in ['logits','vision'] if key in report):
        raise SystemExit('Numerical smoke check failed')

if __name__=='__main__': main()