"""Resident native worker benchmark: excludes model startup and Python preprocessing."""
import argparse,json,os,queue,signal,statistics,subprocess,threading,time,tempfile,shutil
from pathlib import Path

def summarize(results,seconds):
 good=[r for r in results if r.get('finish_reason') in ('eos','length')]
 def stats(values):
  values=sorted(values)
  return {'p50':statistics.median(values),'p95':values[min(len(values)-1,int(.95*(len(values)-1)+.5))]} if values else None
 return {'completed':len(results),'successful':len(good),'wall_seconds':seconds,'output_tokens_per_second':sum(len(r['generated_ids']) for r in good)/seconds,
         'ttft_seconds':stats([r['ttft_seconds'] for r in good if r['ttft_seconds']>=0]),'latency_seconds':stats([r['total_seconds'] for r in good]),
         'all_tokens_equal':all(r['generated_ids']==good[0]['generated_ids'] for r in good) if good else None}

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--model',required=True);p.add_argument('--request',required=True);p.add_argument('--out',required=True)
 p.add_argument('--worker',default='build/avi-worker');p.add_argument('--tp',type=int,default=2);p.add_argument('--concurrency',type=int,default=1)
 p.add_argument('--requests',type=int,default=5);p.add_argument('--warmup',type=int,default=1);p.add_argument('--timeout',type=float,default=600)
 p.add_argument('--mode',choices=['baseline','optimized','graph'],default='optimized');p.add_argument('--prefill-chunk',type=int,default=128);p.add_argument('--cache',action='store_true')
 p.add_argument('--inherit-process-group',action='store_true',help=argparse.SUPPRESS)
 p.add_argument('--extra-fusions',action='store_true');p.add_argument('--cublas-prefill',action='store_true')
 p.add_argument('--tp-lm-head',action='store_true');p.add_argument('--reference-prefill',action='store_true')
 p.add_argument('--vector-gemv',action='store_true')
 p.add_argument('--mtp-tokens',type=int,choices=range(6),default=0)
 p.add_argument('--mtp-draft-graph',action='store_true')
 p.add_argument('--gdn-chunk',action='store_true')
 p.add_argument('--flash-prefill',action='store_true')
 p.add_argument('--gdn-cooperative',action='store_true')
 p.add_argument('--gdn-wy',action='store_true')
 p.add_argument('--mtp-verify-graph',action='store_true')
 p.add_argument('--multi-token-gemv',action='store_true')
 p.add_argument('--multi-token-gemv-fp8',action='store_true')
 p.add_argument('--fused-gdn-prepare',action='store_true')
 p.add_argument('--gdn-wy-fused',action='store_true')
 p.add_argument('--gdn-fused-solve',action='store_true')
 p.add_argument('--gdn-tilelang',action='store_true')
 p.add_argument('--fp8-tensor-small',action='store_true')
 p.add_argument('--tilelang-fp8',action='store_true')
 p.add_argument('--fp8-tensor-split',type=int,choices=[1,4],default=1)
 p.add_argument('--tilelang-dir')
 p.add_argument('--gdn-tensor-prefill',action='store_true')
 p.add_argument('--gdn-tensor-chunk',type=int,choices=[32,64],default=64)
 p.add_argument('--reuse-verify-graph',action='store_true')
 p.add_argument('--profile-kernels',action='store_true')
 p.add_argument('--audit-logits',action='store_true')
 p.add_argument('--fused-gdn-conv',action='store_true')
 p.add_argument('--bf16-tp-reduce',action='store_true')
 p.add_argument('--frontend-format',choices=['hf','vllm-string'],default='hf')
 p.add_argument('--frontend-threads',type=int,default=0)
 p.add_argument('--bf16-patches',action='store_true')
 p.add_argument('--spool-dir')
 p.add_argument('--profile-stages',action='store_true')
 p.add_argument('--cache-vision-weights',action='store_true')
 p.add_argument('--weight-cache-mib',type=int,default=0)
 p.add_argument('--frontend-model');p.add_argument('--body')
 p.add_argument('--max-pixels',type=int,default=4000000)
 a=p.parse_args()
 if a.requests<1 or a.warmup<0 or not 1<=a.concurrency<=8 or a.timeout<=0:p.error('Invalid benchmark limits')
 if Path(a.out).exists():p.error('Output exists')
 req=json.loads((Path(a.request)/'request.json').read_text())
 frontend=None
 if bool(a.frontend_model)!=bool(a.body):p.error('--frontend-model and --body must be provided together')
 if a.frontend_model:
  from benchmark_frontend import Frontend
  if a.frontend_threads<0:p.error('--frontend-threads must be nonnegative')
  if a.frontend_threads:
   import torch
   torch.set_num_threads(a.frontend_threads)
  frontend=Frontend(a.frontend_model,a.body,content_format=a.frontend_format,bf16_patches=a.bf16_patches)
 spool=tempfile.TemporaryDirectory(prefix='avi-benchmark-',dir=a.spool_dir)
 prompt_ids=None
 cmd=['mpirun','-np',str(a.tp),a.worker,'--model',str(Path(a.model).resolve()),'--max-context',str(req['max_context']),
      '--max-concurrency',str(a.concurrency),'--prefill-chunk',str(a.prefill_chunk),'--image-cache-mib',str(256 if a.cache else 0),'--prefix-cache-mib',str(512 if a.cache else 0)]
 if a.mode=='baseline':cmd+=['--baseline']
 if a.mode=='graph':cmd+=['--cuda-graph']
 if a.extra_fusions:cmd+=['--extra-fusions']
 if a.cublas_prefill:cmd+=['--cublas-prefill']
 if a.tp_lm_head:cmd+=['--tp-lm-head']
 if a.reference_prefill:cmd+=['--reference-prefill']
 if a.vector_gemv:cmd+=['--vector-gemv']
 cmd+=['--mtp-tokens',str(a.mtp_tokens),'--weight-cache-mib',str(a.weight_cache_mib)]
 if a.mtp_draft_graph:cmd+=['--mtp-draft-graph']
 if a.gdn_chunk:cmd+=['--gdn-chunk']
 if a.flash_prefill:cmd+=['--flash-prefill']
 if a.gdn_cooperative:cmd+=['--gdn-cooperative']
 if a.gdn_wy:cmd+=['--gdn-wy']
 if a.mtp_verify_graph:cmd+=['--mtp-verify-graph']
 if a.multi_token_gemv:cmd+=['--multi-token-gemv']
 if a.multi_token_gemv_fp8:cmd+=['--multi-token-gemv-fp8']
 if a.fused_gdn_prepare:cmd+=['--fused-gdn-prepare']
 if a.gdn_wy_fused:cmd+=['--gdn-wy-fused']
 for flag in ('gdn_fused_solve','gdn_tilelang','fp8_tensor_small','tilelang_fp8'):
  if getattr(a,flag):cmd+=['--'+flag.replace('_','-')]
 cmd+=['--fp8-tensor-split',str(a.fp8_tensor_split)]
 if a.tilelang_dir:cmd+=['--tilelang-dir',str(Path(a.tilelang_dir).resolve())]
 if a.gdn_tensor_prefill:cmd+=['--gdn-tensor-prefill','--gdn-tensor-chunk',str(a.gdn_tensor_chunk)]
 if a.reuse_verify_graph:cmd+=['--reuse-verify-graph']
 if a.profile_kernels:cmd+=['--profile-kernels']
 if a.audit_logits:cmd+=['--audit-logits']
 if a.fused_gdn_conv:cmd+=['--fused-gdn-conv']
 if a.bf16_tp_reduce:cmd+=['--bf16-tp-reduce']
 if a.profile_stages:cmd+=['--profile-stages']
 if a.cache_vision_weights:cmd+=['--cache-vision-weights']
 process=subprocess.Popen(cmd,stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True,bufsize=1,start_new_session=not a.inherit_process_group);events=queue.Queue()
 def reader():
  for line in process.stdout:
   try:events.put(json.loads(line))
   except ValueError:pass
  events.put({'event':'fatal','message':'Worker exited'})
 threading.Thread(target=reader,daemon=True).start()
 def send(command):process.stdin.write(json.dumps(command)+'\n');process.stdin.flush()
 def receive(deadline):
  event=events.get(timeout=max(.01,deadline-time.monotonic()))
  if event.get('event') in ('error','fatal'):raise RuntimeError(event)
  return event
 def run(count,prefix):
  nonlocal prompt_ids
  submitted=0;running=set();results=[];mtp={};stages={};kernels={};started={};first={};paths={};frontend_seconds={};start=time.monotonic();deadline=start+a.timeout
  while len(results)<count:
   while submitted<count and len(running)<a.concurrency:
    name=prefix+str(submitted);running.add(name);submitted+=1
    started[name]=time.monotonic();path=Path(a.request).resolve()
    if frontend:
     path=Path(spool.name)/name
     prepared=frontend.prepare(path,a.max_pixels,req['max_context'],req['max_new_tokens'])
     import numpy as np
     current=np.fromfile(path/prepared['input_ids']['file'],dtype='<i8').tolist()
     if prompt_ids is not None and current!=prompt_ids:raise RuntimeError('Preprocessing is not repeatable')
     prompt_ids=current;paths[name]=path
    frontend_seconds[name]=time.monotonic()-started[name]
    send({'op':'submit','id':name,'request':str(path),'sampling':{'temperature':0,'timeout_seconds':a.timeout}})
   event=receive(deadline)
   if event.get('event')=='token' and event.get('id') in started:first.setdefault(event['id'],time.monotonic()-started[event['id']])
   if event.get('event')=='mtp_stats':mtp[event['id']]=event
   if event.get('event')=='stage_stats':stages[event['id']]=event
   if event.get('event')=='kernel_stats':kernels[event['id']]=event
   if event.get('event')=='done' and event.get('id') in running:
    event['mtp']=mtp.get(event['id'])
    event['stages']=stages.get(event['id'])
    event['kernel_times']=kernels.get(event['id'])
    if frontend:
     event['text']=frontend.processor.tokenizer.decode(event['generated_ids'],skip_special_tokens=True)
     event['native_worker_seconds']=event['total_seconds']
     event['native_worker_ttft_seconds']=event['ttft_seconds']
     event['frontend_seconds']=frontend_seconds[event['id']]
     event['total_seconds']=time.monotonic()-started[event['id']]
     event['ttft_seconds']=first.get(event['id'],-1)
     shutil.rmtree(paths.pop(event['id']))
    running.remove(event['id']);results.append(event)
  return results,time.monotonic()-start
 try:
  deadline=time.monotonic()+a.timeout
  while receive(deadline).get('event')!='ready':pass
  warmup_results=run(a.warmup,'warmup-')[0] if a.warmup else []
  results,seconds=run(a.requests,'measure-')
  report={'tp_lm_head':a.tp_lm_head,'reference_prefill':a.reference_prefill,'extra_fusions':a.extra_fusions,'cublas_prefill':a.cublas_prefill,'mode':a.mode,'cache_enabled':a.cache,'concurrency':a.concurrency,'prefill_chunk':a.prefill_chunk,'warmup':a.warmup,'input_tokens':req['input_ids']['shape'][0],
          'max_new_tokens':req['max_new_tokens'],**summarize(results,seconds),'results':results,'note':'Resident engine; no trace. Includes per-request Graph capture when enabled. Multirequest batches use eager decode.'}
  if prompt_ids is not None:report['input_tokens']=len(prompt_ids)
  report['vector_gemv']=a.vector_gemv
  report.update(mtp_tokens=a.mtp_tokens,weight_cache_mib=a.weight_cache_mib,mtp_draft_graph=a.mtp_draft_graph)
  report['gdn_chunk']=a.gdn_chunk
  report.update(flash_prefill=a.flash_prefill,cache_vision_weights=a.cache_vision_weights)
  report['warmup_cache_baseline']=warmup_results[-1].get('cache',{}) if warmup_results else {}
  report['synchronized_diagnostic']=a.profile_stages or a.profile_kernels or a.audit_logits
  report.update(gdn_wy=a.gdn_wy,mtp_verify_graph=a.mtp_verify_graph,profile_kernels=a.profile_kernels,audit_logits=a.audit_logits,
    multi_token_gemv=a.multi_token_gemv,multi_token_gemv_fp8=a.multi_token_gemv_fp8,fused_gdn_prepare=a.fused_gdn_prepare,gdn_wy_fused=a.gdn_wy_fused,reuse_verify_graph=a.reuse_verify_graph,gdn_tensor_prefill=a.gdn_tensor_prefill,gdn_tensor_chunk=a.gdn_tensor_chunk,gdn_fused_solve=a.gdn_fused_solve,gdn_tilelang=a.gdn_tilelang,fp8_tensor_small=a.fp8_tensor_small,tilelang_fp8=a.tilelang_fp8,fp8_tensor_split=a.fp8_tensor_split,tilelang_dir=a.tilelang_dir)
  report.update(fused_gdn_conv=a.fused_gdn_conv,gdn_cooperative=a.gdn_cooperative,bf16_tp_reduce=a.bf16_tp_reduce,frontend_format=a.frontend_format,frontend_threads=a.frontend_threads,bf16_patches=a.bf16_patches,spool_dir=a.spool_dir)
  report['timing_scope']='CPU image decode/tokenization/preprocessing + disk IPC + native inference + output decoding' if frontend else 'prepared input + native inference'
  report['prompt_token_ids']=prompt_ids
  with open(a.out,'x',encoding='utf-8') as f:json.dump(report,f,indent=2)
  print(json.dumps({k:v for k,v in report.items() if k not in ('results','prompt_token_ids')},indent=2))
  send({'op':'shutdown'});process.stdin.close();process.wait(timeout=30)
  if process.returncode or report['successful']!=a.requests:raise RuntimeError('Worker or requests failed')
 finally:
  if process.poll() is None:
   process.terminate() if a.inherit_process_group else os.killpg(process.pid,signal.SIGTERM)
   try:process.wait(timeout=10)
   except subprocess.TimeoutExpired:process.kill() if a.inherit_process_group else os.killpg(process.pid,signal.SIGKILL);process.wait()
  spool.cleanup()
if __name__=='__main__':main()
