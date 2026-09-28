"""Resident native worker benchmark: excludes model startup and Python preprocessing."""
import argparse,json,os,queue,signal,statistics,subprocess,threading,time
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
 a=p.parse_args()
 if a.requests<1 or a.warmup<0 or not 1<=a.concurrency<=8 or a.timeout<=0:p.error('Invalid benchmark limits')
 if Path(a.out).exists():p.error('Output exists')
 req=json.loads((Path(a.request)/'request.json').read_text())
 cmd=['mpirun','-np',str(a.tp),a.worker,'--model',str(Path(a.model).resolve()),'--max-context',str(req['max_context']),
      '--max-concurrency',str(a.concurrency),'--prefill-chunk',str(a.prefill_chunk),'--image-cache-mib',str(256 if a.cache else 0),'--prefix-cache-mib',str(512 if a.cache else 0)]
 if a.mode=='baseline':cmd+=['--baseline']
 if a.mode=='graph':cmd+=['--cuda-graph']
 if a.extra_fusions:cmd+=['--extra-fusions']
 if a.cublas_prefill:cmd+=['--cublas-prefill']
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
  submitted=0;running=set();results=[];start=time.monotonic();deadline=start+a.timeout
  while len(results)<count:
   while submitted<count and len(running)<a.concurrency:
    name=prefix+str(submitted);running.add(name);submitted+=1
    send({'op':'submit','id':name,'request':str(Path(a.request).resolve()),'sampling':{'temperature':0,'timeout_seconds':a.timeout}})
   event=receive(deadline)
   if event.get('event')=='done' and event.get('id') in running:
    running.remove(event['id']);results.append(event)
  return results,time.monotonic()-start
 try:
  deadline=time.monotonic()+a.timeout
  while receive(deadline).get('event')!='ready':pass
  if a.warmup:run(a.warmup,'warmup-')
  results,seconds=run(a.requests,'measure-')
  report={'extra_fusions':a.extra_fusions,'cublas_prefill':a.cublas_prefill,'mode':a.mode,'cache_enabled':a.cache,'concurrency':a.concurrency,'prefill_chunk':a.prefill_chunk,'warmup':a.warmup,'input_tokens':req['input_ids']['shape'][0],
          'max_new_tokens':req['max_new_tokens'],**summarize(results,seconds),'results':results,'note':'Resident engine; prepared inputs; no trace. Includes per-request Graph capture when enabled. Multirequest batches use eager decode.'}
  with open(a.out,'x',encoding='utf-8') as f:json.dump(report,f,indent=2)
  print(json.dumps({k:v for k,v in report.items() if k!='results'},indent=2))
  send({'op':'shutdown'});process.stdin.close();process.wait(timeout=30)
  if process.returncode or report['successful']!=a.requests:raise RuntimeError('Worker or requests failed')
 finally:
  if process.poll() is None:
   process.terminate() if a.inherit_process_group else os.killpg(process.pid,signal.SIGTERM)
   try:process.wait(timeout=10)
   except subprocess.TimeoutExpired:process.kill() if a.inherit_process_group else os.killpg(process.pid,signal.SIGKILL);process.wait()
if __name__=='__main__':main()
