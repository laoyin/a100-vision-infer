"""GPU integration test: simultaneous jobs, cancellation, prefix reuse, repeatability."""
import argparse,json,queue,subprocess,threading,time,os,signal,tempfile,shutil
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model',required=True);p.add_argument('--request',required=True)
    p.add_argument('--host-cache',action='store_true');p.add_argument('--worker',default='build/avi-worker');p.add_argument('--tp',type=int,default=2);p.add_argument('--cuda-graph',action='store_true')
    p.add_argument('--tp-lm-head',action='store_true');p.add_argument('--vector-gemv',action='store_true');p.add_argument('--mtp-tokens',type=int,default=0);p.add_argument('--mtp-draft-graph',action='store_true');a=p.parse_args()
    req=json.loads((Path(a.request)/'request.json').read_text())
    cmd=['mpirun','-np',str(a.tp),a.worker,'--model',a.model,'--max-context',str(req['max_context']),
         '--max-concurrency','2','--prefill-chunk','4']+(['--cuda-graph'] if a.cuda_graph else [])
    temp=tempfile.TemporaryDirectory(prefix='avi-worker-test-')
    if a.tp_lm_head:cmd+=['--tp-lm-head']
    if a.vector_gemv:cmd+=['--vector-gemv']
    if a.mtp_tokens:cmd+=['--mtp-tokens',str(a.mtp_tokens)]
    if a.mtp_draft_graph:cmd+=['--mtp-draft-graph']
    variants=Path(temp.name)
    if a.host_cache:
        config=json.loads((Path(a.model)/'manifest.json').read_text())['config']['text_config'];tp=a.tp
        n=req['input_ids']['shape'][0];size=4*config['vocab_size']
        for kind in config['layer_types']:
            if kind=='full_attention':size+=4*n*(config['num_key_value_heads']//tp)*config['head_dim']
            else:
                h=config['linear_num_value_heads']//tp;hk=config['linear_num_key_heads']//tp;k=config['linear_key_head_dim'];v=config['linear_value_head_dim']
                size+=4*h*k*v+2*(2*hk*k+h*v)*(config['linear_conv_kernel_dim']-1)
        cmd+=['--prefix-cache-bytes',str(size),'--host-prefix-cache-mib','64']
    process=subprocess.Popen(cmd,stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True,bufsize=1,start_new_session=True);events=queue.Queue()
    def reader():
        for line in process.stdout:
            try:events.put(json.loads(line))
            except ValueError:pass
        events.put({'event':'fatal','message':'Worker exited'})
    threading.Thread(target=reader,daemon=True).start()
    def send(x):process.stdin.write(json.dumps(x)+'\n');process.stdin.flush()
    def wait(predicate,allow_error=False):
        until=time.monotonic()+300
        while time.monotonic()<until:
            event=events.get(timeout=max(0.1,until-time.monotonic()))
            if event['event']=='fatal' or event['event']=='error' and not allow_error:raise RuntimeError(event)
            if predicate(event):return event
        raise TimeoutError('Worker test timeout')
    try:
        wait(lambda e:e['event']=='ready')
        for name in ['a','b']:send({'op':'submit','id':name,'request':str(Path(a.request).resolve())})
        results={}
        while len(results)<2:
            e=wait(lambda e:e['event']=='done');results[e['id']]=e
        assert results['a']['generated_ids']==results['b']['generated_ids'],'Concurrent state contamination or numerical divergence'
        if a.host_cache:
            # A different prompt displaces A's exact checkpoint into pinned host memory.
            variant=variants/'other';shutil.copytree(a.request,variant)
            import numpy as np
            input_file=variant/req['input_ids']['file'];tokens=np.fromfile(input_file,dtype='<i8')
            special={req['image_token_id']}
            index=next(i for i,t in enumerate(tokens) if int(t) not in special)
            tokens[index]=(int(tokens[index])+7)%config['vocab_size']
            if int(tokens[index]) in special:tokens[index]=0
            input_file.write_bytes(tokens.tobytes())
            send({'op':'submit','id':'other','request':str(variant.resolve())})
            wait(lambda e:e['event']=='done' and e['id']=='other')
        send({'op':'submit','id':'cached','request':str(Path(a.request).resolve())})
        cached=wait(lambda e:e['event']=='done' and e['id']=='cached')
        assert cached['generated_ids']==results['a']['generated_ids'],'Prefix restore changed generation'
        if not a.mtp_tokens:assert cached['cache']['prefix_hits']>0,'Expected exact prompt cache hit'
        else:assert cached['cache']['prefix_hits']==0,'MTP must not reuse incomplete prefix state'
        if a.host_cache:assert cached['cache']['host_prefix_hits']>0,'Host restore path was not exercised'
        invalid=variants/'invalid';invalid.mkdir();bad=dict(req);bad['max_new_tokens']=req['max_context']+1
        (invalid/'request.json').write_text(json.dumps(bad))
        send({'op':'submit','id':'invalid','request':str(invalid.resolve())})
        rejected=wait(lambda e:e.get('id')=='invalid',allow_error=True)
        assert rejected['event']=='error','Invalid budget must be rejected without killing worker'
        send({'op':'submit','id':'cancelled','request':str(Path(a.request).resolve())});send({'op':'cancel','id':'cancelled'})
        cancelled=wait(lambda e:e['event']=='done' and e['id']=='cancelled')
        assert cancelled['finish_reason']=='cancelled','Cancel not honored'
        send({'op':'shutdown'});process.stdin.close();process.wait(timeout=30)
        assert process.returncode==0
        print(json.dumps({'status':'passed','cuda_graph':a.cuda_graph,'results':results,'cached':cached},indent=2))
    finally:
        if process.poll() is None:
            os.killpg(process.pid,signal.SIGTERM)
            try:process.wait(timeout=30)
            except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
        temp.cleanup()

if __name__=='__main__':main()
