"""Run native optimization ablations and compare against installed vLLM MTP."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from benchmark_mtp_upstream import stop_group

def canonical(tokens,eos):
    tokens=list(tokens)
    if tokens and tokens[-1] in eos:tokens.pop()
    return tokens

def compare_outputs(results,reference,eos):
    expected=canonical(reference,eos)
    differences=[]
    for row in results:
        actual=canonical(row['generated_ids'],eos)
        if actual!=expected:
            first=next((i for i,(a,b) in enumerate(zip(actual,expected)) if a!=b),min(len(actual),len(expected)))
            differences.append({'first_difference':first,'actual_length':len(actual),'reference_length':len(expected),
                                'actual_near':actual[max(0,first-3):first+5],'reference_near':expected[max(0,first-3):first+5]})
    return differences

def run(command,log,timeout):
    print('Starting '+log.stem+'; log: '+str(log),flush=True)
    with log.open('x',encoding='utf-8') as stream:
        process=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            return process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            return 124
        finally:
            stop_group(process)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('model','hf-model','request','body','out'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--requests',type=int,default=5);p.add_argument('--timeout',type=float,default=1800)
    p.add_argument('--weight-cache-mib',type=int,default=24576);p.add_argument('--max-pixels',type=int,default=4000000)
    a=p.parse_args()
    if os.name!='posix':p.error('Run on the Linux A100 server')
    if a.requests<1 or a.timeout<=0 or a.weight_cache_mib<0:p.error('Invalid limits')
    a.out.mkdir(parents=True,exist_ok=False)
    request=json.loads((a.request/'request.json').read_text());eos=request['eos_token_ids']
    shared=['--tp-lm-head','--vector-gemv','--extra-fusions','--cublas-prefill']
    cache=['--weight-cache-mib',str(a.weight_cache_mib)]
    profiles=[
        ('native',[],1),('native-graph',['--mode','graph'],1),
        ('cached',cache,1),('chunked',cache+['--gdn-chunk'],1),
        *[(f'mtp{k}',['--mtp-tokens',str(k)],1) for k in (1,2,3)],
        ('mtp2-cached',['--mtp-tokens','2']+cache,1),
        ('mtp3-cached',['--mtp-tokens','3']+cache,1),
        ('mtp3-chunked',['--mtp-tokens','3','--gdn-chunk']+cache,1),
        ('mtp3-graph',['--mtp-tokens','3','--mtp-draft-graph']+cache,1),
        ('combined',['--mtp-tokens','3','--mtp-draft-graph','--gdn-chunk']+cache,1),
        ('combined-c2',['--mtp-tokens','3','--mtp-draft-graph','--gdn-chunk']+cache,2),
    ]
    summary={'native':[],'vllm':None,'comparison':[],
             'note':'Same HF source and request body; both include CPU preprocessing, exclude model load and warmup. Native includes disk IPC; vLLM uses offline chat, not production HTTP. One workload is not general business acceptance.'}
    reference=None;reports={}
    def save():(a.out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    for name,flags,concurrency in profiles:
        output=a.out/(name+'.json')
        command=[sys.executable,'tools/benchmark_worker.py','--model',str(a.model),'--request',str(a.request),
                 '--frontend-model',str(a.hf_model),'--body',str(a.body),'--out',str(output),
                 '--max-pixels',str(a.max_pixels),'--requests',str(a.requests),'--warmup','1','--prefill-chunk','512',
                 '--concurrency',str(concurrency),'--timeout',str(a.timeout),'--inherit-process-group']+shared+flags
        code=run(command,a.out/(name+'.log'),a.timeout)
        row={'name':name,'exit_code':code,'concurrency':concurrency,'status':'failed'}
        if code==0 and output.exists():
            report=json.loads(output.read_text());reports[name]=report
            if reference is None and name=='native':reference=report['results'][0]['generated_ids']
            differences=compare_outputs(report['results'],reference,eos) if reference is not None else None
            row.update(status='passed' if differences==[] else 'output_mismatch',
                       latency_seconds=report['latency_seconds'],ttft_seconds=report['ttft_seconds'],
                       output_tokens_per_second=report['output_tokens_per_second'],differences=differences,
                       mtp=[r.get('mtp') for r in report['results']])
        summary['native'].append(row);save();print(json.dumps(row),flush=True)
    command=[sys.executable,'tools/benchmark_mtp_upstream.py','--model',str(a.hf_model),'--body',str(a.body),
             '--out',str(a.out/'vllm'),'--requests',str(a.requests),'--max-tokens',str(request['max_new_tokens']),
             '--max-context',str(request['max_context']),'--max-pixels',str(a.max_pixels),'--timeout',str(a.timeout)]
    code=run(command,a.out/'vllm.log',a.timeout*4+120)
    summary['vllm']={'exit_code':code};vpath=a.out/'vllm/summary.json'
    if vpath.exists():summary['vllm']['profiles']=json.loads(vpath.read_text())
    for name,report in reports.items():
        if report['concurrency']!=1:continue
        for window in (2,3):
            path=a.out/f'vllm/mtp{window}.json'
            if not path.exists():continue
            upstream=json.loads(path.read_text());target=upstream['results'][0]
            inputs_match=all(r['prompt_token_ids']==report['prompt_token_ids'] for r in upstream['results'])
            differences=compare_outputs(report['results'],target['generated_ids'],eos)
            stable=not compare_outputs(upstream['results'],target['generated_ids'],eos)
            row={'native':name,'vllm':f'mtp{window}','prompt_ids_equal':inputs_match,
                 'output_ids_equal':not differences,'vllm_repeatable':stable,'differences':differences,
                 'latency_ratio_vllm_over_native':None}
            if inputs_match and not differences and stable:
                row['latency_ratio_vllm_over_native']=upstream['latency_p50_seconds']/report['latency_seconds']['p50']
            summary['comparison'].append(row)
    save()
    print('Completed; inspect '+str(a.out/'summary.json')+'. Ratio > 1 means native is faster ONLY for matching inputs/outputs.',flush=True)
    if code or any(r['status']!='passed' for r in summary['native']):raise SystemExit(1)
if __name__=='__main__':main()
