"""Bounded, warmup-excluded native TP2 Nsight capture; no dependency installs."""
import argparse
from contextlib import closing
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import sqlite3
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]


def execute(command,log,timeout,env=None):
    print('Running '+log.name,flush=True)
    with log.open('x',encoding='utf-8') as stream:
        process=subprocess.Popen([str(c) for c in command],cwd=ROOT,env=env,stdout=stream,
                                 stderr=subprocess.STDOUT,start_new_session=True)
        try:
            return process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            return 124
        finally:
            # This job owns a new process group, including MPI ranks/profilers.
            try:os.killpg(process.pid,signal.SIGTERM)
            except ProcessLookupError:pass
            try:process.wait(timeout=5)
            except subprocess.TimeoutExpired:pass
            finally:
                try:os.killpg(process.pid,signal.SIGKILL)
                except ProcessLookupError:pass
            process.wait()



def native_command(previous,model,body,out,tokens,context,optimized,nsys):
    flags=['--tp-lm-head','--vector-gemv','--extra-fusions','--cublas-prefill',
           '--flash-prefill','--fused-gdn-prepare','--gdn-cooperative','--fused-gdn-conv',
           '--bf16-tp-reduce','--multi-token-gemv-fp8','--mtp-tokens','3','--mtp-draft-graph']
    if tokens>1:flags+=['--mtp-verify-graph','--reuse-verify-graph']
    if optimized:flags+=['--fused-residual-norm','--gpu-candidates']
    return [sys.executable,'-u','tools/benchmark_worker.py','--model',str(previous/'fp8-mtp-tp2'),
            '--request',str(previous/'request'),'--frontend-model',str(model),'--body',str(body),
            '--out',str(out/'benchmark.json'),'--max-new-tokens',str(tokens),'--max-context',str(context),
            '--requests','1','--warmup','1','--concurrency','1','--prefill-chunk','512',
            '--weight-cache-mib','24576','--frontend-format','vllm-string','--frontend-threads','4',
            '--bf16-patches','--spool-dir','/dev/shm','--nsys-profile-dir',str(out/'nsys'),
            '--nsys-bin',str(nsys),'--inherit-process-group']+flags


def sqlite_activity(path):
    if not path.is_file():return {'status':'missing_sqlite','kernel_count':0}
    with closing(sqlite3.connect(path.as_uri()+'?mode=ro',uri=True)) as db:
        tables={row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        table='CUPTI_ACTIVITY_KIND_KERNEL'
        if table not in tables:return {'status':'missing_kernel_table','kernel_count':0}
        count,start,end=db.execute('SELECT count(*),min(start),max(end) FROM CUPTI_ACTIVITY_KIND_KERNEL').fetchone()
    return {'status':'captured' if count else 'empty_capture','kernel_count':count,
            'first_kernel_ns':start,'last_kernel_ns':end,
            'kernel_span_seconds':(end-start)/1e9 if count else 0}


def ncu_command(ncu,name,out):
    specs={'fp8':('w8a16',['build/avi-linear-bench','--repeats','3']),
           'gdn-solve':('intra_solve_gdn',['build/avi-gdn-bench','--tokens','512','--repeats','3']),
           'gdn-state':('propagate_gdn',['build/avi-gdn-bench','--tokens','512','--repeats','3'])}
    kernel,binary=specs[name]
    return [str(ncu),'--kernel-name-base','demangled','--kernel-name','regex:'+kernel,
            '--launch-skip','1','--launch-count','2','--section','SpeedOfLight',
            '--section','LaunchStats','--section','Occupancy','--section','MemoryWorkloadAnalysis',
            '--export',str(out/name)]+binary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--previous',type=Path,required=True)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--body',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--decode-tokens',type=int,default=512)
    p.add_argument('--max-context',type=int,default=20480)
    p.add_argument('--max-pixels',type=int,default=4000000)
    p.add_argument('--timeout',type=int,default=1800)
    p.add_argument('--variant',choices=['both','reference','optimized'],default='both')
    p.add_argument('--ncu',choices=['auto','off','required'],default='auto')
    p.add_argument('--skip-build',action='store_true')
    a=p.parse_args()
    if os.name!='posix':p.error('Run on the Linux A100 server')
    if not 2<=a.decode_tokens<=4096 or a.max_context<=a.decode_tokens or a.timeout<60 or a.max_pixels<1:p.error('Invalid capture limits')
    a.previous=a.previous.resolve();a.model=a.model.resolve();a.body=a.body.resolve();a.out=a.out.resolve()
    for path in (a.previous/'fp8-mtp-tp2/manifest.json',a.previous/'request/request.json',a.model/'config.json',a.body):
        if not path.is_file():p.error('Missing '+str(path))
    body=json.loads(a.body.read_text(encoding='utf-8'))
    if not body.get('messages') or body.get('tools') or body.get('response_format'):p.error('Use plain messages with prompted JSON, without tools/response_format')
    a.out.mkdir(parents=True,exist_ok=False)
    summary={'format':'avi-profile-v1','jobs':[],'ncu':[],
             'body_sha256':hashlib.sha256(a.body.read_bytes()).hexdigest(),
             'artifact_sha256':hashlib.sha256((a.previous/'fp8-mtp-tp2/manifest.json').read_bytes()).hexdigest(),
             'decode_token_budget':a.decode_tokens,'max_context':a.max_context,'max_pixels':a.max_pixels,
             'timing_valid_for_speed_claims':False,
             'scope':'Native rank-local CPU/CUDA/MPI timelines after warmup; Python frontend timing is in benchmark.json, not Python stack traces. Decode is a bounded sample, not complete JSON acceptance. NCU uses synthetic isolated shapes, not NCCL replay.'}
    def save():(a.out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    env=dict(os.environ,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',AVI_NVTX='1')
    env.pop('AVI_TILELANG_DIR',None)
    if os.geteuid()==0:env.update(OMPI_ALLOW_RUN_AS_ROOT='1',OMPI_ALLOW_RUN_AS_ROOT_CONFIRM='1')
    nsys=shutil.which(os.environ.get('NSYS_BIN','nsys'))
    ncu=shutil.which(os.environ.get('NCU_BIN','ncu'))
    summary['nsys']=nsys;summary['ncu_binary']=ncu;summary['visible_devices']=env.get('CUDA_VISIBLE_DEVICES')
    summary['revision']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    save()
    if not nsys:
        summary['error']='nsys not found. Set NSYS_BIN to an existing executable; nothing installed.';save();raise SystemExit(summary['error'])
    for name,command in [('nsys-version',[nsys,'--version']),('nsys-help',[nsys,'profile','--help']),
                         ('gpu-info',['nvidia-smi','-q']),('gpu-topology',['nvidia-smi','topo','-m'])]:
        if shutil.which(command[0]):execute(command,a.out/(name+'.log'),30,env)
    help_text=(a.out/'nsys-help.log').read_text(encoding='utf-8',errors='replace')
    required=['--capture-range','--capture-range-end','--cuda-graph-trace','--mpi-impl']
    if any(flag not in help_text for flag in required):
        summary['error']='Installed nsys lacks required capture/Graph/MPI flags; see nsys-help.log';save();raise SystemExit(summary['error'])
    if not a.skip_build:
        code=execute(['bash','scripts/build.sh'],a.out/'build.log',a.timeout,env)
        if code:summary['error']='build failed';save();raise SystemExit(code)
    for binary in ('avi-worker','avi-linear-bench','avi-gdn-bench'):
        if not (ROOT/'build'/binary).is_file():p.error('Missing build/'+binary)
    failed=False
    variants=['reference','optimized'] if a.variant=='both' else [a.variant]
    for workload,tokens in [('first-token',1),('decode-sample',a.decode_tokens)]:
        for variant in variants:
            out=a.out/(workload+'-'+variant);out.mkdir()
            command=native_command(a.previous,a.model,a.body,out,tokens,a.max_context,variant=='optimized',nsys)
            command+=['--timeout',str(a.timeout),'--max-pixels',str(a.max_pixels)]
            (out/'command.json').write_text(json.dumps(command,indent=2),encoding='utf-8')
            code=execute(command,out/'run.log',a.timeout+600,env)
            row={'name':out.name,'exit_code':code,'ranks':[],'status':'failed'}
            for rank in range(2):
                rep=out/'nsys'/f'rank-{rank}.nsys-rep'
                record={'rank':rank,'report_exists':rep.is_file()}
                if rep.is_file():
                    command=[nsys,'stats','--report','cuda_gpu_kern_sum,cuda_api_sum,cuda_gpu_mem_time_sum,nvtx_sum',
                             '--format','csv','--output',str(rep.with_suffix('')),str(rep)]
                    record['stats_exit']=execute(command,out/f'stats-rank-{rank}.log',300,env)
                    try:record.update(sqlite_activity(rep.with_suffix('.sqlite')))
                    except (sqlite3.Error,OSError) as error:record.update(status='sqlite_error',error=str(error))
                row['ranks'].append(record)
            bench=out/'benchmark.json'
            if bench.is_file():
                report=json.loads(bench.read_text(encoding='utf-8'))
                row['profiler']=report.get('profiler');row['output_tokens']=[len(r['generated_ids']) for r in report.get('results',[])]
                row['sample_sufficient']=bool(row['output_tokens']) and (tokens==1 or min(row['output_tokens'])>=128)
            if code==0 and all(r.get('status')=='captured' for r in row['ranks']) and row.get('sample_sufficient'):
                row['status']='captured'
            failed=failed or row['status']!='captured';summary['jobs'].append(row);save();print(json.dumps(row),flush=True)
    if a.ncu=='off' or not ncu:
        summary['ncu_status']='disabled' if a.ncu=='off' else 'skipped_missing_tool'
        failed=failed or (a.ncu=='required' and not ncu)
    else:
        execute([ncu,'--version'],a.out/'ncu-version.log',30,env)
        out=a.out/'ncu';out.mkdir()
        for name in ('fp8','gdn-solve','gdn-state'):
            command=ncu_command(ncu,name,out)
            (out/(name+'-command.json')).write_text(json.dumps(command,indent=2),encoding='utf-8')
            code=execute(command,out/(name+'.log'),a.timeout,env)
            report=out/(name+'.ncu-rep');log=(out/(name+'.log')).read_text(encoding='utf-8',errors='replace')
            ok=code==0 and report.is_file() and report.stat().st_size>0 and 'No kernels were profiled' not in log
            row={'name':name,'exit_code':code,'status':'captured' if ok else 'failed'}
            if 'ERR_NVGPUCTRPERM' in log:row['error']='GPU performance counters are not permitted; no system permissions were changed'
            if ok:row['csv_exit']=execute([ncu,'--import',str(report),'--page','raw','--csv'],out/(name+'.csv'),300,env)
            summary['ncu'].append(row);failed=failed or not ok;save()
        summary['ncu_status']='completed' if all(r['status']=='captured' for r in summary['ncu']) else 'failed'
    summary['output_checks']=[]
    if a.variant=='both':
        for workload in ('first-token','decode-sample'):
            paths=[a.out/(workload+'-'+v)/'benchmark.json' for v in variants]
            if all(p.is_file() for p in paths):
                reports=[json.loads(p.read_text()) for p in paths]
                equal=reports[0]['prompt_token_ids']==reports[1]['prompt_token_ids'] and reports[0]['results'][0]['generated_ids']==reports[1]['results'][0]['generated_ids']
                summary['output_checks'].append({'workload':workload,'tokens_equal':equal})
                failed=failed or not equal
    summary['status']='needs_attention' if failed else 'captured';save()
    print('Retain complete profiler directory: '+str(a.out),flush=True)
    raise SystemExit(1 if failed else 0)


if __name__=='__main__':main()
