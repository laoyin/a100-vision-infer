"""Focused prefill ablations using existing FP8 artifacts and vLLM reports."""
import argparse
import json
from pathlib import Path
import sys
from native_mtp_matrix import run, compare_outputs


def prompt_difference(actual, expected):
    if actual == expected:
        return None
    first = next((i for i, (a, b) in enumerate(zip(actual, expected)) if a != b),
                 min(len(actual), len(expected)))
    return dict(first_difference=first, native_length=len(actual), vllm_length=len(expected),
                native_near=actual[max(0, first-5):first+8],
                vllm_near=expected[max(0, first-5):first+8])


def acceptance(profiles, comparisons, upstream_count=2):
    by_native={}
    for row in comparisons:
        ratio=row['latency_ratio_vllm_over_native']
        if ratio is not None:
            by_native.setdefault(row['native'],[]).append(ratio)
    # A winner must beat the fastest matching vLLM configuration, not merely
    # one slower upstream window. Missing comparisons never count as a win.
    ratios=[min(values) for values in by_native.values() if len(values)==upstream_count]
    return {'native_correctness_passed':all(r['status']=='passed' for r in profiles),
            'cross_engine_matching_comparisons':sum(len(v) for v in by_native.values()),
            'best_latency_ratio_vs_fastest_vllm':max(ratios) if ratios else None,
            'faster_on_this_workload':bool(ratios and max(ratios)>1),
            'scope':'C1 latency on this image only. Both vLLM MTP2/3 must match. C2 has no matched vLLM concurrency result.'}


def fused_profiles(frontend,gdn):
    base=frontend+gdn+['--bf16-tp-reduce']
    graph=['--mtp-verify-graph','--reuse-verify-graph']
    prepare=['--fused-gdn-prepare']
    gemv=['--multi-token-gemv']
    fp8=['--multi-token-gemv-fp8']
    wy=['--gdn-wy-fused']
    return [
        ('reference',base,512,24576),
        ('graph-pool',base+graph,512,24576),
        ('prepare',base+prepare,512,24576),
        ('shared-gemv',base+gemv,512,24576),
        ('shared-fp8',base+fp8,512,24576),
        ('wy-fused',base+prepare+wy,512,24576),
        ('combined',base+graph+prepare+gemv,512,24576),
        ('combined-fp8',base+graph+prepare+fp8,512,24576),
        ('combined-1024',base+graph+prepare+gemv,1024,24576),
        ('combined-2048',base+graph+prepare+gemv,2048,24576),
        ('combined-wy',base+graph+prepare+gemv+wy,2048,24576),
        ('combined-c2',base+graph+prepare+gemv+['--concurrency','2'],512,24576),
        ('diagnostic',base+['--profile-stages','--profile-kernels','--audit-logits'],512,24576),
        ('diagnostic-fused',base+prepare+gemv+['--profile-stages','--profile-kernels'],512,24576),
    ]


def counter_exercised(report,key):
    # This suite measures sequential C1 requests. A cumulative warmup counter
    # alone must not make an unexercised measured request pass.
    previous=(report.get('warmup_cache_baseline') or {}).get(key,0)
    for row in report.get('results',[]):
        current=(row.get('cache') or {}).get(key,0)
        if current<=previous:
            return False
        previous=current
    return bool(report.get('results'))


def tensor_exercised(report):
    return counter_exercised(report,'gdn_tensor_calls')


def tiled_profiles(frontend,gdn,tilelang_dir=None):
    base=frontend+gdn+['--bf16-tp-reduce','--fused-gdn-prepare']
    graph=['--mtp-verify-graph','--reuse-verify-graph']
    shared=['--multi-token-gemv-fp8']
    old=base+graph+shared
    fused=['--gdn-fused-solve']
    tensor=['--gdn-tensor-prefill']
    tc=['--fp8-tensor-small']
    rows=[
        ('reference',old,512,24576),
        ('tensor-32',old+tensor+['--gdn-tensor-chunk','32'],512,24576),
        ('tensor-64',old+tensor,512,24576),
        ('fused-32',old+fused+['--gdn-tensor-chunk','32'],512,24576),
        ('fused-64',old+fused,512,24576),
        ('fp8-tensor-1',old+tc,512,24576),
        ('fp8-tensor-4',old+tc+['--fp8-tensor-split','4'],512,24576),
        ('combined-32',old+fused+tc+['--gdn-tensor-chunk','32'],512,24576),
        ('combined-64-s4',old+fused+tc+['--fp8-tensor-split','4'],512,24576),
        ('combined-2048',old+fused+tc+['--gdn-tensor-chunk','32'],2048,24576),
        ('diagnostic-fused',base+shared+fused+tc+['--profile-stages','--profile-kernels'],512,24576),
    ]
    if tilelang_dir:
        plugin=['--tilelang-dir',str(Path(tilelang_dir).resolve())]
        tg=['--gdn-tilelang']
        tf=['--tilelang-fp8']
        rows += [
            ('tilelang-gdn32',old+plugin+tg+['--gdn-tensor-chunk','32'],512,24576),
            ('tilelang-gdn64',old+plugin+tg,512,24576),
            ('tilelang-fp8-1',old+plugin+tf,512,24576),
            ('tilelang-fp8-4',old+plugin+tf+['--fp8-tensor-split','4'],512,24576),
            ('tilelang-combined32',old+plugin+tg+tf+['--gdn-tensor-chunk','32'],512,24576),
            ('tilelang-combined64-s4',old+plugin+tg+tf+['--fp8-tensor-split','4'],512,24576),
            ('tilelang-combined2048',old+plugin+tg+tf+['--gdn-tensor-chunk','32'],2048,24576),
            ('diagnostic-tilelang',base+shared+plugin+tg+tf+['--profile-stages','--profile-kernels'],512,24576),
        ]
    return rows


def tensor_profiles(frontend,gdn):
    base=frontend+gdn+['--bf16-tp-reduce']
    tensor=['--gdn-tensor-prefill']
    prepare=['--fused-gdn-prepare']
    graph=['--mtp-verify-graph','--reuse-verify-graph']
    shared=['--multi-token-gemv-fp8']
    return [
        ('reference',base,512,24576),
        ('tensor-32',base+tensor+['--gdn-tensor-chunk','32'],512,24576),
        ('tensor-64',base+tensor+['--gdn-tensor-chunk','64'],512,24576),
        ('tensor-prepare',base+tensor+prepare,512,24576),
        ('combined-32',base+tensor+prepare+graph+shared+['--gdn-tensor-chunk','32'],512,24576),
        ('combined-64',base+tensor+prepare+graph+shared,512,24576),
        ('combined-2048',base+tensor+prepare+graph+shared,2048,24576),
        ('diagnostic-tensor',base+tensor+prepare+['--profile-stages','--profile-kernels'],512,24576),
    ]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('previous', 'hf-model', 'vllm-results', 'out'):
        p.add_argument('--'+key, type=Path, required=True)
    p.add_argument('--requests', type=int, default=5)
    p.add_argument('--timeout', type=float, default=1800)
    p.add_argument('--max-pixels', type=int, default=4000000)
    p.add_argument('--suite', choices=['prefill','deep','graph-wy','native-fused','gdn-tensor','ampere-tiled'], default='prefill')
    p.add_argument('--tilelang-dir',type=Path)
    a = p.parse_args()
    if a.requests < 1 or a.timeout <= 0:
        p.error('Invalid limits')
    a.out.mkdir(parents=True, exist_ok=False)
    request = json.loads((a.previous/'request/request.json').read_text(encoding='utf-8'))
    upstream = {n: json.loads((a.vllm_results/(n+'.json')).read_text(encoding='utf-8'))
                for n in ('mtp2', 'mtp3')}
    upstream_status={}
    audit=a.vllm_results/'summary.json'
    if audit.exists():
        upstream_status={r['name']:r['status'] for r in json.loads(audit.read_text(encoding='utf-8'))}
    common = ['--tp-lm-head', '--vector-gemv', '--extra-fusions', '--cublas-prefill',
              '--mtp-tokens', '3', '--mtp-draft-graph']
    profiles = [
        ('reference', [], 512, 24576),
        ('flash', ['--flash-prefill'], 512, 24576),
        ('flash-vision', ['--flash-prefill', '--cache-vision-weights'], 512, 24576),
        ('flash-vision-1024', ['--flash-prefill', '--cache-vision-weights'], 1024, 24576),
        ('flash-vision-2048', ['--flash-prefill', '--cache-vision-weights'], 2048, 24576),
        ('flash-vision-cache32', ['--flash-prefill', '--cache-vision-weights'], 512, 32768),
        ('diagnostic', ['--flash-prefill', '--cache-vision-weights', '--profile-stages'], 512, 24576),
    ]
    if a.suite in ('deep','graph-wy','native-fused','gdn-tensor','ampere-tiled'):
        common+=['--flash-prefill','--frontend-format','vllm-string']
        frontend=['--frontend-threads','4','--bf16-patches','--spool-dir','/dev/shm']
        gdn=['--gdn-cooperative','--fused-gdn-conv']
        profiles=[
            ('reference',[],512,24576),
            ('frontend',frontend,512,24576),
            ('gdn-cooperative',['--gdn-cooperative'],512,24576),
            ('gdn-conv',['--fused-gdn-conv'],512,24576),
            ('tp-bf16',['--bf16-tp-reduce'],512,24576),
            ('combined',frontend+gdn+['--bf16-tp-reduce'],512,24576),
            ('combined-2048',frontend+gdn+['--bf16-tp-reduce'],2048,24576),
            ('combined-mtp2',frontend+gdn+['--bf16-tp-reduce','--mtp-tokens','2'],512,24576),
            ('combined-c2',frontend+gdn+['--bf16-tp-reduce','--concurrency','2'],512,24576),
            ('diagnostic',frontend+gdn+['--bf16-tp-reduce','--profile-stages'],512,24576),
        ]
    if a.suite=='graph-wy':
        tuned=frontend+gdn+['--bf16-tp-reduce']
        graph=['--mtp-verify-graph']
        wy=['--gdn-wy']
        profiles=[
            ('reference',tuned,512,24576),
            ('verify-graph',tuned+graph,512,24576),
            ('wy',tuned+wy,512,24576),
            ('combined',tuned+graph+wy,512,24576),
            ('combined-2048',tuned+graph+wy,2048,24576),
            ('verify-graph-mtp2',tuned+graph+['--mtp-tokens','2'],512,24576),
            ('verify-graph-c2',tuned+graph+['--concurrency','2'],512,24576),
            ('diagnostic',tuned+['--profile-stages','--profile-kernels'],512,24576),
            ('diagnostic-wy',tuned+wy+['--profile-stages','--profile-kernels'],512,24576),
        ]
    if a.suite=='native-fused':
        profiles=fused_profiles(frontend,gdn)
    if a.suite=='gdn-tensor':
        profiles=tensor_profiles(frontend,gdn)
    if a.suite=='ampere-tiled':
        profiles=tiled_profiles(frontend,gdn,a.tilelang_dir)
    summary = {'suite':a.suite,'profiles': [], 'comparison': [], 'vllm_reports': str(a.vllm_results.resolve()),
               'note': 'vLLM reports are supplied separately (the deep test script generates them in this run). Measurements are sequential, not interleaved. Diagnostic synchronizes GPU and is excluded from speed comparisons.'}
    reference = None
    for name, flags, chunk, cache in profiles:
        output = a.out/(name+'.json')
        command = [sys.executable, 'tools/benchmark_worker.py',
                   '--model', str(a.previous/'fp8-mtp-tp2'), '--request', str(a.previous/'request'),
                   '--frontend-model', str(a.hf_model), '--body', str(a.previous/'request.json'),
                   '--out', str(output), '--requests', str(1 if name.startswith('diagnostic') else a.requests),
                   '--prefill-chunk', str(chunk), '--weight-cache-mib', str(cache),
                   '--max-pixels', str(a.max_pixels), '--timeout', str(a.timeout),
                   '--inherit-process-group']+common+flags
        code = run(command, a.out/(name+'.log'), a.timeout)
        row = dict(name=name, flags=flags, prefill_chunk=chunk, exit_code=code, status='failed', log=str(a.out/(name+'.log')))
        if code == 0:
            report = json.loads(output.read_text(encoding='utf-8'))
            if name == 'reference':
                reference = report['results'][0]['generated_ids']
            differences = compare_outputs(report['results'], reference, request['eos_token_ids']) if reference is not None else None
            row.update(status='passed' if differences == [] else 'output_mismatch',
                       concurrency=report['concurrency'],output_tokens_per_second=report['output_tokens_per_second'],
                       differences=differences, latency=report['latency_seconds'], ttft=report['ttft_seconds'],
                       frontend_seconds=[r.get('frontend_seconds') for r in report['results']],
                       stages=[r.get('stages') for r in report['results']],
                       kernel_times=[r.get('kernel_times') for r in report['results']],
                       graph_counters=[r.get('cache') for r in report['results']])
            for flag,key in (('--gdn-fused-solve','gdn_fused_calls'),('--gdn-tilelang','gdn_tilelang_calls'),
                             ('--fp8-tensor-small','fp8_tensor_calls'),('--tilelang-fp8','tilelang_fp8_calls')):
                if flag in flags and not counter_exercised(report,key):
                    row.update(status=key+'_not_exercised')
            if '--gdn-tensor-prefill' in flags and not tensor_exercised(report):
                row.update(status='tensor_gdn_not_exercised')
            if '--mtp-verify-graph' in flags and not any((r.get('cache') or {}).get('verify_graph_replays',0)>0 for r in report['results']):
                row.update(status='graph_not_exercised')
            if '--reuse-verify-graph' in flags and not any((r.get('cache') or {}).get('verify_graph_reuses',0)>0 for r in report['results']):
                row.update(status='graph_reuse_not_exercised')
            if not report.get('synchronized_diagnostic',False) and report['concurrency']==1:
                for label, target in upstream.items():
                    prompt = prompt_difference(report['prompt_token_ids'], target['results'][0]['prompt_token_ids'])
                    outputs = compare_outputs(report['results'], target['results'][0]['generated_ids'], request['eos_token_ids'])
                    stable = all(r['prompt_token_ids'] == target['results'][0]['prompt_token_ids'] and
                                 r['generated_ids'] == target['results'][0]['generated_ids'] for r in target['results'])
                    baseline_valid=upstream_status.get(label)=='passed'
                    valid = baseline_valid and prompt is None and not outputs and stable and row['status'] == 'passed'
                    summary['comparison'].append(dict(native=name, vllm=label, prompt_difference=prompt,
                        output_differences=outputs, vllm_repeatable=stable, vllm_baseline_valid=baseline_valid,
                        latency_ratio_vllm_over_native=target['latency_p50_seconds']/report['latency_seconds']['p50'] if valid else None))
        summary['profiles'].append(row)
        (a.out/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
        print(json.dumps(row), flush=True)
    ratios=[r['latency_ratio_vllm_over_native'] for r in summary['comparison'] if r['latency_ratio_vllm_over_native'] is not None]
    summary['acceptance']=acceptance(summary['profiles'],summary['comparison'],len(upstream))
    summary['failure_reasons']={'native_profiles':[r['name'] for r in summary['profiles'] if r['status']!='passed'],
                                'no_matching_cross_engine_comparison':not bool(ratios)}
    (a.out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary['acceptance']),flush=True)
    if any(r['status'] != 'passed' for r in summary['profiles']) or (a.suite in ('deep','graph-wy','native-fused','gdn-tensor','ampere-tiled') and not ratios):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
