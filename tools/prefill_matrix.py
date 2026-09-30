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


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('previous', 'hf-model', 'vllm-results', 'out'):
        p.add_argument('--'+key, type=Path, required=True)
    p.add_argument('--requests', type=int, default=5)
    p.add_argument('--timeout', type=float, default=1800)
    p.add_argument('--max-pixels', type=int, default=4000000)
    a = p.parse_args()
    if a.requests < 1 or a.timeout <= 0:
        p.error('Invalid limits')
    a.out.mkdir(parents=True, exist_ok=False)
    request = json.loads((a.previous/'request/request.json').read_text(encoding='utf-8'))
    upstream = {n: json.loads((a.vllm_results/(n+'.json')).read_text(encoding='utf-8'))
                for n in ('mtp2', 'mtp3')}
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
    summary = {'profiles': [], 'comparison': [], 'vllm_reports': str(a.vllm_results.resolve()),
               'note': 'vLLM timings reused from an earlier run; not a simultaneous hardware-controlled benchmark. Diagnostic profile synchronizes GPU and is excluded from speed comparisons.'}
    reference = None
    for name, flags, chunk, cache in profiles:
        output = a.out/(name+'.json')
        command = [sys.executable, 'tools/benchmark_worker.py',
                   '--model', str(a.previous/'fp8-mtp-tp2'), '--request', str(a.previous/'request'),
                   '--frontend-model', str(a.hf_model), '--body', str(a.previous/'request.json'),
                   '--out', str(output), '--requests', str(1 if name == 'diagnostic' else a.requests),
                   '--prefill-chunk', str(chunk), '--weight-cache-mib', str(cache),
                   '--max-pixels', str(a.max_pixels), '--timeout', str(a.timeout),
                   '--inherit-process-group']+common+flags
        code = run(command, a.out/(name+'.log'), a.timeout)
        row = dict(name=name, exit_code=code, status='failed')
        if code == 0:
            report = json.loads(output.read_text(encoding='utf-8'))
            if name == 'reference':
                reference = report['results'][0]['generated_ids']
            differences = compare_outputs(report['results'], reference, request['eos_token_ids']) if reference is not None else None
            row.update(status='passed' if differences == [] else 'output_mismatch',
                       differences=differences, latency=report['latency_seconds'], ttft=report['ttft_seconds'],
                       frontend_seconds=[r.get('frontend_seconds') for r in report['results']],
                       stages=[r.get('stages') for r in report['results']])
            if name != 'diagnostic':
                for label, target in upstream.items():
                    prompt = prompt_difference(report['prompt_token_ids'], target['results'][0]['prompt_token_ids'])
                    outputs = compare_outputs(report['results'], target['results'][0]['generated_ids'], request['eos_token_ids'])
                    stable = all(r['prompt_token_ids'] == target['results'][0]['prompt_token_ids'] and
                                 r['generated_ids'] == target['results'][0]['generated_ids'] for r in target['results'])
                    valid = prompt is None and not outputs and stable and row['status'] == 'passed'
                    summary['comparison'].append(dict(native=name, vllm=label, prompt_difference=prompt,
                        output_differences=outputs, vllm_repeatable=stable,
                        latency_ratio_vllm_over_native=target['latency_p50_seconds']/report['latency_seconds']['p50'] if valid else None))
        summary['profiles'].append(row)
        (a.out/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
        print(json.dumps(row), flush=True)
    if any(r['status'] != 'passed' for r in summary['profiles']):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
