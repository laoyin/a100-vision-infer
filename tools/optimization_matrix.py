"""Run isolated accuracy checks and resident benchmarks on existing FP8 artifacts."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def profiles(suite='standard'):
    if suite == 'deep':
        fast = ['--extra-fusions', '--cublas-prefill']
        return [
            ('baseline', ['--baseline'], 'baseline', 1, 128),
            ('cublas128-diagnostic', ['--cublas-prefill'], 'optimized', 1, 128),
            ('reference-prefill', ['--reference-prefill'], 'optimized', 1, 128),
            ('combined-chunk512', fast, 'optimized', 1, 512),
            ('tp-head', fast + ['--tp-lm-head'], 'optimized', 1, 512),
            ('vector-gemv', fast + ['--vector-gemv'], 'optimized', 1, 512),
            ('deep-combined', fast + ['--tp-lm-head', '--vector-gemv'], 'optimized', 1, 512),
            ('deep-graph', fast + ['--tp-lm-head', '--vector-gemv', '--cuda-graph'], 'graph', 1, 512),
            ('deep-c2', fast + ['--tp-lm-head', '--vector-gemv'], 'optimized', 2, 512),
            ('deep-chunk1024', fast + ['--tp-lm-head', '--vector-gemv'], 'optimized', 1, 1024),
        ]
    return [
        ('baseline', ['--baseline'], 'baseline', 1, 128),
        ('optimized', [], 'optimized', 1, 128),
        ('fusions', ['--extra-fusions'], 'optimized', 1, 128),
        ('cublas', ['--cublas-prefill'], 'optimized', 1, 128),
        ('combined', ['--extra-fusions', '--cublas-prefill'], 'optimized', 1, 128),
        ('combined-graph', ['--extra-fusions', '--cublas-prefill', '--cuda-graph'], 'graph', 1, 128),
        ('combined-chunk256', ['--extra-fusions', '--cublas-prefill'], 'optimized', 1, 256),
        ('combined-chunk512', ['--extra-fusions', '--cublas-prefill'], 'optimized', 1, 512),
        ('combined-c2', ['--extra-fusions', '--cublas-prefill'], 'optimized', 2, 128),
    ]


def execute(command, log, timeout):
    print('+ ' + ' '.join(map(str, command)), flush=True)
    with open(log, 'x', encoding='utf-8') as stream:
        process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = process.wait(timeout=timeout)
            if code:
                raise RuntimeError(f'exit {code}; see {log}')
        except BaseException:
            # Includes nested benchmark MPI children sharing this process group.
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass
            # The group leader can exit before its MPI ranks; reap the entire group.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
            raise


def ranking(rows):
    # Numerical agreement is not business acceptance; report token changes separately.
    good = [r for r in rows if r.get('status') == 'passed' and r.get('successful', 0) > 0]
    return [r['name'] for r in sorted(good, key=lambda r: r['output_tokens_per_second'], reverse=True)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--request', required=True)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--requests', type=int, default=3)
    parser.add_argument('--timeout', type=float, default=1800)
    parser.add_argument('--suite', choices=['standard', 'deep'], default='standard')
    args = parser.parse_args()
    if args.requests < 2 or args.timeout <= 0:
        parser.error('requests >= 2 and timeout > 0 required')
    args.out.mkdir(exist_ok=False)
    rows = []
    baseline = args.out / 'baseline.json'
    selected = profiles(args.suite)
    for name, flags, mode, concurrency, chunk in selected:
        print(f'[{len(rows)+1}/{len(selected)}] {name}', flush=True)
        row = {'name': name, 'flags': flags, 'concurrency': concurrency, 'prefill_chunk': chunk}
        start = time.monotonic()
        try:
            row['stage'] = 'trace'
            result = args.out / (name + '.json')
            diagnostic = args.suite == 'deep' and name in ('baseline', 'cublas128-diagnostic', 'reference-prefill', 'combined-chunk512')
            execute(['mpirun', '-np', '2', 'build/avi-infer', '--model', args.model,
                     '--request', args.request, '--output', str(result), '--prefill-chunk', str(chunk),
                     '--trace', *(['--layer-trace'] if diagnostic else []), *flags], args.out / (name + '-trace.log'), args.timeout)
            if diagnostic and name != 'baseline':
                execute([sys.executable, 'tools/compare_layers.py', '--baseline', str(baseline), '--candidate', str(result)],
                        args.out / (name + '-layers.json'), args.timeout)
            if name != 'baseline':
                row['stage'] = 'comparison'
                comparison = args.out / (name + '-comparison.json')
                try:
                    execute([sys.executable, 'tools/compare_native.py', '--baseline', str(baseline),
                             '--candidate', str(result)], comparison, args.timeout)
                finally:
                    if comparison.exists():
                        try: row['trace_comparison'] = json.loads(comparison.read_text())
                        except ValueError: pass
            row['stage'] = 'benchmark'
            bench = args.out / (name + '-bench.json')
            # Baseline and Graph switches are represented by --mode in this tool.
            options = [f for f in flags if f not in ('--baseline', '--cuda-graph')]
            execute([sys.executable, 'tools/benchmark_worker.py', '--model', args.model,
                     '--request', args.request, '--out', str(bench), '--mode', mode,
                     '--concurrency', str(concurrency), '--prefill-chunk', str(chunk),
                     '--requests', str(args.requests), '--warmup', str(concurrency),
                     '--timeout', str(args.timeout), '--inherit-process-group', *options],
                    args.out / (name + '-bench.log'), args.timeout * 3 + 60)
            report = json.loads(bench.read_text())
            reference = json.loads(baseline.read_text())['generated_ids']
            row.update({k: v for k, v in report.items() if k != 'results'})
            row['tokens_equal_to_baseline'] = all(r['generated_ids'] == reference for r in report['results'])
            row['status'] = 'passed'
            row['stage'] = 'complete'
        except Exception as error:
            row.update(status='failed', error=str(error))
            print(f'{name}: FAILED: {error}', flush=True)
        row['test_wall_seconds'] = time.monotonic() - start
        rows.append(row)
        summary = {'profiles': rows, 'throughput_ranking': ranking(rows),
                   'note': 'Cache disabled. Trace runs excluded from timing. Numerical pass does not imply identical tokens or business accuracy. Compare c1 and c2 throughput separately.'}
        (args.out / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
        if name == 'baseline' and row['status'] != 'passed':
            break  # Without a valid baseline subsequent comparisons are meaningless.
    print(json.dumps(summary, indent=2), flush=True)
    if len(rows) != len(selected) or any(r['status'] != 'passed' for r in rows):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
