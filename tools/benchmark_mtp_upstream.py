"""Reuse installed vLLM for MTP A/B experiments. Does not implement native AVI speculation."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time
from workload_metrics import workload_metrics
try:
    from .inspect_mtp import inspect, prepare_trial_model
except ImportError:
    from inspect_mtp import inspect, prepare_trial_model


def engine_options(model, window, max_context, memory, max_pixels=4000000):
    options = dict(model=str(Path(model).resolve()), tensor_parallel_size=2, dtype='bfloat16',
                   max_model_len=max_context, gpu_memory_utilization=memory,
                   enable_prefix_caching=False, max_num_seqs=1, seed=42,
                   trust_remote_code=False, disable_log_stats=False,
                   mm_processor_kwargs={'max_pixels': max_pixels}, mm_processor_cache_gb=0)
    if window:
        options['speculative_config'] = {'method': 'mtp', 'num_speculative_tokens': window}
    return options


def run_profile(a):
    # Import only in the isolated child, after eligibility checks. No installs/downloads.
    from vllm import LLM, SamplingParams
    import torch
    if torch.cuda.device_count() < 2 or any(torch.cuda.get_device_capability(i) != (8, 0) for i in range(2)):
        raise ValueError('This TP2 experiment requires two visible A100/SM80 GPUs')
    body = json.loads(a.body.read_text(encoding='utf-8'))
    if not isinstance(body.get('messages'), list) or not body['messages']:
        raise ValueError('Expected a nonempty OpenAI messages list')
    if body.get('response_format') or body.get('tools'):
        raise ValueError('First trial uses plain greedy output; structured decoding/tool processing requires separate validation')
    params = SamplingParams(temperature=0, max_tokens=a.max_tokens, seed=42)
    llm = LLM(**engine_options(a.model, a.window, a.max_context, a.memory, a.max_pixels))
    def generate():
        start = time.perf_counter()
        result = llm.chat(body['messages'], sampling_params=params, use_tqdm=False,
                          chat_template_kwargs={'enable_thinking': False})
        duration = time.perf_counter()-start
        if len(result) != 1 or len(result[0].outputs) != 1:
            raise ValueError('Expected one output')
        r, out = result[0], result[0].outputs[0]
        if out.finish_reason not in ('stop', 'length'):
            raise ValueError('Unexpected finish: ' + str(out.finish_reason))
        return {'latency_seconds': duration, 'generated_ids': list(out.token_ids), 'text': out.text,
                'input_tokens': len(r.prompt_token_ids), 'prompt_token_ids': list(r.prompt_token_ids), 'finish_reason': out.finish_reason}
    def metrics():
        # Engine versions expose different metric objects; unavailable values stay unavailable.
        getter = getattr(llm, 'get_metrics', None)
        if getter is None:
            return None
        try:
            return {m.name: float(m.value) for m in getter()
                    if ('spec' in m.name or 'draft' in m.name) and isinstance(getattr(m, 'value', None), (float, int))}
        except Exception:
            return None
    generate()  # Warmup is excluded, caches disabled for every profile.
    before = metrics()
    start = time.perf_counter()
    results = [generate() for _ in range(a.requests)]
    elapsed = time.perf_counter()-start
    after = metrics()
    report = {'backend': 'vllm', 'version': importlib.metadata.version('vllm'), 'window': a.window,
              'successful': len(results), 'latency_p50_seconds': statistics.median(r['latency_seconds'] for r in results),
              'output_tokens_per_second': sum(len(r['generated_ids']) for r in results)/elapsed,
              'wall_seconds': elapsed, 'results': results,
              'speculative_metric_deltas': {k: v-before[k] for k, v in after.items() if k in before} if before is not None and after is not None else None,
              'note': 'Upstream vLLM inference including request preprocessing; excludes model load and warmup. Not comparable directly to native prepared-input timing. No TTFT estimate from non-streaming timing.'}
    report['workload']=workload_metrics(results)
    with a.out.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False)


def stop_group(process):
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pass


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True, type=Path)
    p.add_argument('--body', required=True, type=Path)
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--requests', type=int, default=3)
    p.add_argument('--max-tokens', type=int, default=128)
    p.add_argument('--max-context', type=int, default=20480)
    p.add_argument('--memory', type=float, default=.8)
    p.add_argument('--max-pixels', type=int, default=4000000)
    p.add_argument('--timeout', type=float, default=1800)
    p.add_argument('--window', type=int, default=0, choices=(0, 1, 2, 3))
    p.add_argument('--child', action='store_true', help=argparse.SUPPRESS)
    a = p.parse_args()
    if a.requests < 1 or a.max_tokens < 1 or a.max_pixels < 1 or a.max_context <= a.max_tokens or not 0 < a.memory <= .95 or a.timeout <= 0:
        p.error('Invalid experiment limits')
    if a.child:
        run_profile(a)
        return
    if os.name != 'posix':
        p.error('Run experiments on the Linux GPU server')
    a.out.mkdir(parents=True, exist_ok=False)
    audit = inspect(a.model)
    (a.out/'mtp-audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    if not audit['eligible_for_trial']:
        p.error('Checkpoint lacks validated dense MTP shapes; see mtp-audit.json. No model started.')
    try:
        version = importlib.metadata.version('vllm')
    except importlib.metadata.PackageNotFoundError:
        p.error('vLLM is absent from this environment. Nothing will be installed automatically.')
    if audit.get('unindexed_mtp_sidecar'):
        a.model = prepare_trial_model(a.model, a.out/'model-view')
        print(f'Created complete checkpoint index in {a.model}; source weights unchanged. Runtime compatibility still requires successful MTP trials.', flush=True)
    body = json.loads(a.body.read_text(encoding='utf-8'))
    for message in body.get('messages', []):
        content = message.get('content', [])
        if isinstance(content, list):
            for item in content:
                if item.get('type') == 'image_url' and not item['image_url']['url'].startswith('data:image/'):
                    p.error('Use embedded image data; remote image downloads are disabled for reproducible trials')
    env = dict(os.environ, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    # PYTHONPATH also reaches vLLM's spawned workers. No site-packages edits,
    # dependency installs, runtime replacement, or changes to production services.
    compat = str(Path(__file__).resolve().parent/'compat_flashinfer')
    env['AVI_FLASHINFER_RUNTIME_COMPAT'] = '1'
    env['PYTHONPATH'] = compat + (os.pathsep + env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
    print('Enabled process-local FlashInfer CUDA runtime lookup compatibility', flush=True)
    rows, reference, reference_prompt = [], None, None
    for window in (0, 1, 2, 3):
        name = 'baseline' if window == 0 else f'mtp{window}'
        output = a.out/(name+'.json')
        command = [sys.executable, str(Path(__file__).resolve()), '--child', '--model', str(a.model.resolve()),
                   '--body', str(a.body.resolve()), '--out', str(output.resolve()), '--window', str(window),
                   '--requests', str(a.requests), '--max-tokens', str(a.max_tokens), '--max-context', str(a.max_context), '--memory', str(a.memory), '--max-pixels', str(a.max_pixels)]
        row = {'name': name, 'vllm_version': version, 'cuda_runtime_lookup_compat': True}
        print(f'Starting {name}; log: {a.out/(name+".log")}', flush=True)
        with (a.out/(name+'.log')).open('x', encoding='utf-8') as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
            try:
                code = process.wait(timeout=a.timeout)
                if code:
                    raise RuntimeError(f'Process exited {code}; see {name}.log')
                result = json.loads(output.read_text(encoding='utf-8'))
                if window == 0:
                    reference = result['results'][0]['generated_ids']
                    reference_prompt = result['results'][0]['prompt_token_ids']
                row.update({k: v for k, v in result.items() if k != 'results'})
                row['tokens_equal_to_baseline'] = all(r['generated_ids'] == reference for r in result['results'])
                row['input_tokens_equal'] = all(r['prompt_token_ids'] == reference_prompt for r in result['results'])
                row['status'] = 'passed' if row['tokens_equal_to_baseline'] and row['input_tokens_equal'] else 'output_mismatch'
            except Exception as error:
                row.update(status='failed', error=str(error))
            finally:
                stop_group(process)
        rows.append(row)
        (a.out/'summary.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
        print(json.dumps(row), flush=True)
        if window == 0 and row['status'] != 'passed':
            break
    if len(rows) != 4 or any(r['status'] != 'passed' for r in rows):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
