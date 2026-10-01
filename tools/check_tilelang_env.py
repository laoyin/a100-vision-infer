"""Read-only TileLang preflight. Never install packages or change CUDA."""
import argparse
import importlib.util
import json
import re
import shutil
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    report = {'status': 'failed', 'python': sys.version, 'installation_performed': False}
    code = 1
    try:
        import torch
        report.update(torch=torch.__version__, torch_cuda=torch.version.cuda, nvcc=shutil.which('nvcc'))
        if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
            raise RuntimeError('Expose two test A100 GPUs with AVI_GPUS; do not use production GPUs')
        report['devices'] = [{'name': torch.cuda.get_device_name(i),
                              'capability': list(torch.cuda.get_device_capability(i))}
                             for i in range(torch.cuda.device_count())]
        if any(row['capability'] != [8, 0] for row in report['devices']):
            raise RuntimeError('This experiment compiles only for SM80')
        if importlib.util.find_spec('tilelang') is None:
            report.update(status='missing', reason='TileLang is absent; native CUDA optimizations remain testable')
            code = 3
        else:
            import tilelang
            version = getattr(tilelang, '__version__', '')
            report['tilelang'] = version
            numbers = tuple(int(n) for n in re.findall(r'\d+', version)[:3])
            if numbers < (0, 1, 15):
                raise RuntimeError('Standalone Cython export expects TileLang >=0.1.15; no packages will be upgraded')
            if not report['nvcc']:
                raise RuntimeError('The installed CUDA nvcc compiler is required')
            report['status'], code = 'ready', 0
    except Exception as exc:
        report['reason'] = f'{type(exc).__name__}: {exc}'
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
