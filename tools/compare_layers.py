"""Compare final prefill and decode-74 layer traces; recurrent tensors are sampled."""
import argparse
import json
from pathlib import Path
import re
import numpy as np


def metrics(x, y):
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if x.shape != y.shape or not x.size or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('Invalid or nonfinite layer trace')
    nx, ny = np.linalg.norm(x), np.linalg.norm(y)
    return {'cosine': float(x @ y / (nx * ny)) if nx and ny else float(nx == ny),
            'relative_rmse': float(np.linalg.norm(x-y) / max(nx, 1e-30)),
            'max_abs_error': float(np.max(np.abs(x-y)))}


def compare(baseline, candidate):
    baseline, candidate = Path(baseline), Path(candidate)
    b = json.loads(baseline.read_text())['generated_ids']
    c = json.loads(candidate.read_text())['generated_ids']
    records, skipped = [], []
    for phase in ('prefill_layers', 'decode74_layers'):
        if phase == 'decode74_layers' and (len(b) < 75 or len(c) < 75 or b[:74] != c[:74]):
            skipped.append('decode74: differing token histories or too few tokens')
            continue
        for source in baseline.parent.glob(baseline.name + '.' + phase + '.*.f32'):
            suffix = source.name[len(baseline.name):]
            target = Path(str(candidate) + suffix)
            if not target.exists():
                raise ValueError('Missing candidate trace: ' + str(target))
            match = re.search(r'\.rank(\d+)\.layer(\d+)\.(\w+)\.f32$', suffix)
            if not match:
                continue
            records.append({'phase': phase, 'rank': int(match[1]), 'layer': int(match[2]), 'kind': match[3],
                            **metrics(np.fromfile(source, '<f4'), np.fromfile(target, '<f4'))})
    if not records:
        raise ValueError('No comparable layer traces')
    records.sort(key=lambda r: (r['phase'], r['layer'], r['rank'], r['kind']))
    return {'records': records, 'skipped': skipped,
            'note': 'Diagnostic only. Hidden = final token of chunk; recurrent = evenly spaced sample of at most 4096 elements per layer/rank. Different chunks change accumulation order. No relaxed acceptance threshold.'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline', required=True)
    p.add_argument('--candidate', required=True)
    a = p.parse_args()
    print(json.dumps(compare(a.baseline, a.candidate), indent=2))


if __name__ == '__main__':
    main()
