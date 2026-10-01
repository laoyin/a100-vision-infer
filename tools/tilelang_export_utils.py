"""CPU-only ABI validation and candidate selection for standalone exports."""
import math
import re

GDN_ABI = 'Q,K,V,G,B,A,W,U,SQ,WK,last,blocks:i32,stream'
FP8_ABI = 'X,Codes,Scales,Partial,K:i32,N:i32,stream'


def validate_call_abi(source, kind):
    # Inspect the real Cython host wrapper, never infer a C ABI from a Python
    # function signature. Different scalar sizes/orders are hard failures.
    match = re.search(r'\bint\s+call\s*\(([^()]*)\)\s*\{', source)
    if not match:
        raise ValueError('No standalone extern-C call wrapper in generated source')
    pointers = [('Q', 'bfloat16'), ('K', 'bfloat16'), ('V', 'bfloat16'),
                ('G', 'float'), ('B', 'float'), ('A', 'float'), ('W', 'float'),
                ('U', 'float'), ('SQ', 'float'), ('WK', 'float'), ('last', 'float')] if kind == 'gdn' else [
                ('X', 'bfloat16'), ('Codes', 'uint8'), ('Scales', 'float'), ('Partial', 'float')]
    integers = ['blocks'] if kind == 'gdn' else ['K', 'N']
    fields = match.group(1).split(',')
    if len(fields) != len(pointers) + len(integers) + 1:
        raise ValueError(f'Unexpected exported argument count: {fields}')
    for field, (name, dtype) in zip(fields, pointers):
        clean = re.sub(r'\b(const|__restrict__|__restrict|restrict)\b', '', field).strip()
        expected_type = {'bfloat16': r'(?:tl::)?bfloat16_t|(?:__nv_)?bfloat16',
                         'uint8': r'uint8_t|unsigned\s+char', 'float': r'float'}[dtype]
        if not re.fullmatch(rf'(?:{expected_type})\s*\*\s*{name}', clean):
            raise ValueError(f'Unexpected {name} pointer ABI: {field}')
    for field, name in zip(fields[len(pointers):], integers):
        if not re.fullmatch(rf'\s*(?:int|int32_t)\s+{name}\s*', field):
            raise ValueError(f'Expected int32 scalar {name}, got {field}')
    if not re.fullmatch(r'\s*cudaStream_t\s+stream(?:\s*=\s*(?:cudaStreamDefault|0|nullptr))?\s*', fields[-1]):
        raise ValueError(f'Unexpected CUDA stream ABI: {fields[-1]}')
    return GDN_ABI if kind == 'gdn' else FP8_ABI


def choose_candidate(candidates):
    valid = [row for row in candidates if row.get('validated') and
             isinstance(row.get('event_interval_ms'), (int, float)) and
             math.isfinite(row['event_interval_ms']) and row['event_interval_ms'] > 0]
    if not valid:
        raise ValueError('No numerically valid tuning candidate; retain tuning.json and generated sources')
    return min(valid, key=lambda row: row['event_interval_ms'])


def split_ranges(k, split):
    if k <= 0 or k % 128 or split not in (1, 4):
        raise ValueError('K must be positive and divisible by 128; split must be 1 or 4')
    total = k // 128
    steps = (total + split - 1) // split
    return [(min(total, s*steps)*128, min(total, (s+1)*steps)*128) for s in range(split)]
