"""Compile, qualify and tune SM80 kernels, then export standalone CUDA .so files.

Requires an existing TileLang >=0.1.15 environment. No package installation.
Cython wrappers expose only raw tensor pointers, int32 dimensions and a CUDA
stream. The resident C++ worker loads those wrappers without Python/TileLang.
"""
import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import shutil
import statistics
import subprocess
import time
import traceback

from tilelang_export_utils import choose_candidate, validate_call_abi
from tilelang_kernels import gdn_prepare, fp8_small


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2), encoding='utf-8')


def raw_library(path, pointer_count, scalar_count):
    lib = ctypes.CDLL(str(path.resolve()))
    lib.init.restype = ctypes.c_int
    lib.get_last_error.restype = ctypes.c_char_p
    if lib.init():
        raise RuntimeError(lib.get_last_error().decode())
    lib.call.restype = ctypes.c_int
    lib.call.argtypes = ([ctypes.c_void_p] * pointer_count + [ctypes.c_int] * scalar_count
                         + [ctypes.c_void_p])
    def invoke(tensors, scalars):
        import torch
        result = lib.call(*[ctypes.c_void_p(x.data_ptr()) for x in tensors], *scalars,
                          ctypes.c_void_p(torch.cuda.current_stream().cuda_stream))
        if result:
            raise RuntimeError(lib.get_last_error().decode())
    return lib, invoke


def compile_export(func, directory, name, kind):
    import tilelang
    kernel = tilelang.compile(func, target={'kind': 'cuda', 'arch': 'sm_80'},
                              execution_backend='cython')
    source = kernel.get_host_source()
    (directory/(name+'.cu')).write_text(source, encoding='utf-8')
    (directory/(name+'.device.cu')).write_text(kernel.get_kernel_source(), encoding='utf-8')
    abi = validate_call_abi(source, kind)
    # export_library() exports the TVM runtime ABI in some releases. The Cython
    # adapter's libpath is the standalone library exposing call/init/error.
    original = Path(kernel.adapter.libpath)
    target = directory/(name+'.so')
    shutil.copyfile(original, target)
    result = subprocess.run(['ldd', str(target)], capture_output=True, text=True)
    (directory/(name+'.ldd.txt')).write_text(result.stdout+result.stderr, encoding='utf-8')
    if result.returncode or 'not found' in result.stdout:
        raise RuntimeError('Exported library has unresolved runtime dependencies')
    if any(word in result.stdout.lower() for word in ('libpython', 'libtvm', 'libtilelang', 'libtorch')):
        raise RuntimeError('Exported module unexpectedly requires Python/TVM/Torch runtime')
    return target, abi


def timing(invoke, tensors, scalars, repeats):
    import torch
    for _ in range(3):
        invoke(tensors, scalars)
    torch.cuda.synchronize()
    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    values = []
    # Multiple launches per interval amortize Python submission gaps. The
    # report names this an event interval, not a pure GPU instruction time.
    for _ in range(repeats):
        begin.record()
        for _ in range(10):
            invoke(tensors, scalars)
        end.record()
        end.synchronize()
        values.append(begin.elapsed_time(end)/10)
    return statistics.median(values)


def gdn_data(C, HQ, H, blocks, decay=.02, partial=False):
    import torch
    q = torch.randn(blocks, HQ, C, 128, device='cuda')
    k = torch.randn_like(q)
    q = (q/q.norm(dim=-1, keepdim=True)).bfloat16()
    k = (k/k.norm(dim=-1, keepdim=True)).bfloat16()
    v = torch.randn(blocks, H, C, 128, device='cuda').bfloat16()
    g = -torch.rand(blocks, H, C, device='cuda')*decay
    b = torch.rand_like(g)
    if partial:
        q[-1, :, -7:] = 0
        k[-1, :, -7:] = 0
        v[-1, :, -7:] = 0
        g[-1, :, -7:] = 0
        b[-1, :, -7:] = 0
    g = g.cumsum(-1)
    shape = (blocks, H, C, 128)
    outputs = [torch.empty(blocks, H, C, C, device='cuda'),
               *[torch.empty(shape, device='cuda') for _ in range(4)],
               torch.empty(blocks, H, device='cuda')]
    return [q, k, v, g, b, *outputs]


def gdn_reference(tensors):
    import torch
    q, k, v, g, b = [x.cpu().double() for x in tensors[:5]]
    groups = v.shape[1]//q.shape[1]
    q, k = q.repeat_interleave(groups, 1), k.repeat_interleave(groups, 1)
    C = q.shape[2]
    mask = torch.ones(C, C, dtype=torch.bool).tril()
    diff = g.unsqueeze(-1)-g.unsqueeze(-2)
    decay = torch.exp(torch.where(mask, diff, 0))*mask
    lower = torch.eye(C)+(k@k.transpose(-1, -2)*decay*b.unsqueeze(-1)).tril(-1)
    solved = torch.linalg.solve_triangular(lower,
        torch.cat((b.unsqueeze(-1)*g.exp().unsqueeze(-1)*k, b.unsqueeze(-1)*v), -1),
        upper=False, unitriangular=True)
    return [q@k.transpose(-1, -2)*decay, solved[..., :128], solved[..., 128:],
            q*g.exp().unsqueeze(-1), k*(g[..., -1:]-g).exp().unsqueeze(-1), g[..., -1]]


def qualify_gdn(invoke, C, HQ, H):
    import torch
    maximum = 0.
    # Dynamic block count, partial tail and underflow/zero decay.
    for blocks, decay, partial in ((1, .02, False), (3, 0., True), (3, 30., True)):
        tensors = gdn_data(C, HQ, H, blocks, decay, partial)
        expected = gdn_reference(tensors)
        for out in tensors[5:]:
            out.fill_(float('nan'))
        invoke(tensors, [blocks])
        torch.cuda.synchronize()
        for out, ref in zip(tensors[5:], expected):
            torch.testing.assert_close(out.cpu().double(), ref, rtol=2e-4, atol=2e-6)
            maximum = max(maximum, (out.cpu().double()-ref).abs().max().item())
    return maximum


def fp8_data(M, N, K, split):
    import torch
    x = (torch.randn(M, K, device='cuda')*.1).bfloat16()
    # Independent lookup reference covers finite subnormal, signed zero and
    # exponent=15 codes. The invalid 0x7f/0xff codes are not model weights.
    codes = torch.randint(0, 256, (N, K), device='cuda', dtype=torch.int32).to(torch.uint8)
    codes[(codes & 127) == 127] = 0
    scales = torch.rand(N, K//128, device='cuda')*.001 + .0001
    partial = torch.empty(split, M, N, device='cuda')
    return [x, codes, scales, partial]


def fp8_reference(tensors):
    import torch
    x, codes, scales = tensors[:3]
    decoded = codes.view(torch.float8_e4m3fn).float()*scales.repeat_interleave(128, 1)
    weight = decoded.bfloat16()
    # FP64 matmul is a separate oracle, not the tiled GPU algorithm.
    return x.cpu().double()@weight.cpu().double().T


def qualify_fp8(invoke, M, split):
    import torch
    maximum = 0.
    for N, K in ((80, 128), (144, 384), (128, 512), (256, 5120)):
        tensors = fp8_data(M, N, K, split)
        ref = fp8_reference(tensors)
        tensors[-1].fill_(float('nan'))
        invoke(tensors, [K, N])
        torch.cuda.synchronize()
        out = tensors[-1].sum(0).bfloat16().cpu().double()
        torch.testing.assert_close(out, ref.bfloat16().double(), rtol=.01, atol=.001)
        maximum = max(maximum, (out-ref).abs().max().item())
    return maximum


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--repeats', type=int, default=7)
    parser.add_argument('--quick', action='store_true', help='One compiler configuration per specialization; still qualify')
    args = parser.parse_args()
    if not 3 <= args.repeats <= 100:
        parser.error('--repeats must be 3..100')
    import torch
    import tilelang
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (8, 0):
        raise RuntimeError('Requires SM80 GPU; expose test GPUs before compiling')
    torch.manual_seed(314)
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out/'INCOMPLETE').write_text('Qualification/tuning incomplete', encoding='utf-8')
    manifest = {'format': 'avi-tilelang-v1', 'arch': 'sm_80',
                'tilelang': tilelang.__version__, 'torch': torch.__version__,
                'cuda': torch.version.cuda, 'kernels': []}
    trials = []
    held_libraries = []
    def tune(kind, geometry, configs, make_func, qualify, cases):
        rows = []
        for index, config in enumerate(configs):
            name = kind+'-'+'-'.join(map(str, geometry.values()))+f'-c{index}'
            row = {'kind': kind, **geometry, 'name': name, 'config': config, 'validated': False}
            print(f'Compile/qualify {name}: {config}', flush=True)
            started = time.monotonic()
            try:
                file, abi = compile_export(make_func(config), args.out, name, kind)
                lib, invoke = raw_library(file, 11 if kind == 'gdn' else 4, 1 if kind == 'gdn' else 2)
                held_libraries.append(lib)
                error = qualify(invoke)
                measured = {label: timing(invoke, tensors, scalars, args.repeats)
                            for label, tensors, scalars in cases}
                ms = statistics.mean(measured.values())
                row.update(validated=True, max_abs_error=error, event_interval_ms=ms, shape_timings_ms=measured, file=file.name, abi=abi,
                           sha256=hashlib.sha256(file.read_bytes()).hexdigest())
            except Exception as exc:
                row.update(error=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc())
            row['compile_and_test_seconds'] = time.monotonic()-started
            rows.append(row)
            trials.append(row)
            write_json(args.out/'tuning.json', trials)
            print(json.dumps(row), flush=True)
        best = choose_candidate(rows)
        manifest['kernels'].append(best)
        write_json(args.out/'manifest.json', manifest)

    # HQ/H=8/24 production TP2, 1/3 oracle tests, 1/2 and 2/4 TP2/TP1 smoke.
    for HQ, H in ((8, 24), (1, 3), (1, 2), (2, 4)):
        for C in (32, 64):
            configs = [{'threads': n} for n in ((128,) if args.quick else (128, 256))]
            cases = [(f'tokens-{tokens}', gdn_data(C, HQ, H, tokens//C), [tokens//C])
                     for tokens in (512, 2048)]
            tune('gdn', dict(chunk=C, key_heads=HQ, heads=H), configs,
                 lambda cfg: gdn_prepare(C, HQ, H, **cfg),
                 lambda call: qualify_gdn(call, C, HQ, H), cases)

    # Dynamic N/K preserve all fixed-model projections, including fused QKV and
    # gate-up. Rank-local scales are the importer's row-expanded F32 layout.
    for M in range(2, 9):
        for split in (1, 4):
            configs = [{'block_n': 64, 'threads': 128, 'stages': 1}]
            if not args.quick:
                configs += [{'block_n': 64, 'threads': 128, 'stages': 2},
                            {'block_n': 128, 'threads': 256, 'stages': 2},
                            {'block_n': 128, 'threads': 256, 'stages': 3}]
            cases = [(f'N{N}-K{K}', fp8_data(M, N, K, split), [K, N])
                     for N, K in ((17408, 5120), (5120, 8704))]
            tune('fp8', dict(rows=M, split=split), configs,
                 lambda cfg: fp8_small(M, split, **cfg),
                 lambda call: qualify_fp8(call, M, split), cases)
    (args.out/'INCOMPLETE').unlink()
    print(f'Exported {len(manifest["kernels"])} validated specializations: {args.out}', flush=True)


if __name__ == '__main__':
    main()
