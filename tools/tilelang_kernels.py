"""Original SM80 TileLang kernels; no Hopper/TMA/WGMMA instructions.

Algorithm references: Gated Delta Networks (arXiv:2412.06464), FLA chunk WY,
TileLang's tiled GEMM examples. No upstream kernel source is vendored.
Imports are lazy so CPU tooling and the ordinary native build need no TileLang.
"""


def gdn_prepare(chunk, key_heads, heads, threads=128):
    import tilelang.language as T
    if chunk not in (32, 64) or heads % key_heads or threads not in (128, 256):
        raise ValueError('Invalid GDN specialization')
    blocks = T.dynamic('blocks', 'int32')
    C, HQ, H = chunk, key_heads, heads

    @T.prim_func
    def prepare(Q: T.Tensor((blocks, HQ, C, 128), 'bfloat16'),
                K: T.Tensor((blocks, HQ, C, 128), 'bfloat16'),
                V: T.Tensor((blocks, H, C, 128), 'bfloat16'),
                G: T.Tensor((blocks, H, C), 'float32'),
                B: T.Tensor((blocks, H, C), 'float32'),
                A: T.Tensor((blocks, H, C, C), 'float32'),
                W: T.Tensor((blocks, H, C, 128), 'float32'),
                U: T.Tensor((blocks, H, C, 128), 'float32'),
                SQ: T.Tensor((blocks, H, C, 128), 'float32'),
                WK: T.Tensor((blocks, H, C, 128), 'float32'),
                last: T.Tensor((blocks, H), 'float32')):
        with T.Kernel(blocks, HQ, threads=threads) as (block, hq):
            qs = T.alloc_shared((C, 128), 'bfloat16')
            ks = T.alloc_shared((C, 128), 'bfloat16')
            kk = T.alloc_fragment((C, C), 'float32')
            qk = T.alloc_fragment((C, C), 'float32')
            lower = T.alloc_shared((C, C), 'float32')
            gs = T.alloc_shared((C,), 'float32')
            bs = T.alloc_shared((C,), 'float32')
            ws = T.alloc_shared((C, 128), 'float32')
            us = T.alloc_shared((C, 128), 'float32')
            rw = T.alloc_local((1,), 'float32')
            ru = T.alloc_local((1,), 'float32')
            T.copy(Q[block, hq, 0, 0], qs)
            T.copy(K[block, hq, 0, 0], ks)
            T.clear(kk)
            T.clear(qk)
            T.gemm(ks, ks, kk, transpose_B=True)
            T.gemm(qs, ks, qk, transpose_B=True)
            # Reuse both dot products across all grouped value heads.
            for group in T.serial(H // HQ):
                h = hq * (H // HQ) + group
                T.copy(G[block, h, 0], gs)
                T.copy(B[block, h, 0], bs)
                T.sync_threads()
                for r, c in T.Parallel(C, C):
                    lower[r, c] = T.if_then_else(c < r,
                        kk[r, c] * T.exp(gs[r] - gs[c]) * bs[r], 0.0)
                    A[block, h, r, c] = T.if_then_else(c <= r,
                        qk[r, c] * T.exp(gs[r] - gs[c]), 0.0)
                T.sync_threads()
                for r in T.serial(C):
                    for d in T.Parallel(128):
                        rw[0] = bs[r] * T.exp(gs[r]) * T.cast(ks[r, d], 'float32')
                        ru[0] = bs[r] * T.cast(V[block, h, r, d], 'float32')
                        for c in T.serial(r):
                            rw[0] = rw[0] - lower[r, c] * ws[c, d]
                            ru[0] = ru[0] - lower[r, c] * us[c, d]
                        ws[r, d] = rw[0]
                        us[r, d] = ru[0]
                        W[block, h, r, d] = rw[0]
                        U[block, h, r, d] = ru[0]
                        SQ[block, h, r, d] = T.cast(qs[r, d], 'float32') * T.exp(gs[r])
                        WK[block, h, r, d] = T.cast(ks[r, d], 'float32') * T.exp(gs[C-1] - gs[r])
                    T.sync_threads()
                for writer in T.Parallel(1):
                    last[block, h] = gs[C-1]
                T.sync_threads()
    return prepare


def fp8_small(rows, split=1, block_n=64, threads=128, stages=2):
    import tilelang.language as T
    if rows not in range(2, 9) or split not in (1, 4):
        raise ValueError('Invalid FP8 specialization')
    K, N = T.dynamic('K N', 'int32')
    M, BN = rows, block_n

    @T.prim_func
    def linear(X: T.Tensor((M, K), 'bfloat16'),
               Codes: T.Tensor((N, K), 'uint8'),
               Scales: T.Tensor((N, T.ceildiv(K, 128)), 'float32'),
               Partial: T.Tensor((split, M, N), 'float32')):
        with T.Kernel(T.ceildiv(N, BN), split, threads=threads) as (bx, part):
            xs = T.alloc_shared((16, 128), 'bfloat16')
            codes = T.alloc_shared((BN, 128), 'uint8')
            weights = T.alloc_shared((BN, 128), 'bfloat16')
            scales = T.alloc_shared((BN,), 'float32')
            accum = T.alloc_fragment((16, BN), 'float32')
            b = T.alloc_local((1,), 'int32')
            e = T.alloc_local((1,), 'int32')
            m = T.alloc_local((1,), 'int32')
            decoded = T.alloc_local((1,), 'float32')
            T.clear(accum)
            steps = T.ceildiv(T.ceildiv(K, 128), split)
            for step in T.Pipelined(steps, num_stages=stages):
                block = part * steps + step
                for i, d in T.Parallel(16, 128):
                    xs[i, d] = T.if_then_else(i < M and block*128+d < K,
                                              X[i, block*128+d], T.cast(0, "bfloat16"))
                for n, d in T.Parallel(BN, 128):
                    codes[n, d] = T.if_then_else(bx*BN+n < N and block*128+d < K,
                                                 Codes[bx*BN+n, block*128+d], 0)
                for n in T.Parallel(BN):
                    scales[n] = T.if_then_else(bx*BN+n < N and block*128 < K,
                                              Scales[bx*BN+n, block], 0.0)
                T.sync_threads()
                for n, d in T.Parallel(BN, 128):
                    b[0] = T.cast(codes[n, d], 'int32')
                    e[0] = (b[0] >> 3) & 15
                    m[0] = b[0] & 7
                    decoded[0] = T.if_then_else(e[0] == 0, T.cast(m[0], 'float32') / 512.0,
                        (1.0 + T.cast(m[0], 'float32') / 8.0) * T.exp2(T.cast(e[0]-7, 'float32')))
                    decoded[0] = T.if_then_else((b[0] & 128) != 0, -decoded[0], decoded[0])
                    # E4M3FN only 0x7f/0xff are NaNs; exponent=15 has finite values.
                    decoded[0] = T.if_then_else(e[0] == 15 and m[0] == 7,
                                                T.reinterpret('float32', T.uint32(2143289344)), decoded[0])
                    weights[n, d] = T.cast(decoded[0] * scales[n], 'bfloat16')
                T.gemm(xs, weights, accum, transpose_B=True)
            for i, n in T.Parallel(16, BN):
                if i < M and bx*BN+n < N:
                    Partial[part, i, bx*BN+n] = accum[i, n]
    return linear
