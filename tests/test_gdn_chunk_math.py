"""Independent CPU checks of chunk-WY algebra; GPU qualification is in CTest."""
import unittest
import numpy as np


def recurrent(q, k, v, g, beta, initial):
    state = initial.copy()
    out = []
    groups = v.shape[1] // q.shape[1]
    for t in range(len(q)):
        state *= np.exp(g[t])[:, None, None]
        key = np.repeat(k[t], groups, axis=0)
        query = np.repeat(q[t], groups, axis=0)
        memory = np.einsum('hkv,hk->hv', state, key)
        delta = beta[t, :, None] * (v[t] - memory)
        state += key[:, :, None] * delta[:, None, :]
        out.append(np.einsum('hkv,hk->hv', state, query) / np.sqrt(q.shape[-1]))
    return np.array(out), state


def chunk_wy(q, k, v, g, beta, initial, chunk):
    state = initial.copy()
    out = []
    groups = v.shape[1] // q.shape[1]
    for start in range(0, len(q), chunk):
        n = min(chunk, len(q) - start)
        for h in range(v.shape[1]):
            hq = h // groups
            Q, K = q[start:start+n, hq], k[start:start+n, hq]
            V = v[start:start+n, h]
            G = g[start:start+n, h].cumsum()
            B = beta[start:start+n, h]
            lower_mask = np.tri(n, dtype=bool)
            diff = G[:, None] - G[None, :]
            # Avoid exponentiating masked positive upper-triangle differences.
            decay = np.exp(np.where(lower_mask, diff, 0)) * lower_mask
            L = np.eye(n) + np.tril((K @ K.T) * decay * B[:, None], -1)
            rhs = np.concatenate((B[:, None] * np.exp(G[:, None]) * K, B[:, None] * V), axis=1)
            solved = np.linalg.solve(L, rhs)
            W, U = solved[:, :K.shape[1]], solved[:, K.shape[1]:]
            update = U - W @ state[h]
            y = ((Q * np.exp(G[:, None])) @ state[h] + ((Q @ K.T) * decay) @ update) / np.sqrt(K.shape[1])
            if h == 0:
                block = np.empty((n, v.shape[1], v.shape[2]))
            block[:, h] = y
            state[h] = np.exp(G[-1]) * state[h] + (K * np.exp(G[-1] - G)[:, None]).T @ update
        out.extend(block)
    return np.array(out), state


def tf32_truncate(x):
    x = np.asarray(x, dtype=np.float32)
    return (x.view(np.uint32) & np.uint32(0xffffe000)).view(np.float32)


class ChunkMathTests(unittest.TestCase):
    def inputs(self, tokens=65, decay=.05):
        rng = np.random.default_rng(314)
        q, k = rng.normal(size=(2, tokens, 2, 16))
        q /= np.linalg.norm(q, axis=-1, keepdims=True)
        k /= np.linalg.norm(k, axis=-1, keepdims=True)
        v = rng.normal(size=(tokens, 6, 17))
        g = -rng.random((tokens, 6)) * decay
        beta = rng.random((tokens, 6))
        initial = rng.normal(size=(6, 16, 17))
        return q, k, v, g, beta, initial

    def test_grouped_heads_nonzero_state_partial_chunks(self):
        for T in (1, 31, 33, 65, 129):
            args = self.inputs(T)
            expected = recurrent(*args)
            for C in (32, 64):
                actual = chunk_wy(*args, C)
                for a, b in zip(actual, expected):
                    np.testing.assert_allclose(a, b, rtol=1e-10, atol=1e-10)

    def test_zero_and_strong_decay(self):
        for decay in (0, 30):
            args = self.inputs(65, decay)
            for a, b in zip(chunk_wy(*args, 64), recurrent(*args)):
                np.testing.assert_allclose(a, b, rtol=1e-10, atol=1e-10)

    def test_continuation_with_different_chunk_boundaries(self):
        q, k, v, g, beta, initial = self.inputs(129)
        first, state = chunk_wy(q[:37], k[:37], v[:37], g[:37], beta[:37], initial, 64)
        second, state = chunk_wy(q[37:], k[37:], v[37:], g[37:], beta[37:], state, 32)
        expected, expected_state = recurrent(q, k, v, g, beta, initial)
        np.testing.assert_allclose(np.concatenate((first, second)), expected, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(state, expected_state, rtol=1e-10, atol=1e-10)

    def test_split_tensor_products_against_fp64(self):
        rng = np.random.default_rng(711)
        for scale in (.001, 1, 100):
            a = (rng.normal(size=(16, 128)) * scale).astype(np.float32)
            b = rng.normal(size=(128, 16)).astype(np.float32)
            ah, bh = tf32_truncate(a), tf32_truncate(b)
            al, bl = tf32_truncate(a-ah), tf32_truncate(b-bh)
            result = al @ bl + al @ bh + ah @ bl + ah @ bh
            expected = a.astype(np.float64) @ b.astype(np.float64)
            # This validates splitting error only, not CUDA MMA accumulation.
            np.testing.assert_allclose(result, expected, rtol=2e-5, atol=scale*4e-6)


if __name__ == '__main__':
    unittest.main()
