"""Independent double-precision check of the batched WY decomposition."""
import unittest
import numpy as np


class WYMathTests(unittest.TestCase):
    def test_partial_chunks_nonzero_state_and_strong_decay(self):
        rng=np.random.default_rng(31)
        for T in (1,31,32,33,65):
            for strength in (.1,20):
                K,V,C=8,5,32
                q=rng.normal(size=(T,K));k=rng.normal(size=(T,K))
                q/=np.linalg.norm(q,axis=-1,keepdims=True)
                k/=np.linalg.norm(k,axis=-1,keepdims=True)
                v=rng.normal(size=(T,V));g=-rng.random(T)*strength;b=rng.random(T)
                state=rng.normal(size=(K,V));reference=state.copy();expected=[]
                for t in range(T):
                    reference*=np.exp(g[t])
                    reference+=np.outer(k[t],b[t]*(v[t]-k[t]@reference))
                    expected.append(q[t]@reference/np.sqrt(K))
                actual=[]
                for start in range(0,T,C):
                    n=min(C,T-start)
                    Q=np.pad(q[start:start+n],((0,C-n),(0,0)))
                    keys=np.pad(k[start:start+n],((0,C-n),(0,0)))
                    values=np.pad(v[start:start+n],((0,C-n),(0,0)))
                    G=np.cumsum(np.pad(g[start:start+n],(0,C-n)))
                    B=np.pad(b[start:start+n],(0,C-n))[:,None]
                    mask=np.tri(C,dtype=bool)
                    decay=np.exp(np.where(mask,G[:,None]-G[None,:],0))*mask
                    L=np.eye(C)+np.tril((keys@keys.T)*decay*B,-1)
                    solved=np.linalg.solve(L,np.concatenate([B*np.exp(G[:,None])*keys,B*values],axis=-1))
                    update=solved[:,K:]-solved[:,:K]@state
                    y=(np.exp(G[:,None])*Q@state+((Q@keys.T)*decay)@update)/np.sqrt(K)
                    actual.extend(y[:n])
                    state=np.exp(G[-1])*state+(keys*np.exp(G[-1]-G)[:,None]).T@update
                np.testing.assert_allclose(actual,expected,atol=1e-12,rtol=1e-10)
                np.testing.assert_allclose(state,reference,atol=1e-12,rtol=1e-10)
