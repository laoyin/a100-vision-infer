import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from format_utils import mtp_shapes, partition, decode_fp8
from fp8_import_utils import partition_fp8
from native_mtp_matrix import canonical,compare_outputs

class NativeMTPTests(unittest.TestCase):
    def test_mtp_geometry(self):
        t=dict(mtp_num_hidden_layers=1,hidden_size=5120,head_dim=256,
               intermediate_size=17408,num_attention_heads=24,num_key_value_heads=4)
        shapes=mtp_shapes({'text_config':t})
        self.assertEqual(len(shapes),15)
        self.assertEqual(shapes['mtp.layers.0.self_attn.q_proj.weight'],(12288,5120))
        self.assertEqual(shapes['mtp.fc.weight'],(5120,10240))
        t['mtp_use_dedicated_embeddings']=True
        with self.assertRaises(ValueError):mtp_shapes({'text_config':t})

    def test_mtp_tp_scales_preserve_codes(self):
        rng=np.random.default_rng(18)
        for suffix,n,k in [('mlp.gate_proj.weight',512,512),('mlp.down_proj.weight',256,512),
                           ('self_attn.q_proj.weight',1024,512),('self_attn.o_proj.weight',256,512)]:
            name='mtp.layers.0.'+suffix
            codes=rng.integers(0,127,(n,k),dtype=np.uint8)
            scales=rng.uniform(.01,.2,(n//128,k//128)).astype('float32')
            reference=decode_fp8(codes)*scales.repeat(128,0).repeat(128,1)
            for tp in (1,2,4):
                for rank in range(tp):
                    q,s=partition_fp8(name,codes,scales,{},rank,tp)
                    np.testing.assert_array_equal(decode_fp8(q)*s.repeat(128,1),partition(name,reference,{},rank,tp))
                    np.testing.assert_array_equal(q,partition(name,codes,{},rank,tp))

    def test_mtp_fc_replicated(self):
        matrix=np.arange(64).reshape(4,16)
        for rank in range(4):
            np.testing.assert_array_equal(partition('mtp.fc.weight',matrix,{},rank,4),matrix)

    def test_comparison_only_normalizes_terminal_eos(self):
        self.assertEqual(canonical([1,2,9],[9]),[1,2])
        self.assertEqual(canonical([1,9,2],[9]),[1,9,2])
        self.assertEqual(compare_outputs([{'generated_ids':[1,2,9]}],[1,2],[9]),[])
        diff=compare_outputs([{'generated_ids':[1,3]}],[1,2],[9])
        self.assertEqual(diff[0]['first_difference'],1)

    def test_block_delta_algebra_against_sequential_recurrence(self):
        # Independent FP64 derivation tests signs, beta placement and causal mask.
        rng=np.random.default_rng(12)
        for T in (1,7,32):
            K,V=8,5
            q=rng.normal(size=(T,K));k=rng.normal(size=(T,K))
            q/=np.linalg.norm(q,axis=1,keepdims=True);k/=np.linalg.norm(k,axis=1,keepdims=True)
            v=rng.normal(size=(T,V));g=-rng.random(T);beta=rng.random(T)
            initial=rng.normal(size=(K,V));state=initial.copy();outputs=[]
            for i in range(T):
                state*=np.exp(g[i])
                state+=np.outer(k[i],beta[i]*(v[i]-k[i]@state))
                outputs.append(q[i]@state/np.sqrt(K))
            G=np.cumsum(g);decay=np.exp(np.minimum(G[:,None]-G[None,:],0))*np.tri(T)
            L=np.eye(T)+np.tril((k@k.T)*decay*beta[:,None],-1)
            U=np.linalg.solve(L,beta[:,None]*(v-np.exp(G)[:,None]*(k@initial)))
            actual=(np.exp(G)[:,None]*(q@initial)+((q@k.T)*decay)@U)/np.sqrt(K)
            end=np.exp(G[-1])*initial+k.T@(np.exp(G[-1]-G)[:,None]*U)
            np.testing.assert_allclose(actual,outputs,rtol=1e-12,atol=1e-12)
            np.testing.assert_allclose(end,state,rtol=1e-12,atol=1e-12)
