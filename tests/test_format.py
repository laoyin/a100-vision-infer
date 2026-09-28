import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import unittest
import numpy as np
from format_utils import bf16_bytes, decode_fp8, quantize_fp8, partition, image_geometry, rope_positions

class FormatTests(unittest.TestCase):
    def test_fp8_values_and_roundtrip(self):
        np.testing.assert_array_equal(decode_fp8([0, 1, 56, 126, 184]), [0, 2**-9, 1, 448, -1])
        self.assertTrue(np.isnan(decode_fp8([127, 255])).all())
        rng = np.random.default_rng(42)
        x = rng.normal(size=(8, 64)).astype(np.float32)
        q, scales = quantize_fp8(x)
        reconstructed = decode_fp8(q) * scales[:, None]
        self.assertLess(np.linalg.norm(x-reconstructed)/np.linalg.norm(x), 0.04)
        zero, zs = quantize_fp8(np.zeros((2, 4), dtype=np.float32))
        self.assertTrue((zero == 0).all())
        self.assertTrue((zs > 0).all())
    def test_bf16_round_even(self):
        x=np.array([1., -2., 0., 1+2**-8, 1+3*2**-8], dtype=np.float32)
        out=np.frombuffer(bf16_bytes(x), dtype='<u2').astype('<u4') << 16
        np.testing.assert_array_equal(out.view('<f4'), [1., -2., 0., 1., 1+2**-6])
    def test_gdn_fused_partition(self):
        cfg={'linear_num_key_heads':4,'linear_key_head_dim':2,'linear_num_value_heads':8,'linear_value_head_dim':2}
        x=np.arange(32*3).reshape(32,3)
        a=partition('model.language_model.layers.0.linear_attn.in_proj_qkv.weight',x,cfg,0,2)
        b=partition('model.language_model.layers.0.linear_attn.in_proj_qkv.weight',x,cfg,1,2)
        np.testing.assert_array_equal(a,np.concatenate([x[:4],x[8:12],x[16:24]]))
        np.testing.assert_array_equal(b,np.concatenate([x[4:8],x[12:16],x[24:]]))
    def test_row_column_parallel_mlp(self):
        rng=np.random.default_rng(1); x=rng.normal(size=(2,8)); up=rng.normal(size=(12,8)); down=rng.normal(size=(8,12))
        parts=[]
        for rank in range(2):
            a=partition('model.language_model.layers.0.mlp.up_proj.weight',up,{},rank,2)
            b=partition('model.language_model.layers.0.mlp.down_proj.weight',down,{},rank,2)
            parts.append((x@a.T)@b.T)
        np.testing.assert_allclose(sum(parts),(x@up.T)@down.T,atol=1e-12)
    def test_vision_geometry_and_rope(self):
        coords,indices,factors=image_geometry(4,4,2,4)
        np.testing.assert_array_equal(coords[:4],[[0,0],[0,1],[1,0],[1,1]])
        np.testing.assert_allclose(factors.sum(0),1)
        pos,nxt=rope_positions([1,9,9,9,9,2],[[1,4,4]],9,2)
        np.testing.assert_array_equal(pos,[[0,1,1,1,1,3],[0,1,1,2,2,3],[0,1,2,1,2,3]])
        self.assertEqual(nxt,4)
    def test_bad_metadata_rejected(self):
        with self.assertRaises(ValueError): rope_positions([9],[],9,2)
        with self.assertRaises(ValueError): image_geometry(3,4)
        with self.assertRaises(ValueError): quantize_fp8(np.array([[np.inf]],dtype=np.float32))

if __name__=='__main__': unittest.main()