import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from fp8_import_utils import partition_fp8,validate_quantization
from format_utils import partition,decode_fp8
class BlockFP8Tests(unittest.TestCase):
 def test_scale_partition_equivalence(self):
  text={'linear_num_key_heads':4,'linear_key_head_dim':128,'linear_num_value_heads':4,'linear_value_head_dim':128}
  for suffix,n,k in [('mlp.gate_proj.weight',512,256),('mlp.down_proj.weight',256,512),('linear_attn.in_proj_qkv.weight',1536,256)]:
   name='model.language_model.layers.0.'+suffix
   rng=np.random.default_rng(3);codes=rng.integers(0,127,(n,k),dtype=np.uint8);scales=rng.uniform(.01,.1,((n+127)//128,(k+127)//128)).astype('float32')
   reference=decode_fp8(codes)*scales.repeat(128,0)[:n].repeat(128,1)[:,:k]
   for tp in (1,2,4):
    for rank in range(tp):
     q,s=partition_fp8(name,codes,scales,text,rank,tp)
     actual=decode_fp8(q)*s.repeat(128,1)[:,:q.shape[1]]
     np.testing.assert_array_equal(actual,partition(name,reference,text,rank,tp))
 def test_invalid_scales_and_alignment(self):
  q=np.zeros((128,256),dtype='uint8')
  for scales in [np.ones((1,1)),np.zeros((1,2)),np.full((1,2),np.nan)]:
   with self.assertRaises(ValueError):partition_fp8('lm_head.weight',q,scales,{},0,1)
  with self.assertRaises(ValueError):partition_fp8('model.language_model.layers.0.mlp.down_proj.weight',q,np.ones((1,2)),{},0,4)
 def test_config(self):
  self.assertEqual(validate_quantization({'quantization_config':{'quant_method':'fp8','fmt':'e4m3','activation_scheme':'dynamic','weight_block_size':[128,128]}})['fmt'],'e4m3')
  with self.assertRaises(ValueError):validate_quantization({})
