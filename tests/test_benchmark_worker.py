import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from benchmark_worker import summarize
class BenchmarkWorkerTests(unittest.TestCase):
 def test_failed_requests_excluded(self):
  good={'finish_reason':'length','generated_ids':[1,2],'ttft_seconds':.1,'total_seconds':.3}
  bad={'finish_reason':'timeout','generated_ids':[1]}
  result=summarize([good,good,bad],2)
  self.assertEqual(result['successful'],2);self.assertEqual(result['output_tokens_per_second'],2);self.assertTrue(result['all_tokens_equal'])
