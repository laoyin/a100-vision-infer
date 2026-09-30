import sys
from pathlib import Path
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from prefill_matrix import fused_profiles,acceptance

class FusedProfileTests(unittest.TestCase):
    def test_isolated_and_combined_optimizations(self):
        rows=fused_profiles(['--frontend-threads','4'],['--gdn-cooperative'])
        by_name={name:(flags,chunk) for name,flags,chunk,cache in rows}
        self.assertEqual(len(rows),len(by_name))
        self.assertNotIn('--gdn-wy',by_name['reference'][0])
        self.assertNotIn('--multi-token-gemv',by_name['reference'][0])
        self.assertIn('--reuse-verify-graph',by_name['graph-pool'][0])
        self.assertIn('--fused-gdn-prepare',by_name['prepare'][0])
        self.assertIn('--multi-token-gemv-fp8',by_name['shared-fp8'][0])
        self.assertIn('--gdn-wy-fused',by_name['wy-fused'][0])
        self.assertEqual(by_name['combined-2048'][1],2048)
        for name,(flags,chunk) in by_name.items():
            if '--reuse-verify-graph' in flags:self.assertIn('--mtp-verify-graph',flags)
            if name.startswith('diagnostic'):
                self.assertIn('--profile-kernels',flags)
                self.assertNotIn('--reuse-verify-graph',flags)
    def test_no_matching_output_means_no_speed_win(self):
        result=acceptance([{'status':'passed'}],[{'native':'reference','latency_ratio_vllm_over_native':None}])
        self.assertFalse(result['faster_on_this_workload'])
        self.assertIsNone(result['best_latency_ratio_vs_fastest_vllm'])

if __name__=='__main__':unittest.main()
