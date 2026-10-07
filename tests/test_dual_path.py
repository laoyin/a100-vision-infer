import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from workload_metrics import workload_metrics
from benchmark_worker import override_limits
from prefill_matrix import dual_profiles
from summarize_dual_path import combine

class DualPathTests(unittest.TestCase):
    def row(self,**kwargs):
        return dict(dict(text='{"x":[1,2]}',generated_ids=list(range(2048)),finish_reason='eos',total_seconds=10,ttft_seconds=2),**kwargs)
    def test_natural_long_json(self):
        r=workload_metrics([self.row()],1024)
        self.assertTrue(r['all_complete_json']);self.assertTrue(r['all_long_enough'])
        self.assertEqual(r['post_first_token_tps_p50'],2047/8)
    def test_truncation_is_not_completion(self):
        r=workload_metrics([self.row(finish_reason='length')],1024)
        self.assertFalse(r['all_complete_json']);self.assertTrue(r['requests'][0]['json_valid'])
    def test_short_output_does_not_pass_long_test(self):
        r=workload_metrics([self.row(generated_ids=[1,2])],1024)
        self.assertFalse(r['all_long_enough'])
    def test_invalid_json(self):
        for text in ('{"a":','```json\n{}\n```','42','{"x":NaN}'):
            self.assertFalse(workload_metrics([self.row(text=text)])['all_complete_json'])
    def test_no_fabricated_ttft(self):
        row=self.row();row.pop('ttft_seconds')
        self.assertIsNone(workload_metrics([row])['post_first_token_tps_p50'])
        self.assertIsNone(workload_metrics([self.row(ttft_seconds=10)])['post_first_token_tps_p50'])
    def test_empty_results_fail(self):
        self.assertFalse(workload_metrics([])['all_complete_json'])
    def test_limits_do_not_mutate_source(self):
        req={'max_new_tokens':64,'max_context':20480}
        self.assertEqual(override_limits(req,8192)['max_new_tokens'],8192)
        self.assertEqual(req['max_new_tokens'],64)
        with self.assertRaises(ValueError):override_limits(req,20480)
    def test_profiles_keep_one_token_probe_free_of_unused_graphs(self):
        for name,flags,_,_ in dual_profiles([],[],True):
            self.assertNotIn('--mtp-verify-graph',flags)
        full={name:flags for name,flags,_,_ in dual_profiles([],[],False)}
        self.assertIn('--fused-residual-norm',full['combined'])
        self.assertIn('--gpu-candidates',full['combined'])
        self.assertIn('--mtp-verify-graph',full['combined'])

    def test_joint_win_requires_same_profile_and_both_upstreams(self):
        def report(name,ratios):
            return {'profiles':[{'name':name,'status':'passed'}],
                    'comparison':[{'native':name,'vllm':v,'latency_ratio_vllm_over_native':ratio} for v,ratio in ratios.items()]}
        good=report('combined',{'mtp2':1.5,'mtp3':1.2})
        self.assertTrue(combine(good,good)['both_paths_faster'])
        self.assertFalse(combine(good,report('other',{'mtp2':2,'mtp3':2}))['both_paths_faster'])
        self.assertFalse(combine(good,report('combined',{'mtp2':2}))['both_paths_faster'])
        self.assertFalse(combine(good,report('combined',{'mtp2':2,'mtp3':.9}))['both_paths_faster'])
