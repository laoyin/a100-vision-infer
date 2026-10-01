import math
import sys
from pathlib import Path
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from tilelang_export_utils import GDN_ABI,FP8_ABI,validate_call_abi,choose_candidate,split_ranges
from prefill_matrix import tiled_profiles,counter_exercised


def wrapper(kind):
    pointers = [('Q','bfloat16_t'),('K','bfloat16_t'),('V','bfloat16_t'),
                *[(n,'float') for n in ('G','B','A','W','U','SQ','WK','last')]] if kind=='gdn' else [
                ('X','bfloat16_t'),('Codes','uint8_t'),('Scales','float'),('Partial','float')]
    fields=[f'{dtype}* __restrict__ {name}' for name,dtype in pointers]
    fields += ['int blocks'] if kind=='gdn' else ['int K','int N']
    return 'extern "C" TL_EXPORT int call('+', '.join(fields+['cudaStream_t stream=cudaStreamDefault'])+') { return 0; }'


class TileLangExportTests(unittest.TestCase):
    def test_supported_standalone_abis(self):
        self.assertEqual(validate_call_abi(wrapper('gdn'),'gdn'),GDN_ABI)
        self.assertEqual(validate_call_abi(wrapper('fp8'),'fp8'),FP8_ABI)

    def test_reject_reordered_pointer_or_dimensions(self):
        for source,kind in [
            (wrapper('gdn').replace('bfloat16_t* __restrict__ Q','bfloat16_t* __restrict__ Z'),'gdn'),
            (wrapper('fp8').replace('int K, int N','int N, int K'),'fp8'),
            (wrapper('fp8').replace('uint8_t*','float*'),'fp8')]:
            with self.assertRaises(ValueError):validate_call_abi(source,kind)

    def test_reject_scalar_width_and_missing_stream(self):
        with self.assertRaises(ValueError):
            validate_call_abi(wrapper('gdn').replace('int blocks','int64_t blocks'),'gdn')
        with self.assertRaises(ValueError):
            validate_call_abi(wrapper('fp8').replace(', cudaStream_t stream=cudaStreamDefault',''),'fp8')
        with self.assertRaises(ValueError):validate_call_abi('int main() { return 0; }','gdn')

    def test_tuner_never_selects_failed_or_invalid_timings(self):
        good={'validated':True,'event_interval_ms':2}
        rows=[{'validated':False,'event_interval_ms':.001},good]
        rows += [{'validated':True,'event_interval_ms':x} for x in (0,-1,float('nan'),float('inf'))]
        self.assertIs(choose_candidate(rows),good)
        with self.assertRaises(ValueError):choose_candidate(rows[2:])

    def test_split_k_covers_each_block_exactly_including_empty_partitions(self):
        for k in (128,256,384,512,5120,8704,17408):
            for split in (1,4):
                ranges=split_ranges(k,split)
                self.assertEqual(len(ranges),split)
                self.assertEqual(ranges[0][0],0)
                self.assertEqual(ranges[-1][1],k)
                self.assertTrue(all(a<=b and a%128==0 and b%128==0 for a,b in ranges))
                self.assertTrue(all(a[1]==b[0] for a,b in zip(ranges,ranges[1:])))
        self.assertEqual(split_ranges(128,4),[(0,128),(128,128),(128,128),(128,128)])
        for k,s in ((0,1),(127,1),(128,2)):
            with self.assertRaises(ValueError):split_ranges(k,s)

    def test_native_matrix_does_not_require_tilelang(self):
        rows=tiled_profiles([],[])
        self.assertEqual(len(rows),len({r[0] for r in rows}))
        for name,flags,_,_ in rows:
            self.assertNotIn('--gdn-tilelang',flags)
            self.assertNotIn('--tilelang-fp8',flags)
            if name.startswith('diagnostic'):self.assertNotIn('--mtp-verify-graph',flags)
        self.assertTrue(any('--fp8-tensor-split' in flags for _,flags,_,_ in rows))

    def test_tilelang_matrix_tests_isolated_and_combined_paths(self):
        rows=tiled_profiles([],[],Path('/kernels'))
        by_name={name:flags for name,flags,_,_ in rows}
        self.assertIn('--gdn-tilelang',by_name['tilelang-gdn32'])
        self.assertNotIn('--tilelang-fp8',by_name['tilelang-gdn32'])
        self.assertIn('--tilelang-fp8',by_name['tilelang-fp8-4'])
        self.assertNotIn('--gdn-tilelang',by_name['tilelang-fp8-4'])
        for name,flags in by_name.items():
            if '--gdn-tilelang' in flags or '--tilelang-fp8' in flags:
                self.assertIn('--tilelang-dir',flags)
            self.assertFalse('--gdn-fused-solve' in flags and '--gdn-tilelang' in flags)
            self.assertFalse('--fp8-tensor-small' in flags and '--tilelang-fp8' in flags)

    def test_each_measured_request_must_execute_new_kernel(self):
        key='tilelang_fp8_calls'
        report={'warmup_cache_baseline':{key:48},'results':[{'cache':{key:96}},{'cache':{key:144}}]}
        self.assertTrue(counter_exercised(report,key))
        report['results'][1]['cache'][key]=96
        self.assertFalse(counter_exercised(report,key))
        self.assertFalse(counter_exercised({'results':[]},key))
        self.assertFalse(counter_exercised({'results':[{}]},key))


if __name__=='__main__':unittest.main()
