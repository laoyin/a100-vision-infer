import sys
from pathlib import Path
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from chat_format import vllm_string_messages
from prefill_matrix import prompt_difference, acceptance


class ChatFormatTests(unittest.TestCase):
    def test_image_and_text_have_exact_separator(self):
        message={'role':'user','content':[{'type':'image'},{'type':'text','text':'Identify'}]}
        self.assertEqual(vllm_string_messages([message],'<image>')[0]['content'],'<image>\nIdentify')
        self.assertIsInstance(message['content'],list)

    def test_default_vllm_noninterleaved_order(self):
        message={'role':'user','content':[{'type':'text','text':'before'},{'type':'image'},
                 {'type':'text','text':'after'},{'type':'image'}]}
        self.assertEqual(vllm_string_messages([message],'<image>')[0]['content'],'<image>\n<image>\nbefore\nafter')

    def test_strings_metadata_empty_text_and_image_only(self):
        messages=[{'role':'system','name':'policy','content':'stay\nexact'},
                  {'role':'user','content':[{'type':'image'},{'type':'text','text':''}]}]
        result=vllm_string_messages(messages,'<image>')
        self.assertEqual(result[0],messages[0])
        self.assertEqual(result[1]['content'],'<image>')

    def test_ambiguous_placeholders_fail(self):
        with self.assertRaises(ValueError):
            vllm_string_messages([{'role':'user','content':[{'type':'text','text':'<image>'}]}],'<image>')

    def test_prompt_comparison_catches_single_missing_separator(self):
        self.assertIsNone(prompt_difference([1,2,3],[1,2,3]))
        diff=prompt_difference([1,2,3],[1,198,2,3])
        self.assertEqual(diff['first_difference'],1)
        self.assertEqual((diff['native_length'],diff['vllm_length']),(3,4))
        self.assertIsNotNone(prompt_difference([1],[1,2]))

    def test_speed_win_requires_beating_both_upstream_windows(self):
        profiles=[{'status':'passed'}]
        rows=[{'native':'a','latency_ratio_vllm_over_native':1.1},
              {'native':'a','latency_ratio_vllm_over_native':.9}]
        self.assertFalse(acceptance(profiles,rows)['faster_on_this_workload'])
        self.assertFalse(acceptance(profiles,rows[:1])['faster_on_this_workload'])
        rows[1]['latency_ratio_vllm_over_native']=1.02
        self.assertTrue(acceptance(profiles,rows)['faster_on_this_workload'])


if __name__=='__main__':
    unittest.main()
