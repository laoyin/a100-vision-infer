import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from analyze_logit_audit import analyze

class LogitAuditTests(unittest.TestCase):
    def test_first_divergence_uses_measured_session(self):
        n={'prompt_token_ids':[1,2,3],'results':[{'generated_ids':[10,19,20]}]}
        v={'results':[{'prompt_token_ids':[1,2,3],'generated_ids':[10,18,20]}]}
        events=[{'session':1,'consumed':3,'rows':[{'row':0,'top_ids':[19,18],'logits':[1.,0.]}]},
                {'session':2,'consumed':3,'rows':[{'row':0,'selected_id':19,'top_ids':[19,18],'logits':[1.,.999]}]}]
        r=analyze(n,v,events)
        self.assertEqual(r['first_difference'],1)
        self.assertEqual(r['diagnostic_session'],2)
        self.assertAlmostEqual(r['matching_scores'][0]['native_minus_vllm_logit'],.001)
    def test_matching_outputs_have_no_divergence(self):
        n={'prompt_token_ids':[1],'results':[{'generated_ids':[10,20]}]}
        v={'results':[{'prompt_token_ids':[1],'generated_ids':[10,20]}]}
        self.assertIsNone(analyze(n,v,[])['first_difference'])

if __name__=='__main__':unittest.main()
