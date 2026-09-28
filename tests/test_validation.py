import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from validation import validate_body

class ValidationTests(unittest.TestCase):
    def test_valid(self):
        validate_body({'messages':[{'role':'user','content':'test'}], 'temperature':0, 'seed':2**64-1},20480)
    def test_invalid(self):
        for field, value in [('temperature',float('nan')),('temperature','0'),('top_p',0),('seed',-1),('seed',2**64),('top_k',1.2),('max_tokens',True),('max_tokens',20480),('timeout_seconds',float('inf')),('stream','false')]:
            with self.subTest(field=field,value=value),self.assertRaises(ValueError):
                validate_body({'messages':[{}],field:value},20480)
        for body in [[],None,{}, {'messages':[1]}, {'messages':[{'content':[1]}]}]:
            with self.assertRaises(ValueError):validate_body(body,20480)
