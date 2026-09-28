import json,sys,tempfile,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from export_vocabulary import export_vocabulary
class VocabularyTests(unittest.TestCase):
    def test_bytes_and_specials(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)
            config={'model':{'type':'BPE','vocab':{'A':0,chr(288):1,chr(266):2}},'decoder':{'type':'ByteLevel'},'added_tokens':[{'id':3,'content':'<eos>','special':True}]}
            (path/'tokenizer.json').write_text(json.dumps(config))
            self.assertTrue(export_vocabulary(path,path,4))
            self.assertEqual(json.loads((path/'token_bytes.json').read_text()),['41','20','0a',''])
            config['decoder']={'type':'WordPiece'}
            (path/'tokenizer.json').write_text(json.dumps(config))
            self.assertFalse(export_vocabulary(path,path,4))
