import sys,tempfile,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from preflight import descriptor
class PreflightTests(unittest.TestCase):
 def test_valid_and_truncated(self):
  with tempfile.TemporaryDirectory() as tmp:
   p=Path(tmp);(p/'x').write_bytes(bytes(8));d={'file':'x','dtype':'F32','shape':[2]}
   self.assertEqual(descriptor(p,d),8)
   (p/'x').write_bytes(bytes(7))
   with self.assertRaises(ValueError):descriptor(p,d)
 def test_metadata(self):
  for d in [{'file':'../x','shape':[1],'dtype':'U8'},{'file':'x','shape':[0],'dtype':'F32'},{'file':'x','shape':[True],'dtype':'U8'}]:
   with self.assertRaises(ValueError):descriptor('.',d)
