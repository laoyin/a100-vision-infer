import sys,tempfile,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from reset_cmake_cache import reset
class CMakeResetTests(unittest.TestCase):
 def test_only_generated_configuration_removed(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp)/'build';root.mkdir();(root/'CMakeCache.txt').write_text('old');(root/'CMakeFiles').mkdir();(root/'CMakeFiles'/'old').write_text('old');(root/'avi-infer').write_text('keep')
   reset(root);self.assertFalse((root/'CMakeCache.txt').exists());self.assertFalse((root/'CMakeFiles').exists());self.assertEqual((root/'avi-infer').read_text(),'keep');reset(root)
