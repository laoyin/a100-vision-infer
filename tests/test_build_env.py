import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from check_build_env import cuda_release
class BuildEnvTests(unittest.TestCase):
 def test_versions(self):
  self.assertEqual(cuda_release('Cuda compilation tools, release 12.8, V12.8.93'),'12.8')
  self.assertEqual(cuda_release('release 13.0, V13.0'),'13.0')
  with self.assertRaises(ValueError):cuda_release('driver only')
