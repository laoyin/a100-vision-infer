import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import mock_open, patch


HOOK = Path(__file__).resolve().parents[1]/'tools'/'compat_flashinfer'
spec = importlib.util.spec_from_file_location('avi_runtime_hook', HOOK/'sitecustomize.py')
hook = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hook)


class RuntimeCompatTests(unittest.TestCase):
    def test_library_selection_excludes_stubs_and_directory_matches(self):
        namespace = {}
        exec(hook.REPLACEMENT, namespace)
        invalid = ['libcudart_stub.so', 'libcudart-stub.so', 'libcudart.so.backup',
                   'libcudart.something', 'libcudart.so/other.so']
        for real in ('libcudart.so', 'libcudart.so.13', 'libcudart-ab12.so.12.0'):
            maps = ''.join('0-1 r-xp 0 00:00 0 /libs/'+name+'\n' for name in invalid+[real])
            with patch('builtins.open', mock_open(read_data=maps)):
                self.assertEqual(namespace['find_loaded_library']('libcudart'), '/libs/'+real)
        with patch('builtins.open', mock_open(read_data='0-1 r-xp 0 00:00 0 /libs/libcudart_stub.so\n')):
            self.assertIsNone(namespace['find_loaded_library']('libcudart'))

    def test_child_import_is_fixed_before_module_initialization(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package = root/'flashinfer'/'comm'
            package.mkdir(parents=True)
            (package.parent/'__init__.py').write_text('')
            (package/'__init__.py').write_text('')
            (package/'cuda_ipc.py').write_text(
                'def find_loaded_library(lib_name):\n    return "wrong_stub"\n'
                'selected = find_loaded_library("libcudart")\n')
            code = ('from unittest.mock import patch, mock_open\n'
                    'with patch("builtins.open", mock_open(read_data="0-1 r-xp 0 00:00 0 /libs/libcudart_stub.so\\n0-1 r-xp 0 00:00 0 /libs/libcudart.so.13\\n")):\n'
                    ' import flashinfer.comm.cuda_ipc as m\n'
                    ' print(m.selected)\n')
            for enabled, expected in [('1', '/libs/libcudart.so.13'), ('0', 'wrong_stub')]:
                env = dict(os.environ, PYTHONPATH=str(HOOK)+os.pathsep+str(root),
                           AVI_FLASHINFER_RUNTIME_COMPAT=enabled)
                result = subprocess.run([sys.executable, '-c', code], env=env, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stdout.strip(), expected)


if __name__ == '__main__':
    unittest.main()
