"""Benchmark-only import hook; never edits installed FlashInfer or CUDA files."""
import ast
import importlib.abc
import importlib.machinery
import os
import sys


REPLACEMENT = '''
def find_loaded_library(lib_name):
    import re
    from pathlib import Path
    pattern = re.compile(re.escape(lib_name) + r'(?:-[0-9a-fA-F]+)?\\.so(?:\\.\\d+)*')
    with open('/proc/self/maps') as maps:
        for line in maps:
            fields = line.rstrip().split(None, 5)
            if len(fields) == 6 and pattern.fullmatch(Path(fields[5]).name):
                return fields[5]
    return None
'''


class RuntimeLoader(importlib.machinery.SourceFileLoader):
    def get_code(self, fullname):
        # Transform before the module-level CudaRTLibrary() binds CUDA symbols.
        tree = ast.parse(self.get_data(self.path), filename=self.path)
        matches = [i for i, node in enumerate(tree.body)
                   if isinstance(node, ast.FunctionDef) and node.name == 'find_loaded_library']
        if len(matches) != 1:
            raise ImportError('AVI compatibility hook: unsupported FlashInfer cuda_ipc source; expected one find_loaded_library')
        tree.body[matches[0]] = ast.parse(REPLACEMENT).body[0]
        ast.fix_missing_locations(tree)
        print('[AVI compat] FlashInfer CUDA library lookup uses exact library filenames (excludes stubs)',
              file=sys.stderr, flush=True)
        return compile(tree, self.path, 'exec')


class RuntimeFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname != 'flashinfer.comm.cuda_ipc':
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or not isinstance(spec.loader, importlib.machinery.SourceFileLoader):
            raise ImportError('AVI compatibility hook requires FlashInfer cuda_ipc Python source')
        spec.loader = RuntimeLoader(fullname, spec.origin)
        return spec


if os.environ.get('AVI_FLASHINFER_RUNTIME_COMPAT') == '1':
    sys.meta_path.insert(0, RuntimeFinder())
