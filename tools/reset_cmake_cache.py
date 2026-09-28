"""Reset generated CMake configuration without requiring cmake --fresh."""
import argparse
from pathlib import Path
import shutil

def reset(build):
    build=Path(build)
    if build.is_symlink():raise ValueError('Build directory must not be a symlink')
    root=build.resolve()
    for name in ('CMakeCache.txt','CMakeFiles'):
        path=root/name
        if path.is_symlink():raise ValueError(f'Refusing linked CMake configuration: {path}')
        if path.exists() and path.resolve().parent!=root:raise ValueError('Configuration path escapes build directory')
    cache=root/'CMakeCache.txt'
    if cache.exists():cache.unlink()
    generated=root/'CMakeFiles'
    if generated.exists():shutil.rmtree(generated)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--build',type=Path,required=True)
    reset(parser.parse_args().build)
