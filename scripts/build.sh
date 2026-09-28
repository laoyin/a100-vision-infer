#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
command -v nvcc >/dev/null || { echo 'Existing CUDA toolkit nvcc is required.' >&2; exit 1; }
command -v mpicxx >/dev/null || { echo 'Install OpenMPI development packages.' >&2; exit 1; }
# Use the installed compiler, never the CUDA version printed by nvidia-smi.
export CUDACXX="$(readlink -f "$(command -v nvcc)")"
export CUDA_HOME="$(dirname "$(dirname "$CUDACXX")")"
python tools/check_build_env.py
prefix="$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')"
nccl_root="${NCCL_ROOT:-$(python -c 'import sys,pathlib; print(next((str(pathlib.Path(p)/"nvidia/nccl") for p in sys.path if (pathlib.Path(p)/"nvidia/nccl/include/nccl.h").is_file()),""))')}"
extra=()
if [[ -n "$nccl_root" ]]; then
  extra+=("-DNCCL_INCLUDE_DIR=$nccl_root/include")
  if [[ -f "$nccl_root/lib/libnccl.so.2" ]]; then extra+=("-DNCCL_LIBRARY=$nccl_root/lib/libnccl.so.2"); fi
fi
# Reset only generated CMake configuration; retain compiled objects and all test results.
python tools/reset_cmake_cache.py --build build
command -v cmake
cmake --version
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$prefix" \
  -DCMAKE_CUDA_COMPILER="$CUDACXX" -DCUDAToolkit_ROOT="$CUDA_HOME" -DCUDA_TOOLKIT_ROOT_DIR="$CUDA_HOME" \
  -DCMAKE_CUDA_ARCHITECTURES=80 "${extra[@]}"
cmake --build build --parallel "${BUILD_JOBS:-2}"
ctest --test-dir build --output-on-failure
