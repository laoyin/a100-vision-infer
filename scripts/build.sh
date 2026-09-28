#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
command -v nvcc >/dev/null || { echo 'CUDA toolkit (nvcc) is required, not only an NVIDIA driver.' >&2; exit 1; }
command -v mpicxx >/dev/null || { echo 'Install OpenMPI development packages.' >&2; exit 1; }
python -c 'import torch; assert torch.version.cuda is not None, "CPU-only PyTorch is not supported"; print("PyTorch",torch.__version__,"CUDA",torch.version.cuda)'
prefix="$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')"
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$prefix" -DCMAKE_CUDA_ARCHITECTURES=80
cmake --build build --parallel "${BUILD_JOBS:-2}"
ctest --test-dir build --output-on-failure