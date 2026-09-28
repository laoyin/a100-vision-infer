#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
run="${1:-acceptance-$(date +%Y%m%d-%H%M%S)}"
mkdir "$run"
exec > >(tee "$run/acceptance.log") 2>&1
trap 'status=$?; printf "Exit status: %s\n" "$status"; if [[ $status != 0 ]]; then echo "FAILED: retain this directory and build/CMakeFiles diagnostics."; fi' EXIT
python -m pip freeze > "$run/pip-freeze.txt"
git rev-parse HEAD > "$run/commit.txt"
python tools/check_build_env.py
nvidia-smi
nvidia-smi topo -m
nvcc --version
python -c 'import torch,platform; print(platform.platform()); print(torch.__version__,torch.version.cuda); print("NCCL",torch.cuda.nccl.version())'
python -m unittest discover -s tests -v
bash scripts/build.sh
bash scripts/smoke.sh "$run/smoke"
echo 'Small-model acceptance passed; run real checkpoint and business quality validation next.'
