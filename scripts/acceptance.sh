#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
run="${1:-acceptance-$(date +%Y%m%d-%H%M%S)}"
mkdir "$run"
exec > >(tee "$run/acceptance.log") 2>&1
nvidia-smi
nvidia-smi topo -m
nvcc --version
python -c 'import torch,platform; print(platform.platform()); print(torch.__version__,torch.version.cuda); print("NCCL",torch.cuda.nccl.version())'
python -m unittest discover -s tests -v
bash scripts/build.sh
bash scripts/smoke.sh "$run/smoke"
echo 'Small-model acceptance passed; run real checkpoint and business quality validation next.'
