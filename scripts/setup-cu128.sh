#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
command -v nvcc >/dev/null || { echo 'Missing nvcc; this script does not install system CUDA.' >&2; exit 1; }
nvcc --version | grep -q 'release 12.8,' || { echo 'Expected existing CUDA toolkit 12.8.' >&2; exit 1; }
venv="${AVI_VENV:-$PWD/.venv-cu128}"
if [[ ! -e "$venv" ]]; then "${PYTHON_BOOTSTRAP:-python3}" -m venv "$venv"; fi
[[ -f "$venv/pyvenv.cfg" && -x "$venv/bin/python" ]] || { echo 'Not a valid isolated venv' >&2; exit 1; }
export PIP_REQUIRE_VIRTUALENV=true
"$venv/bin/python" -m pip install --upgrade pip
"$venv/bin/python" -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
"$venv/bin/python" -m pip install -r requirements-test.txt -c configs/cu128-constraints.txt
"$venv/bin/python" -m pip check
"$venv/bin/python" -c 'import torch,torchvision; assert torch.version.cuda=="12.8"; print(torch.__version__,torchvision.__version__,torch.version.cuda)'
printf 'Ready. Activate with: source "%s/bin/activate"\n' "$venv"
