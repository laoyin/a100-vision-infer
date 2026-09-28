#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
venv="${AVI_VENV:-$PWD/.venv-cu128}"
[[ -f "$venv/bin/activate" ]] || { echo 'Run bash scripts/setup-cu128.sh first.' >&2; exit 1; }
source "$venv/bin/activate"
export CUDA_VISIBLE_DEVICES="${AVI_GPUS:-2,3}"
if [[ $(id -u) == 0 ]]; then
  export OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
fi
run="${1:-acceptance-$(date +%Y%m%d-%H%M%S)}"
bash scripts/acceptance.sh "$run"
