#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
: "${AVI_MODEL:?Set AVI_MODEL to existing imported fp8-tp2}"
: "${AVI_REQUEST:?Set AVI_REQUEST to existing prepared request}"
export CUDA_VISIBLE_DEVICES="${AVI_GPUS:-2,3}"
if [[ $(id -u) == 0 ]]; then export OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1; fi
run="${1:-optimizations-$(date +%Y%m%d-%H%M%S)}"
mkdir "$run"
exec > >(tee "$run/test.log") 2>&1
git rev-parse HEAD
python -m unittest discover -s tests -p 'test_*.py'
bash scripts/build.sh
python tools/preflight.py --model "$AVI_MODEL" --tp 2
python tools/optimization_matrix.py --model "$AVI_MODEL" --request "$AVI_REQUEST" --out "$run/matrix" \
 --requests "${BENCH_REQUESTS:-3}" --timeout "${PROFILE_TIMEOUT:-1800}"
echo "Completed: $run/matrix/summary.json"
