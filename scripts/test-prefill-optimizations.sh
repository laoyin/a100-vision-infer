#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
: "${HF_MODEL:?Set the complete merged HF checkpoint}"
previous="${1:?Pass previous native-mtp run directory}"
upstream="${2:?Pass successful upstream retest directory}"
run="${3:-prefill-test-$(date +%Y%m%d-%H%M%S)}"
test -f "$previous/fp8-mtp-tp2/manifest.json"
test -f "$previous/request/request.json"
test -f "$upstream/vllm/mtp2.json"
mkdir "$run"
exec > >(tee "$run/test.log") 2>&1
trap 'code=$?; if (( code != 0 )); then echo "FAILED: exit=$code; retain $run"; fi' EXIT
export CUDA_VISIBLE_DEVICES="${AVI_GPUS:-2,3}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
if [[ $(id -u) == 0 ]]; then export OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1; fi
git rev-parse HEAD
bash scripts/build.sh
python -u tools/prefill_matrix.py --previous "$previous" --hf-model "$HF_MODEL" \
 --vllm-results "$upstream/vllm" --out "$run/matrix" \
 --requests "${BENCH_REQUESTS:-5}" --timeout "${PROFILE_TIMEOUT:-1800}" --max-pixels "${MAX_PIXELS:-4000000}"
echo "PASSED: inspect $run/matrix/summary.json; speed ratios require exact input/output equality."
