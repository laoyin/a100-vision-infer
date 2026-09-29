#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
: "${HF_MODEL:?Set the complete merged HF FP8 checkpoint containing MTP}"
: "${TEST_IMAGE:?Set a representative image path}"
export CUDA_VISIBLE_DEVICES="${AVI_GPUS:-2,3}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
if [[ $(id -u) == 0 ]]; then export OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1; fi
run="${1:-native-mtp-$(date +%Y%m%d-%H%M%S)}"
mkdir "$run"
exec > >(tee "$run/test.log") 2>&1
stage="checkpoint audit"
trap 'code=$?; if (( code != 0 )); then echo "FAILED: $stage exit=$code; retain $run"; fi' EXIT
git rev-parse HEAD
python tools/inspect_mtp.py --model "$HF_MODEL" --require-mtp > "$run/mtp-audit.json"
stage="compile and kernel/state tests"
bash scripts/build.sh
stage="small-model native MTP tests"
bash scripts/smoke-native-mtp.sh "$run/smoke"
stage="import original FP8 codes plus MTP"
python tools/import_fp8.py --model "$HF_MODEL" --out "$run/fp8-mtp-tp2" --tp 2 --include-mtp
python tools/preflight.py --model "$run/fp8-mtp-tp2" --tp 2
stage="prepare identical native/upstream request"
prompt="${TEST_PROMPT:-Identify the equipment, counts and visible labels. Return JSON.}"
python tools/prepare_request.py --model "$HF_MODEL" --image "$TEST_IMAGE" --prompt "$prompt" \
 --max-pixels "${MAX_PIXELS:-4000000}" --max-context 20480 --max-new-tokens "${MAX_NEW_TOKENS:-128}" --out "$run/request"
python tools/make_http_request.py --image "$TEST_IMAGE" --prompt "$prompt" --max-tokens "${MAX_NEW_TOKENS:-128}" --out "$run/request.json"
stage="native optimization matrix and vLLM MTP comparison"
python -u tools/native_mtp_matrix.py --model "$run/fp8-mtp-tp2" --hf-model "$HF_MODEL" \
 --request "$run/request" --body "$run/request.json" --out "$run/matrix" \
 --requests "${BENCH_REQUESTS:-5}" --timeout "${PROFILE_TIMEOUT:-1800}" \
 --weight-cache-mib "${WEIGHT_CACHE_MIB:-24576}" --max-pixels "${MAX_PIXELS:-4000000}"
echo "PASSED: $run/matrix/summary.json"
