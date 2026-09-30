#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
: "${HF_MODEL:?Set the complete merged HF checkpoint}"
previous="${1:?Pass previous native-mtp run directory}"
run="${2:-graph-wy-test-$(date +%Y%m%d-%H%M%S)}"
test -f "$previous/fp8-mtp-tp2/manifest.json"
test -f "$previous/request/request.json"
mkdir "$run"
exec > >(tee "$run/test.log") 2>&1
trap 'code=$?; if (( code != 0 )); then echo "FAILED: exit=$code; retain $run"; fi' EXIT
export CUDA_VISIBLE_DEVICES="${AVI_GPUS:-2,3}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
if [[ $(id -u) == 0 ]]; then export OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1; fi
git rev-parse HEAD
bash scripts/build.sh
mpirun -np 2 build/avi-tp-reduce-test
bash scripts/smoke-native-mtp.sh "$run/small-model"
limits=$(python - "$previous/request/request.json" <<'PY'
import json,sys
r=json.load(open(sys.argv[1]))
print(int(r['max_new_tokens']),int(r['max_context']))
PY
)
read -r tokens context <<< "$limits"
# Fresh comparison on the same selected GPUs; never connects to production HTTP.
upstream_status=0
python -u tools/benchmark_mtp_upstream.py --model "$HF_MODEL" --body "$previous/request.json" \
 --out "$run/vllm" --requests "${BENCH_REQUESTS:-5}" --max-tokens "$tokens" --max-context "$context" \
 --max-pixels "${MAX_PIXELS:-4000000}" --timeout "${PROFILE_TIMEOUT:-1800}" || upstream_status=$?
test -f "$run/vllm/mtp2.json"
test -f "$run/vllm/mtp3.json"
python -u tools/prefill_matrix.py --suite graph-wy --previous "$previous" --hf-model "$HF_MODEL" \
 --vllm-results "$run/vllm" --out "$run/matrix" --requests "${BENCH_REQUESTS:-5}" \
 --timeout "${PROFILE_TIMEOUT:-1800}" --max-pixels "${MAX_PIXELS:-4000000}"
if (( upstream_status != 0 )); then echo "vLLM suite had failures/mismatches; inspect $run/vllm/summary.json"; exit "$upstream_status"; fi
echo "Correctness checks passed. See $run/matrix/summary.json acceptance for measured speed; pass does not imply faster."
