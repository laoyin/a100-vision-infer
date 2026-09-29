#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
: "${HF_MODEL:?Set the original complete merged HF checkpoint}"
previous="${1:?Usage: bash scripts/retest-vllm-mtp.sh native-mtp-TIMESTAMP}"
test -f "$previous/request.json"
test -f "$previous/request/request.json"
run="${2:-upstream-mtp-retest-$(date +%Y%m%d-%H%M%S)}"
mkdir "$run"
exec > >(tee "$run/test.log") 2>&1
trap 'code=$?; if (( code != 0 )); then echo "FAILED: exit=$code; retain $run"; fi' EXIT
export CUDA_VISIBLE_DEVICES="${AVI_GPUS:-2,3}"
git rev-parse HEAD
limits=$(python - "$previous/request/request.json" <<'PY'
import json, sys
r = json.load(open(sys.argv[1]))
print(int(r['max_new_tokens']), int(r['max_context']))
PY
)
read -r tokens context <<< "$limits"
python -u tools/benchmark_mtp_upstream.py --model "$HF_MODEL" \
  --body "$previous/request.json" --out "$run/vllm" \
  --requests "${BENCH_REQUESTS:-5}" --max-tokens "$tokens" --max-context "$context" \
  --max-pixels "${MAX_PIXELS:-4000000}" --timeout "${PROFILE_TIMEOUT:-1800}"
echo "PASSED: $run/vllm/summary.json"
