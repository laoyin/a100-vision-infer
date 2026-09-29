#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
: "${HF_MODEL:?Set HF_MODEL to the original merged HF FP8 checkpoint, not fp8-tp2}"
export CUDA_VISIBLE_DEVICES="${AVI_GPUS:-2,3}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
run="${1:-upstream-mtp-$(date +%Y%m%d-%H%M%S)}"
mkdir "$run"
exec > >(tee "$run/test.log") 2>&1
git rev-parse HEAD
python tools/inspect_mtp.py --model "$HF_MODEL" --require-mtp > "$run/mtp-audit.json"
body="${REQUEST_BODY:-}"
if [[ -z "$body" ]]; then
 : "${TEST_IMAGE:?Set TEST_IMAGE, or provide REQUEST_BODY containing embedded images}"
 body="$run/request.json"
 python tools/make_http_request.py --image "$TEST_IMAGE" \
  --prompt "${TEST_PROMPT:-Identify the equipment, counts and visible labels. Return JSON.}" \
  --max-tokens "${MAX_NEW_TOKENS:-128}" --out "$body"
fi
python tools/benchmark_mtp_upstream.py --model "$HF_MODEL" --body "$body" --out "$run/matrix" \
 --requests "${BENCH_REQUESTS:-3}" --max-tokens "${MAX_NEW_TOKENS:-128}" \
 --max-pixels "${MAX_PIXELS:-4000000}" --timeout "${PROFILE_TIMEOUT:-1800}"
