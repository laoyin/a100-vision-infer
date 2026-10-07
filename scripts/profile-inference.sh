#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
: "${HF_MODEL:?Set the merged HF model path}"
previous="${1:?Pass the previous native-mtp artifact directory}"
run="${2:-profile-test-$(date +%Y%m%d-%H%M%S)}"
body="${AVI_BODY:-$previous/request.json}"
export CUDA_VISIBLE_DEVICES="${AVI_GPUS:-2,3}"
flags=()
if [[ "${PROFILE_SKIP_BUILD:-0}" == 1 ]]; then flags+=(--skip-build); fi
# Python creates a fresh output directory and logs every subprocess separately.
python -u tools/profile_inference.py --previous "$previous" --model "$HF_MODEL" --body "$body" --out "$run" \
 --decode-tokens "${PROFILE_TOKENS:-512}" --max-context "${MAX_CONTEXT:-20480}" \
 --max-pixels "${MAX_PIXELS:-4000000}" --timeout "${PROFILE_TIMEOUT:-1800}" \
 --variant "${PROFILE_VARIANT:-both}" --ncu "${PROFILE_NCU:-auto}" "${flags[@]}"
