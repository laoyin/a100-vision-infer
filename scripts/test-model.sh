#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
: "${HF_MODEL:?Set HF_MODEL to the existing block-FP8 checkpoint with processor/tokenizer}"
: "${TEST_IMAGE:?Set TEST_IMAGE to the image file}"
command -v python >/dev/null || { echo "Activate your existing Python environment first." >&2; exit 1; }
export CUDA_VISIBLE_DEVICES="${AVI_GPUS:-2,3}"
if [[ $(id -u) == 0 ]]; then export OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1; fi
run="${1:-model-test-$(date +%Y%m%d-%H%M%S)}"
mkdir "$run"
exec > >(tee "$run/test.log") 2>&1
python tools/check_build_env.py
bash scripts/build.sh
bash scripts/smoke-block-fp8.sh "$run/block-smoke"
python tools/prepare_request.py --model "$HF_MODEL" --image "$TEST_IMAGE" \
  --prompt "Identify the equipment, counts and visible labels. Return JSON." \
  --max-pixels "${MAX_PIXELS:-4000000}" --max-context 20480 --max-new-tokens 128 --out "$run/request"
python tools/import_fp8.py --model "$HF_MODEL" --out "$run/fp8-tp2" --tp 2
python tools/preflight.py --model "$run/fp8-tp2" --tp 2
for mode in baseline optimized graph; do
  flags=()
  [[ "$mode" != baseline ]] || flags+=(--baseline)
  [[ "$mode" != graph ]] || flags+=(--cuda-graph)
  mpirun -np 2 ./build/avi-infer --model "$run/fp8-tp2" --request "$run/request" \
    --output "$run/result-$mode.json" --prefill-chunk 128 --trace "${flags[@]}"
  python tools/decode.py --model "$HF_MODEL" --result "$run/result-$mode.json"
  if [[ "$mode" != baseline ]]; then
    python tools/compare_native.py --baseline "$run/result-baseline.json" --candidate "$run/result-$mode.json"
  fi
done
echo 'FP8 native paths completed. Compare business outputs with the existing serving runtime; no BF16 model artifact was generated.'
