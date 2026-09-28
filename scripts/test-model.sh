#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
: "${HF_MODEL:?Set HF_MODEL to the merged BF16 checkpoint with processor/tokenizer}"
: "${TEST_IMAGE:?Set TEST_IMAGE to the image file}"
command -v python >/dev/null || { echo "Activate your existing Python environment first." >&2; exit 1; }
export CUDA_VISIBLE_DEVICES="${AVI_GPUS:-2,3}"
if [[ $(id -u) == 0 ]]; then export OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1; fi
run="${1:-model-test-$(date +%Y%m%d-%H%M%S)}"
mkdir "$run"
exec > >(tee "$run/test.log") 2>&1
python tools/check_build_env.py
python tools/prepare_request.py --model "$HF_MODEL" --image "$TEST_IMAGE" \
  --prompt "Identify the equipment, counts and visible labels. Return JSON." \
  --max-pixels "${MAX_PIXELS:-4000000}" --max-context 20480 --max-new-tokens 128 --out "$run/request"
for precision in bf16 fp8; do
  python tools/convert.py --model "$HF_MODEL" --out "$run/$precision-tp2" --tp 2 --precision "$precision"
  python tools/preflight.py --model "$run/$precision-tp2" --tp 2
  mpirun -np 2 ./build/avi-infer --model "$run/$precision-tp2" --request "$run/request" \
    --output "$run/result-$precision.json" --prefill-chunk 128 --trace
  python tools/decode.py --model "$HF_MODEL" --result "$run/result-$precision.json"
  python tools/compare_reference.py --model "$HF_MODEL" --request "$run/request" \
    --native-output "$run/result-$precision.json" --decode-check 8
done
echo 'Numerical checks passed. Review image recognition content before performance testing.'
