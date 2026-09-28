#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
root="${1:?Set a new directory for small-model validation}"
export CUDA_VISIBLE_DEVICES="${AVI_GPUS:-2,3}"
if [[ $(id -u) == 0 ]]; then export OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1; fi
bash scripts/smoke-block-fp8.sh "$root"
for mode in reference head vector combined graph; do
 flags=()
 case "$mode" in
  reference) flags+=(--reference-prefill);;
  head) flags+=(--tp-lm-head);;
  vector) flags+=(--vector-gemv);;
  combined) flags+=(--tp-lm-head --vector-gemv --extra-fusions --cublas-prefill);;
  graph) flags+=(--tp-lm-head --vector-gemv --extra-fusions --cublas-prefill --cuda-graph);;
 esac
 mpirun -np 2 ./build/avi-infer --model "$root/native-tp2" --request "$root/request" \
  --output "$root/deep-$mode.json" --prefill-chunk 16 --trace "${flags[@]}"
 python tools/compare_native.py --baseline "$root/baseline.json" --candidate "$root/deep-$mode.json"
done
python tools/test_worker.py --model "$root/native-tp2" --request "$root/request" --tp 2 --tp-lm-head --vector-gemv --cuda-graph
