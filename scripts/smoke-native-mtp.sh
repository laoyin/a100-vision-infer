#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
root="${1:?Supply a new small-model directory}"
export CUDA_VISIBLE_DEVICES="${AVI_GPUS:-2,3}"
if [[ $(id -u) == 0 ]]; then export OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1; fi
python tools/make_tiny_fixture.py --out "$root" --block-fp8 --mtp
for tp in 1 2; do
 python tools/import_fp8.py --model "$root/model" --out "$root/tp$tp" --tp "$tp" --include-mtp
 python tools/preflight.py --model "$root/tp$tp" --tp "$tp"
 for window in 0 1 2 3 5; do
  mpirun -np "$tp" ./build/avi-infer --model "$root/tp$tp" --request "$root/request" \
   --output "$root/tp$tp-mtp$window.json" --mtp-tokens "$window" --tp-lm-head --vector-gemv --prefill-chunk 4
 done
 for variant in graph cached; do
  flags=(--mtp-draft-graph)
  if [[ "$variant" == cached ]]; then flags=(--weight-cache-mib 64); fi
  mpirun -np "$tp" ./build/avi-infer --model "$root/tp$tp" --request "$root/request" \
   --output "$root/tp$tp-mtp-$variant.json" --mtp-tokens 3 --tp-lm-head --vector-gemv --prefill-chunk 4 "${flags[@]}"
 done
 python tools/check_native_mtp_results.py --root "$root" --tp "$tp"
done
python tools/test_worker.py --model "$root/tp2" --request "$root/request" --tp 2 --tp-lm-head --vector-gemv --mtp-tokens 3
python tools/test_worker.py --model "$root/tp2" --request "$root/request" --tp 2 --tp-lm-head --vector-gemv --mtp-tokens 3 --mtp-draft-graph
