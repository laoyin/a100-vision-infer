#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES="${AVI_GPUS:-2,3}"
if [[ $(id -u) == 0 ]]; then export OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1; fi
root="${1:-smoke-run-block-fp8-$(date +%Y%m%d-%H%M%S)}"
python tools/make_tiny_fixture.py --out "$root" --block-fp8
exec > >(tee "$root/test.log") 2>&1
python tools/import_fp8.py --model "$root/model" --out "$root/native-tp2" --tp 2
python tools/preflight.py --model "$root/native-tp2" --tp 2
for mode in baseline optimized graph; do
 flags=(); [[ "$mode" != baseline ]] || flags+=(--baseline); [[ "$mode" != graph ]] || flags+=(--cuda-graph)
 mpirun -np 2 ./build/avi-infer --model "$root/native-tp2" --request "$root/request" --output "$root/$mode.json" --prefill-chunk 4 --trace "${flags[@]}"
 if [[ "$mode" != baseline ]]; then python tools/compare_native.py --baseline "$root/baseline.json" --candidate "$root/$mode.json"; fi
done
python tools/test_worker.py --model "$root/native-tp2" --request "$root/request" --tp 2 --cuda-graph
echo 'Block FP8 import and native path consistency passed; original W8A8/business accuracy not validated.'
