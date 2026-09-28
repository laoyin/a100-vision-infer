#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
: "${AVI_MODEL:?Set AVI_MODEL to the existing imported fp8-tp2 directory}"
: "${AVI_REQUEST:?Set AVI_REQUEST to the existing prepared request directory}"
export CUDA_VISIBLE_DEVICES="${AVI_GPUS:-2,3}"
if [[ $(id -u) == 0 ]]; then export OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1; fi
run="${1:-benchmark-$(date +%Y%m%d-%H%M%S)}"
mkdir "$run"
exec > >(tee "$run/benchmark.log") 2>&1
python tools/check_build_env.py
bash scripts/build.sh
python tools/preflight.py --model "$AVI_MODEL" --tp 2
for mode in baseline optimized graph; do
 python tools/benchmark_worker.py --model "$AVI_MODEL" --request "$AVI_REQUEST" --mode "$mode" \
  --concurrency 1 --requests "${BENCH_REQUESTS:-5}" --warmup 1 --prefill-chunk "${PREFILL_CHUNK:-128}" --out "$run/$mode-c1.json"
done
python tools/benchmark_worker.py --model "$AVI_MODEL" --request "$AVI_REQUEST" --mode optimized \
 --concurrency 2 --requests "${BENCH_REQUESTS:-5}" --warmup 2 --prefill-chunk "${PREFILL_CHUNK:-128}" --out "$run/optimized-c2.json"
python tools/summarize_benchmarks.py "$run"
echo "Benchmark reports saved in $run; cache disabled, no trace or weight conversion."
