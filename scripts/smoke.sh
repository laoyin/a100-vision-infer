#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
root="${1:-smoke-run}"
python tools/make_tiny_fixture.py --out "$root"
for tp in 1 2; do
  for precision in bf16 fp8; do
    artifact="$root/$precision-tp$tp"
    python tools/convert.py --model "$root/model" --out "$artifact" --tp "$tp" --precision "$precision"
    python tools/preflight.py --model "$artifact" --tp "$tp"
    for mode in baseline optimized graph; do
      flags=()
      [[ "$mode" != baseline ]] || flags+=(--baseline)
      [[ "$mode" != graph ]] || flags+=(--cuda-graph)
      result="$root/result-$precision-tp$tp-$mode.json"
      mpirun -np "$tp" ./build/avi-infer --model "$artifact" --request "$root/request" --output "$result" --prefill-chunk 4 --trace "${flags[@]}"
      python tools/compare_reference.py --model "$root/model" --request "$root/request" --native-output "$result"
    done
  done
done
python tools/test_worker.py --model "$root/fp8-tp2" --request "$root/request" --tp 2
python tools/test_worker.py --model "$root/fp8-tp2" --request "$root/request" --tp 2 --cuda-graph

python tools/test_worker.py --model "$root/fp8-tp2" --request "$root/request" --tp 2 --host-cache
