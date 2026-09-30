#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
root="${1:?Supply a new tensor-GDN small-model directory}"
export CUDA_VISIBLE_DEVICES="${AVI_GPUS:-2,3}"
if [[ $(id -u) == 0 ]]; then export OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1; fi
python tools/make_tiny_fixture.py --out "$root" --block-fp8 --mtp --text-tokens 64
for tp in 1 2; do
 python tools/import_fp8.py --model "$root/model" --out "$root/tp$tp" --tp "$tp" --include-mtp
 for variant in reference tensor32 tensor64 prepared32 prepared64; do
  flags=()
  if [[ "$variant" != reference ]]; then
   flags+=(--gdn-tensor-prefill --gdn-tensor-chunk "${variant: -2}")
  fi
  if [[ "$variant" == prepared* ]]; then flags+=(--fused-gdn-prepare); fi
  mpirun -np "$tp" build/avi-infer --model "$root/tp$tp" --request "$root/request" \
   --output "$root/tp$tp-$variant.json" --prefill-chunk 128 --mtp-tokens 3 --tp-lm-head \
   --vector-gemv --extra-fusions --cublas-prefill --weight-cache-mib 64 \
   --mtp-draft-graph --mtp-verify-graph "${flags[@]}"
 done
 python - "$root" "$tp" <<'PY'
import json,sys
from pathlib import Path
root,tp=Path(sys.argv[1]),sys.argv[2]
ref=json.loads((root/f'tp{tp}-reference.json').read_text())
for name in ('tensor32','tensor64','prepared32','prepared64'):
    r=json.loads((root/f'tp{tp}-{name}.json').read_text())
    assert r['generated_ids']==ref['generated_ids'], (tp,name,'Token mismatch')
    assert r['finish_reason']==ref['finish_reason'], (tp,name,'Stop reason mismatch')
    assert r['cache']['gdn_tensor_calls']>0, (tp,name,'New kernel not exercised')
print(f'TP{tp} tensor GDN + MTP + verification graph strict output checks passed')
PY
done
