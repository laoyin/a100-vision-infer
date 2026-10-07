#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
root="${1:?Supply a new smoke-test directory}"
export CUDA_VISIBLE_DEVICES="${AVI_GPUS:-2,3}"
if [[ $(id -u) == 0 ]]; then export OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1; fi
python tools/make_tiny_fixture.py --out "$root" --block-fp8 --mtp --text-tokens 64
for tp in 1 2; do
 python tools/import_fp8.py --model "$root/model" --out "$root/tp$tp" --tp "$tp" --include-mtp
 variants=(reference residual candidates dual fused32 fused64 fp8s1 fp8s4 combined)
 if [[ -n "${AVI_TILELANG_DIR:-}" ]]; then variants+=(tile32 tile64 tilefp8s1 tilefp8s4 tilecombined); fi
 for variant in "${variants[@]}"; do
  flags=()
  case "$variant" in
   residual) flags=(--fused-residual-norm);;
   candidates) flags=(--gpu-candidates);;
   dual) flags=(--fused-residual-norm --gpu-candidates);;
   fused32) flags=(--gdn-fused-solve --gdn-tensor-chunk 32);;
   fused64) flags=(--gdn-fused-solve);;
   fp8s1) flags=(--fp8-tensor-small);;
   fp8s4) flags=(--fp8-tensor-small --fp8-tensor-split 4);;
   combined) flags=(--gdn-fused-solve --fp8-tensor-small --fp8-tensor-split 4);;
   tile32) flags=(--gdn-tilelang --gdn-tensor-chunk 32);;
   tile64) flags=(--gdn-tilelang);;
   tilefp8s1) flags=(--tilelang-fp8);;
   tilefp8s4) flags=(--tilelang-fp8 --fp8-tensor-split 4);;
   tilecombined) flags=(--gdn-tilelang --tilelang-fp8 --fp8-tensor-split 4);;
  esac
  if [[ "$variant" == tile* ]]; then flags+=(--tilelang-dir "$AVI_TILELANG_DIR"); fi
  echo "Small-model TP=$tp variant=$variant"
  mpirun -np "$tp" build/avi-infer --model "$root/tp$tp" --request "$root/request" \
   --output "$root/tp$tp-$variant.json" --prefill-chunk 128 --mtp-tokens 3 --tp-lm-head \
   --vector-gemv --extra-fusions --cublas-prefill --weight-cache-mib 64 --fused-gdn-prepare \
   --mtp-draft-graph --mtp-verify-graph "${flags[@]}"
 done
 python - "$root" "$tp" "${variants[@]}" <<'PYTHON'
import json,sys
from pathlib import Path
root,tp=Path(sys.argv[1]),sys.argv[2]
ref=json.loads((root/f'tp{tp}-reference.json').read_text())
for name in sys.argv[3:]:
    actual=json.loads((root/f'tp{tp}-{name}.json').read_text())
    if actual['generated_ids']!=ref['generated_ids'] or actual['finish_reason']!=ref['finish_reason']:
        raise SystemExit(f'TP{tp} {name}: strict output mismatch')
    needed=[]
    if name.startswith('fused') or name=='combined':needed.append('gdn_fused_calls')
    if name.startswith('fp8') or name=='combined':needed.append('fp8_tensor_calls')
    if name in ('tile32','tile64','tilecombined'):needed.append('gdn_tilelang_calls')
    if name.startswith('tilefp8') or name=='tilecombined':needed.append('tilelang_fp8_calls')
    for key in needed:
        if actual.get('cache',{}).get(key,0)<=0:
            raise SystemExit(f'TP{tp} {name}: {key} did not execute')
print(f'TP{tp}: strict output and new-kernel execution checks passed')
PYTHON
done
