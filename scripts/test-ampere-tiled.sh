#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
: "${HF_MODEL:?Set the complete merged HF checkpoint}"
previous="${1:?Pass previous native-mtp run directory}"
run="${2:-ampere-tiled-test-$(date +%Y%m%d-%H%M%S)}"
mode="${AVI_TILELANG:-auto}"
case "$mode" in auto|required|off) ;; *) echo 'AVI_TILELANG must be auto, required or off'; exit 2;; esac
for file in fp8-mtp-tp2/manifest.json request/request.json request.json; do test -f "$previous/$file"; done
provided="${AVI_TILELANG_DIR:-}"
unset AVI_TILELANG_DIR
mkdir "$run"
run="$(cd "$run" && pwd)"
exec > >(tee "$run/test.log") 2>&1
stage=build
trap 'code=$?; if (( code != 0 )); then echo "FAILED: stage=$stage exit=$code; retain $run"; fi' EXIT
export CUDA_VISIBLE_DEVICES="${AVI_GPUS:-2,3}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
if [[ $(id -u) == 0 ]]; then export OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1; fi
git rev-parse HEAD
echo '[1/7] Build native CUDA backends and run CTest (no package installation)'
bash scripts/build.sh
stage=tp-reduction
mpirun -np 2 build/avi-tp-reduce-test
stage=tilelang
echo '[2/7] Optional TileLang compile, numerical qualification and offline tuning'
plugin=()
tilelang_failed=0
tilelang_status=disabled
if [[ "$mode" != off ]]; then
 if [[ -n "$provided" ]]; then
  kernels="$(cd "$provided" && pwd)"
  test -f "$kernels/manifest.json"
  plugin=(--tilelang-dir "$kernels")
  tilelang_status=precompiled
 else
  env_status=0
  python tools/check_tilelang_env.py --out "$run/tilelang-environment.json" || env_status=$?
  if (( env_status == 0 )); then
   kernels="$run/kernels"
   tune_flags=()
   if [[ "${TILELANG_QUICK:-0}" == 1 ]]; then tune_flags+=(--quick); fi
   if python -u tools/export_tilelang.py --out "$kernels" --repeats "${TUNE_REPEATS:-7}" "${tune_flags[@]}" 2>&1 | tee "$run/tilelang-export.log"; then
    plugin=(--tilelang-dir "$kernels")
    tilelang_status=exported
   else
    tilelang_status=export_failed
    tilelang_failed=1
    echo "TileLang export failed; native tests will continue. See $run/tilelang-export.log"
   fi
  elif (( env_status == 3 )) && [[ "$mode" == auto ]]; then
   tilelang_status=skipped_missing_package
   echo 'TileLang is absent: running native fused GDN/W8A16 only. No dependencies installed.'
  else
   tilelang_status=environment_failed
   tilelang_failed=1
   echo "TileLang environment check failed; native tests will continue."
  fi
 fi
fi
if (( ${#plugin[@]} )); then
 if AVI_TILELANG_DIR="$kernels" ctest --test-dir build -R '^cuda_kernels$' --output-on-failure 2>&1 | tee "$run/tilelang-ctest.log"; then
  export AVI_TILELANG_DIR="$kernels"
 else
  tilelang_failed=1
  tilelang_status=native_abi_or_numerical_failed
  plugin=()
  echo "TileLang native correctness failed; native tests will continue. See $run/tilelang-ctest.log"
 fi
fi
python - "$run/tilelang-status.json" "$tilelang_status" "$tilelang_failed" <<'PYTHON'
import json,sys
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps({'status':sys.argv[2],'failed':bool(int(sys.argv[3])),
 'installation_performed':False},indent=2))
PYTHON
stage=small-model
echo '[3/7] Small-model TP1/TP2, MTP and CUDA Graph checks'
bash scripts/smoke-ampere-tiled.sh "$run/small-model"
stage=operator-benchmarks
echo '[4/7] GDN and FP8 operator benchmarks'
build/avi-gdn-bench --repeats "${GDN_BENCH_REPEATS:-9}" "${plugin[@]}" > "$run/gdn-benchmark.json" 2> >(tee "$run/gdn-benchmark.log" >&2)
build/avi-linear-bench --repeats "${LINEAR_BENCH_REPEATS:-7}" --lm-head "${plugin[@]}" > "$run/linear-benchmark.json" 2> >(tee "$run/linear-benchmark.log" >&2)
limits=$(python - "$previous/request/request.json" <<'PYTHON'
import json,sys
r=json.load(open(sys.argv[1]))
print(int(r['max_new_tokens']),int(r['max_context']))
PYTHON
)
read -r tokens context <<< "$limits"
stage=vllm
echo '[5/7] Fresh vLLM baselines on the same test GPUs'
upstream_status=0
python -u tools/benchmark_mtp_upstream.py --model "$HF_MODEL" --body "$previous/request.json" \
 --out "$run/vllm" --requests "${BENCH_REQUESTS:-5}" --max-tokens "$tokens" --max-context "$context" \
 --max-pixels "${MAX_PIXELS:-4000000}" --timeout "${PROFILE_TIMEOUT:-1800}" || upstream_status=$?
test -f "$run/vllm/mtp2.json"
test -f "$run/vllm/mtp3.json"
stage=native-matrix
echo '[6/7] Native ablations and strict matched-output comparison'
matrix_status=0
python -u tools/prefill_matrix.py --suite ampere-tiled --previous "$previous" --hf-model "$HF_MODEL" \
 --vllm-results "$run/vllm" --out "$run/matrix" --requests "${BENCH_REQUESTS:-5}" \
 --timeout "${PROFILE_TIMEOUT:-1800}" --max-pixels "${MAX_PIXELS:-4000000}" "${plugin[@]}" || matrix_status=$?
stage=acceptance
echo "[7/7] TileLang=$tilelang_status; upstream_exit=$upstream_status; matrix_exit=$matrix_status"
echo "Retain the whole result directory: $run"
if (( tilelang_failed || upstream_status || matrix_status )); then exit 1; fi
echo 'Correctness passed. Read matrix/summary.json acceptance for measured performance; passing does not imply faster than vLLM.'
