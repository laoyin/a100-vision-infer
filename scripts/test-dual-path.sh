#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
: "${HF_MODEL:?Set complete merged HF checkpoint}"
: "${AVI_BODY:?Set the real business OpenAI request JSON with embedded image data}"
previous="${1:?Pass previous native-mtp artifact directory}"
run="${2:-dual-path-test-$(date +%Y%m%d-%H%M%S)}"
test -f "$AVI_BODY"
for file in fp8-mtp-tp2/manifest.json request/request.json; do test -f "$previous/$file"; done
python - "$AVI_BODY" "${MAX_TOKENS:-8192}" "${MAX_CONTEXT:-20480}" "${MIN_OUTPUT_TOKENS:-1024}" <<'PY'
import json,sys
body=json.load(open(sys.argv[1],encoding='utf-8'))
tokens,context,minimum=map(int,sys.argv[2:])
if not 0<minimum<=tokens<context:raise SystemExit('Require 0 < MIN_OUTPUT_TOKENS <= MAX_TOKENS < MAX_CONTEXT')
if not body.get('messages'):raise SystemExit('Business body needs messages')
if body.get('tools') or body.get('response_format'):raise SystemExit('This comparison covers prompted JSON, not constrained decoding/tools')
for m in body['messages']:
 for p in m['content'] if isinstance(m.get('content'),list) else []:
  if p.get('type')=='image_url':
   url=p['image_url'];url=url['url'] if isinstance(url,dict) else url
   if not url.startswith('data:image/'):raise SystemExit('Use embedded business images')
PY
mkdir "$run"
run="$(cd "$run" && pwd)"
exec > >(tee "$run/test.log") 2>&1
stage=build
trap 'code=$?; if (( code != 0 )); then echo "FAILED: stage=$stage exit=$code; retain $run"; fi' EXIT
export CUDA_VISIBLE_DEVICES="${AVI_GPUS:-2,3}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
if [[ $(id -u) == 0 ]]; then export OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1; fi
# This suite exercises native CUDA; no optional TileLang installation or tuning.
unset AVI_TILELANG_DIR
git rev-parse HEAD
python - "$AVI_BODY" "$previous/fp8-mtp-tp2/manifest.json" "$run/workload.json" "${MAX_TOKENS:-8192}" "${MAX_CONTEXT:-20480}" "${MIN_OUTPUT_TOKENS:-1024}" <<'PY'
import hashlib,json,sys,subprocess
from pathlib import Path
body,model=map(Path,sys.argv[1:3])
Path(sys.argv[3]).write_text(json.dumps(dict(body=str(body.resolve()),body_sha256=hashlib.sha256(body.read_bytes()).hexdigest(),
 artifact_manifest_sha256=hashlib.sha256(model.read_bytes()).hexdigest(),revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
 max_new_tokens=int(sys.argv[4]),max_context=int(sys.argv[5]),min_output_tokens=int(sys.argv[6])),indent=2))
PY
echo '[1/4] Build, GPU oracle tests and TP2 candidate merge'
bash scripts/build.sh
mpirun -np 2 build/avi-tp-reduce-test
stage=smoke
echo '[2/4] TP1/TP2 output equality and Graph tests'
bash scripts/smoke-ampere-tiled.sh "$run/small-model"
failed=0
for workload in first-token long-json; do
 stage="$workload"
 tokens=1
 quality=()
 if [[ "$workload" == long-json ]]; then
  tokens="${MAX_TOKENS:-8192}"
  quality=(--min-output-tokens "${MIN_OUTPUT_TOKENS:-1024}" --require-complete-json)
 fi
 echo "[3/4] $workload: fresh vLLM comparison, budget=$tokens"
 python -u tools/benchmark_mtp_upstream.py --model "$HF_MODEL" --body "$AVI_BODY" \
  --out "$run/vllm-$workload" --requests "${BENCH_REQUESTS:-3}" --max-tokens "$tokens" \
  --max-context "${MAX_CONTEXT:-20480}" --max-pixels "${MAX_PIXELS:-4000000}" \
  --timeout "${PROFILE_TIMEOUT:-7200}" || failed=1
 if [[ ! -f "$run/vllm-$workload/mtp2.json" || ! -f "$run/vllm-$workload/mtp3.json" ]]; then
  echo "Missing upstream $workload reports; continuing other workload";failed=1;continue
 fi
 echo "[4/4] $workload: native ablations and strict quality checks"
 python -u tools/prefill_matrix.py --suite dual-path --previous "$previous" --hf-model "$HF_MODEL" \
  --body "$AVI_BODY" --max-new-tokens "$tokens" --max-context "${MAX_CONTEXT:-20480}" \
  --vllm-results "$run/vllm-$workload" --out "$run/matrix-$workload" \
  --requests "${BENCH_REQUESTS:-3}" --timeout "${PROFILE_TIMEOUT:-7200}" \
  --max-pixels "${MAX_PIXELS:-4000000}" "${quality[@]}" || failed=1
done
python tools/summarize_dual_path.py "$run"
echo "Retain $run: first-token completion proxy and full long-JSON results are separate."
echo 'No claim of speedup unless matching-output comparisons pass; JSON syntax alone does not establish business field accuracy.'
exit "$failed"
