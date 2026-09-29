# Reusing existing speculative decoding

Reviewed 2026-09-29. This document describes the upstream vLLM reference experiment, which delegates drafting, target verification and state commit to installed vLLM. The separate [native MTP candidate](native-mtp.md) now implements C++/CUDA speculation and has its own server validation suite.

## Sources and selection

- [NInfer CLI](https://github.com/Neroued/ninfer/blob/master/docs/cli.md): MTP with 1–5 proposals; Qwen3.8-27B also has DFlash2 with separate companion weights. Useful C++ execution/state design reference. Its published single-GPU RTX 5090 results do not establish A100 TP2 compatibility or speed.
- [SGLang hybrid speculative state handling](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/speculative/spec_utils.py): target verification retains intermediate convolution/SSM states, then commits the last accepted step. This is the principal reference for adapting our GDN state handling; KV truncation alone is insufficient.
- [vLLM Qwen3.5 MTP implementation](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/qwen3_5_mtp.py): existing model and vocabulary-head integration. Used as the executable A/B backend because the user's server already has a vLLM environment. The installed version and actual model/kernel compatibility must still be tested.
- [vLLM speculative decoding](https://docs.vllm.ai/en/latest/features/speculative_decoding/): use upstream verification instead of inventing acceptance/rejection rules.

The selected code paths are integrated with their runtime schedulers, attention backends and memory pools; they are not drop-in standalone C++ functions. Reusing the full backend for a reference experiment preserves upstream behavior. A future native port needs its own A100/TP tests and attribution for any copied implementation. Source references can change; record installed runtime versions with every experiment.

## Checkpoint prerequisite

The supplied model config declares `mtp_num_hidden_layers=1` and shared embeddings. This does not prove that the merged/quantized checkpoint contains its tensors. Our existing native import explicitly skips `mtp.*` and `model.mtp.*`.

```bash
python tools/inspect_mtp.py --model /models/your-merged-fp8 --require-mtp
```

This reads safetensors headers only, checks expected dense MTP names/shapes and shard extents, and does not allocate GPU memory or load tensor payloads. Eligibility means suitable for a runtime trial, not validated contents or accuracy. Missing/differently named MTP weights stop the experiment; no weights are silently taken from another base model. Fine-tuning the target without training the draft can lower acceptance even when the weights load correctly.

## One upstream A/B run

The audit also scans a separate `mtp.safetensors`, even when the original shard index omits it. Tensor records include their source file. Duplicate names or malformed headers fail explicitly. For an unindexed sidecar, the Linux runner creates `matrix/model-view` containing symlinks and a complete generated index; original checkpoint files are never edited. Keep the source checkpoint available while testing. This addresses file discovery, not installed vLLM architecture compatibility. Unknown MTP tensor naming still fails the audit rather than being guessed.

`test.log` now identifies the Git commit, audit/request/runtime stages and the failing stage with its exit code. A successful audit alone is not a completed test: require all four profiles in `matrix/summary.json` to pass. Audit failures preserve JSON diagnostics and print a concise error to the log.

```bash
git pull --ff-only
HF_MODEL=/models/your-merged-fp8 \
TEST_IMAGE=/data/test.png \
AVI_GPUS=2,3 bash scripts/test-speculative-upstream.sh
```

Use the original HF checkpoint, not native `fp8-tp2`. Optionally set REQUEST_BODY to a local OpenAI messages JSON with embedded images and TEST_PROMPT to your business prompt. The first experiment uses plain greedy output, thinking disabled, no grammar constraints, max_pixels=4,000,000 and 128 output tokens. MAX_PIXELS and MAX_NEW_TOKENS are explicit overrides. No packages are installed and model downloads are disabled. If vLLM is unavailable or the installed version does not support this exact MTP/FP8/A100 combination, stop and inspect the log; no automatic environment changes or precision fallback is performed.

Each baseline/MTP1/MTP2/MTP3 profile loads an isolated TP2 engine, warms up once, disables prefix caching, then measures three identical requests. A timeout terminates that experiment's process group. Metrics include latency, output throughput, generated token IDs/text, runtime version and speculative metric deltas when exposed by the installed vLLM API. Missing acceptance counters remain unavailable. Any output mismatch is recorded as a mismatch, not a pass. A failing baseline stops the trial; failing candidates retain logs and do not stop other candidates.

Reports are in `upstream-mtp-*/matrix`. Timings include upstream request preprocessing and are intended for comparison within this A/B run, not direct comparison with native prepared-input 7.28-second timing. Local CPU tests do not verify vLLM runtime or GPU execution.

## Native follow-on

After measuring draft acceptance and net latency, select MTP or a separately compatible DFlash/EAGLE draft. Adapt the upstream verification/commit approach to native KV, GDN and convolution storage; validate zero/partial/full acceptance, EOS and output budgets, TP agreement, then CUDA Graph. Do not label state-copy scaffolding as an accelerated speculative engine: real speedup requires efficient multi-token target verification.
