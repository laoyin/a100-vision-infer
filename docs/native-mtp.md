# Native A100 optimization candidate

This change implements MTP inside the C++/CUDA engine. It does not delegate native generation to vLLM. The separate vLLM runner remains a performance/quality reference. GPU compilation, numerical checks and speedups for this change are pending server validation; no claim of beating vLLM is made.

## Implemented paths

| Option | Behavior |
| --- | --- |
| import `--include-mtp` | Preserve original FP8 codes/scales and BF16 MTP parameters; partition MTP projections for TP1/2/4. Reject absent or incompatible weights. |
| `--mtp-tokens 1..5` | One-layer Qwen3.5-family MTP proposal, multi-token target verification, greedy acceptance and bonus token. |
| automatic state tracking | CUDA GDN scan saves each candidate's FP32 recurrent state. Commit accepted state and convolution history; truncate attention KV logically. No target replay. |
| automatic draft correction | Truncate hypothetical draft KV and recompute accepted draft positions with verified target hidden states. |
| `--mtp-draft-graph` | Capture single-token MTP layer with mutable position/KV offset; reuse graphs and captured KV allocations across equal-capacity sessions. Keep at most two idle capacity shapes. |
| automatic short-query GQA | Process up to eight causal queries against shared KV with a split-K CUDA kernel, without repeating KV heads or allocating a dense attention mask. |
| `--tp-lm-head` with MTP greedy | Exchange local maximum score/token pairs instead of full-vocabulary logits. Global ties choose the lowest token ID. Grammar/repetition constraints use complete logits. |
| `--weight-cache-mib N` | Bound a resident BF16 decoded-matrix cache for multi-token GEMMs; original FP8 codes remain loaded. Single-token FP8 GEMV remains available. Cache uses additional memory, not native A100 FP8 tensor-core compute. |
| `--gdn-chunk` | Optional FP32 blockwise delta recurrence with 32-token triangular solves, used for prefill chunks of at least 32 tokens. Must pass numerical/performance comparison before selection. |

Existing projection fusions, gates, vector FP8 GEMV, cuBLAS prefill and vocabulary sharding compose with these switches. Cache and chunked GDN are separate ablations: they are not assumed faster at every shape.

## Scope and state correctness

The supported native draft has `mtp_num_hidden_layers=1`, shared embeddings/head, full attention and the same text dimensions as the target. The importer never recovers missing weights from a different base model or modifies the HF checkpoint. Old native artifacts that skipped MTP need a new import.

Prefill supplies shifted image/text embeddings and preceding normalized target hidden states to the draft; chunk boundaries retain the last target hidden state. Each verification round consumes the already emitted pending token plus draft proposals. Target predictions are authoritative, including when proposals disagree. EOS, remaining output budget, TP agreement and convolution/GDN/KV commit lengths are handled explicitly.

Worker support includes multiple independent sessions, cancellation between rounds, greedy JSON grammar and repetition penalties. Stochastic sampling is rejected in MTP mode; use the ordinary worker for nonzero temperature. Whole-prompt prefix reuse is disabled in MTP mode because the old cache has no complete draft state. Image caching remains separately controlled. Multi-session MTP rounds currently execute one session at a time.

`--cuda-graph` is the existing target **single-token** graph and must not be combined with MTP. Use `--mtp-draft-graph` for MTP. Multi-token target verification currently runs eagerly. Full target verification graphs, paged/shared KV, KV quantization and batched multi-session verification are not implemented by this change.

## One server run

Use the existing server environment; no installation or CUDA change is performed:

```bash
git pull --ff-only
HF_MODEL=/path/to/complete-merged-fp8 \
TEST_IMAGE=/path/to/business-image.jpg \
AVI_GPUS=2,3 \
bash scripts/test-native-mtp.sh
```

Optional `TEST_PROMPT`, `BENCH_REQUESTS` (default 5), `MAX_NEW_TOKENS` (128), `MAX_PIXELS` (4000000), `WEIGHT_CACHE_MIB` (24576 per rank) and `PROFILE_TIMEOUT` (1800 seconds per native profile) control the run. The cache default spends up to 24 GiB **additional** GPU memory per rank to test the 80GB hardware tradeoff. Set it lower if the GPUs have less free memory.

The script audits, builds, runs CUDA/CPU C++ checks, runs small FP8/MTP models at TP1/2 (including graph reuse/concurrent sessions), imports a new real artifact, then runs 13 native ablations and vLLM baseline/MTP1/2/3. Small-model regressions stop before expensive real-model trials. Within the real-model matrix, failed profiles retain logs and later profiles continue.

Results are under `native-mtp-*/matrix/summary.json`. The summary records exact token differences, acceptance counts and per-profile latency. Cross-backend ratios are emitted only for identical prompt token IDs and matching output IDs (one terminal EOS may be removed for API representation). Ratio >1 means native was faster on that matched case, not on all images.

Both backends include CPU image processing/tokenization and output decoding, and exclude model load and warmup. Native additionally includes disk IPC. This is an offline end-to-end comparison, not a comparison to the production HTTP deployment. Native concurrency-2 rows are not compared against the sequential vLLM rows. Prefix/image caches are disabled. A single image repeated five times is a regression case, not business acceptance.

## Open-source references

Reviewed 2026-09-29:

- [NInfer MTP CLI/design](https://github.com/Neroued/ninfer/blob/master/docs/cli.md): fixed-model draft windows and explicit state management.
- [vLLM Qwen3.5 MTP](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/qwen3_5_mtp.py): shifted embedding/hidden fusion, single full-attention draft layer, normalization and shared output head.
- [SGLang speculative state handling](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/speculative/spec_utils.py): retain intermediate hybrid states and commit the accepted position.
- [Flash Linear Attention](https://github.com/fla-org/flash-linear-attention): chunk-oriented linear-attention optimization direction. This implementation uses a separately derived triangular-solve formulation, not FLA's Triton kernels.

No upstream implementation files are vendored in this change. Native code reuses the project's operators and LibTorch/CUDA/NCCL. Mathematical equivalence does not imply identical floating-point token decisions across kernels; the test suite retains mismatches instead of relaxing them.
