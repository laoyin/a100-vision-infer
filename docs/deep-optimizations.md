# GDN, TP2 and frontend optimization round

The previous run measured about 1.22 s text prefill, 0.42 s vision and 0.46 s
frontend/serialization in its synchronized diagnostic. This round targets those
costs and verifies exact business output before attributing any speedup.

## Implementation

- `--gdn-cooperative`: four CUDA lanes cooperate on a GDN value column. The FP32
  recurrent state remains in registers, with K/4 values per thread and subgroup
  dot-product reductions. Supports K=16/128, value-column tails, and MTP per-token
  state trajectories. Floating-point summation order changes; not enabled by default.
- `--fused-gdn-conv`: depthwise convolution and SiLU in one CUDA kernel for
  prefill and MTP verification. Preserves the intermediate BF16 rounding boundary.
- `--bf16-tp-reduce`: NCCL BF16 sum for TP2 projection outputs removes two dtype
  conversions and halves collective payload relative to FP32. Restricted to TP1/2;
  TP4 is rejected because multiple BF16 partial sums can introduce extra rounding.
  A standalone two-rank test compares exact output against FP32 all-reduce followed
  by BF16 rounding before any benchmark starts.
- `--frontend-format vllm-string`: Qwen text/image messages follow the default
  vLLM 0.20.1 non-interleaved string convention, including newline separators.
  HF formatting stays available. This is a formatting choice, not token injection.
  Manual image placeholders are rejected in multipart content. Video/tools and
  vLLM interleaved-string mode are outside this compatibility mode.
- `--bf16-patches`: serialize already-rounded BF16 pixel patches to halve request
  file and host/device transfer bytes. Preserves the worker's previous F32-to-BF16
  input rounding. Does not change FP8 checkpoint codes or quantize activations to FP8.
- Benchmark-only `--frontend-threads 4 --spool-dir /dev/shm`: cap CPU preprocessing
  threads and use temporary shared-memory files. No image/response caching. Requires
  enough container shared memory; generated files are deleted after completion.

The server CLI exposes the kernel options, frontend format and BF16 patches.
The benchmark additionally tests CPU thread count and shared-memory transport.
No production service is restarted and no installed Python/CUDA files are changed.

## One round on the server

```bash
HF_MODEL=/path/to/complete/merged-checkpoint AVI_GPUS=2,3 \
bash scripts/test-deep-optimizations.sh native-mtp-20260929-150950
```

Uses the existing imported model and request. Builds, runs CUDA/CPU tests and TP2
reduction validation, then runs a fresh vLLM baseline/MTP suite and ten native
profiles (individual changes, combinations, MTP2, C2 and one diagnostic).

Look at `deep-test-*/matrix/summary.json`:

- Native output changes versus the newly aligned reference are failures.
- Cross-engine comparison requires full prompt equality, output equality and
  repeatable vLLM output. A missing ratio is not a speedup.
- `acceptance.faster_on_this_workload` only concerns this image's C1 latency.
  C2 is a native correctness/throughput test, not a matched vLLM C2 comparison.
- Synchronized diagnostics are excluded from latency comparisons.

Passing this matrix cannot establish universal superiority. A subsequent workload
suite must cover representative images, resolutions, output lengths, concurrency,
memory use and tail latency. Existing model imports and previous logs remain intact.

Reference for formatting:
[vLLM 0.20.1 chat_utils](https://github.com/vllm-project/vllm/blob/v0.20.1/vllm/entrypoints/chat_utils.py).
CUDA kernels in this round are independently implemented; no upstream kernels copied.
