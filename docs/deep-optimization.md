# Deep A100 optimization batch

This batch builds on the measured cuBLAS + fusion + chunk512 path (2.73 s TTFT, 9.75 s latency for the supplied 3913-token/128-output request). New paths are opt-in; no new speedup is claimed before A100 validation.

## Implementation

- `--tp-lm-head`: each rank retains a contiguous vocabulary slice of the output head, computes local logits, then NCCL AllGather assembles full logits. BF16 or FP8 head weights/scales are supported. The vocabulary must divide TP size. Batch layout is explicitly converted from rank/batch/vocabulary to batch/full-vocabulary. This reduces duplicated head computation and resident head memory, at the cost of a logits collective. Embeddings remain replicated. Existing model artifacts work unchanged; initialization still transiently loads the complete head before slicing it.
- `--vector-gemv`: four adjacent FP8 weights and BF16 inputs per lane, with integer exponent reconstruction for normal FP8 numbers. It preserves BF16 weight rounding. Odd widths or unaligned views fall back to the existing scalar kernel. FP32 reduction order changes, so this path needs independent numerical checks.
- `--reference-prefill`: uses the original unfused/baseline transformer prefill and the optimized decode. Original projection names reference slices of fused storage, without keeping another weight copy. This is a diagnostic and fallback candidate for the observed decode-74 discrepancy, not a claim that it fixes the discrepancy. Vision and final logits retain their selected execution path.
- `--layer-trace`: records the final prefill chunk and decode step 74 for every layer/rank: last-token hidden state, convolution state, and at most 4096 evenly spaced recurrent-state elements. It rejects CUDA Graph to avoid host I/O during capture. Traces are diagnostic runs only.

Comparison failures now retain machine-readable metrics for all steps with matching token histories. Existing cosine/RMSE thresholds remain unchanged. Layer comparisons skip decode-74 if the preceding generated histories differ.

## One server command

```bash
git pull --ff-only
OPT_SUITE=deep \
AVI_MODEL=model-test-20260928-151515/fp8-tp2 \
AVI_REQUEST=model-test-20260928-151515/request \
AVI_GPUS=2,3 bash scripts/test-optimizations.sh
```

Uses existing dependencies and real-model artifacts. Builds once, runs CUDA tests plus a generated tiny block-FP8 model for the new TP head, vector GEMV, Graph, and multi-request state paths, then runs ten real-model profiles. Tiny fixture import is only for the synthetic model, not the existing 27B checkpoint. The matrix retains chunk128 as a diagnostic control; it may reproduce the known failure and therefore cause a nonzero overall exit even when other profiles succeed.

The deep suite independently measures the two new kernels, their combination, Graph, concurrency 2 at chunk512, and chunk1024. Numerical failures skip that profile's benchmark and continue other profiles. Output: `optimizations-*/matrix/summary.json`, `*-comparison.json`, and `*-layers.json`; copy the whole matrix directory for detailed investigation. Passing a numerical check does not mean identical generated tokens or verified business quality. Do not interpret concurrency throughput ranking as single-request latency ranking.

Small-model checks cover different vocabulary partitions and worker batches. Local CPU tests cover report failure handling and layer comparison; actual CUDA compilation, NCCL/Graph execution and performance remain server validations.
