# A100 FP8 optimization batch

Reference design: [ninfer operator organization](https://github.com/Neroued/ninfer/tree/master/src/ops), particularly fused GDN gating and linear/activation operators. These kernels are independently implemented for this engine's A100 BF16 execution path. No throughput improvement is claimed until measured on A100.

The imported E4M3FN FP8 weights and scales remain unchanged. `--extra-fusions` enables GDN decay/beta preparation, RMSNorm + SiLU gating, and attention sigmoid gating. Intermediate BF16 rounding is preserved. Graph attention skips work on inactive KV partitions while retaining fixed launch shapes. `--cublas-prefill` routes FP8 matrices with more than eight input rows through bounded dequantization (2048 output rows at a time) and ATen/cuBLAS GEMM; decode still uses FP8 GEMV. This also applies to large vision projections. It trades temporary memory and launches for vendor GEMM efficiency; it may be slower on small matrices.

Defaults retain the previous compute selection. New flags work in both avi-infer and avi-worker. No dependencies are installed, weights converted, or server CUDA settings changed.

```bash
git pull --ff-only
AVI_MODEL=model-test-20260928-151515/fp8-tp2 \
AVI_REQUEST=model-test-20260928-151515/request \
AVI_GPUS=2,3 bash scripts/test-optimizations.sh
```

The script builds once and runs all GPU kernel tests, then nine profiles: baseline, existing optimized, fusions, cuBLAS, combined, combined Graph, combined chunks 256/512, and combined concurrency 2. Each profile first writes traces and compares against baseline, then runs an untraced resident benchmark with caches disabled. Compilation/kernel-test failure stops the batch. Profile failures/timeouts are logged and subsequent profiles continue; baseline failure stops comparisons. Any profile failure yields nonzero final exit status, with partial summary retained.

Default: three measured requests, one warmup (two for concurrency 2), 1800-second timeout per trace or worker phase. Set BENCH_REQUESTS and PROFILE_TIMEOUT to override. Nine model loads for traces plus nine resident worker loads can take substantial time; progress is printed and per-profile logs are written immediately. Shutdown time is bounded. The prepared request's output budget is reused unchanged.

Return `optimizations-*/test.log`, `matrix/summary.json`, and failing profile logs. Summary includes TTFT, latency, throughput, native numerical checks and token equality. Native trace thresholds are cosine >= .99 and relative RMSE <= .15, with decode comparison stopping when token histories diverge. These are regression checks, not original-runtime or business-quality validation. Ranking does not automatically change defaults; compare concurrency levels separately. Full token equality remains a separate field.

Not implemented by this batch: chunk-parallel GDN prefill, shared paged KV allocation, speculative/MTP decoding, or multi-request CUDA Graph capture. Those require additional state-management and numerical validation work.
