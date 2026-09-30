# A100 prefill optimization experiment

This change targets the measured short-output vision workload. It does not claim
to beat vLLM before the new A100 run completes. Original FP8 codes are preserved.

- `--flash-prefill`: call the FlashAttention implementation already linked in
  PyTorch using native GQA and bottom-right causal alignment. Eliminates repeated
  KV heads and the explicit attention mask for text/MTP prefill. Uses BF16 on A100.
  Short speculative verification and single-token decode retain existing kernels.
- `--cache-vision-weights`: include quantized 2D vision matrices in the bounded
  decoded-weight cache, prioritizing draft and vision matrices before text.
  This caches weights, not image results. It may displace some text matrices;
  the matrix tests this tradeoff and a 32 GiB cache separately.
- `--profile-stages`: worker-only diagnostic option that synchronizes around vision
  and text prefill. Its latency is not used for speed claims. Other profiles retain
  asynchronous execution and report frontend time and original worker TTFT.

The FlashAttention integration uses PyTorch 2.11's internal ATen interface. It
requires the installed PyTorch build to contain CUDA FlashAttention. No package is
installed or modified. Unsupported builds fail explicitly rather than silently
claiming the optimized path ran.

Run:

```bash
HF_MODEL=/path/to/complete/merged-checkpoint AVI_GPUS=2,3 \
bash scripts/test-prefill-optimizations.sh \
 native-mtp-20260929-150950 upstream-mtp-retest-20260929-194207
```

Reuses imported FP8 weights, the image request and previous vLLM reports. Builds
and runs kernel tests, then six performance profiles and one diagnostic profile.
No weights are reimported. Larger chunks may change floating-point rounding;
output differences are failures, not silently accepted speedups.

The comparison records prompt-token differences and only produces a speed ratio
when inputs and outputs match and vLLM repeats consistently. Reused vLLM timings
are provisional: final acceptance also needs an interleaved, same-hardware-state
run on representative images and the required concurrency.

References: [PyTorch FlashAttention integration](https://github.com/pytorch/pytorch/blob/v2.11.0/aten/src/ATen/native/transformers/cuda/attention.cu),
[NInfer measurement conditions](https://github.com/Neroued/ninfer/blob/master/docs/performance/methodology.md).
No NInfer kernels were copied.
