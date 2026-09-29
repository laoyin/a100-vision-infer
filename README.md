# A100 Vision Infer

面向 **NVIDIA A100 80GB** 和 Qwen3.5/Qwen3.8 27B 视觉语言模型的独立 FP8 推理引擎。项目参考 [ninfer](https://github.com/Neroued/ninfer) 的固定模型、离线权重布局、算子融合与显式状态管理思路，针对 A100 SM80 和单机 TP=1/2/4 独立实现。

当前主要验证配置为 2 × A100-SXM4-80GB、TP=2、E4M3FN block-FP8（128 × 128）。权重保持 FP8，激活及 Tensor Core 计算使用 BF16，Gated DeltaNet recurrent state 使用 FP32。

## 优化目标

A100 没有新架构上的原生 FP8 Tensor Core 路径。本项目保留 FP8 压缩权重和 scale，按矩阵形状选择 BF16 Tensor Core、CUDA GEMV 或 cuBLAS，重点减少权重带宽、kernel launch、KV 复制和跨卡通信。

Prefill 优化视觉与长文本的大矩阵计算，目标是降低首 token 延迟；Decode 针对小 batch、单 token 计算，重点减少显存访问和 kernel launch。

## 已实现的优化

### FP8 权重与矩阵计算

- 直接导入 E4M3FN block-FP8 checkpoint，保留原始 FP8 编码与二维 scale，不重新量化。
- Attention、GDN 和 MLP 权重按 TP rank 预分片。
- Decode 使用自定义 FP8 GEMV，直接读取压缩权重并在寄存器累加。
- Prefill 提供自定义 BF16 WMMA，以及分块反量化后调用 cuBLAS 的可选路径。
- 反量化按输出行分块，不生成完整 BF16 模型副本。
- 视觉层及 GDN `a/b` 投影保留原 checkpoint 精度。

### 投影与激活融合

- 融合 Attention Q/K/V/gate 投影。
- 融合 MLP gate/up 投影和 SwiGLU。
- GDN 的 FP8 qkv/z 与 BF16 b/a 按精度分组，从四次投影减少为两次。
- 融合 RMSNorm、L2 normalization、mRoPE、GDN decay/beta。
- 融合 GDN RMSNorm + SiLU gate 和 Attention sigmoid gate。
- Decode 卷积状态更新与 SiLU 在同一 CUDA kernel 完成。

融合路径保留模型需要的 BF16 舍入边界，并通过 CUDA 数值测试与未融合路径比较。

### Attention、GDN 与状态

- GQA decode 直接使用共享 KV heads，不复制到全部 query heads。
- Split-K attention 分段计算 KV，再合并 softmax 统计量。
- CUDA Graph 使用固定容量 launch，空 KV 分段跳过计算。
- GDN Q/K normalization 使用融合 kernel；key dimension 16/128 使用专用展开 kernel。
- Decode state 按 value column 分块，在时间循环中保存在寄存器。
- recurrent state 使用 FP32，卷积历史原地更新。
- 每个请求独立保存 KV、卷积与 recurrent state；KV 容量按实际 token 预算分配。

### Tensor Parallel 与 Decode

- Attention/GDN 按 head 分片，MLP 使用 column/row parallel。
- 只在 row-parallel 输出处执行 NCCL AllReduce。
- TP ranks 共享采样结果，避免多卡状态分叉。
- 单 token decode 使用 FP8 GEMV、融合 gate、GQA 和 GDN 专用 kernel。
- 单请求支持 CUDA Graph replay；多请求可合并投影和 MLP 计算。
- 贪心采样一次传回最大值与 token ID，减少 CPU/GPU 同步。
- 可选词表输出头 TP 分片，各卡只计算部分词表，再汇总 logits，减少重复计算与权重驻留。
- 可选向量化 FP8 GEMV，每个线程一次加载四组权重和激活，减少加载指令与循环次数。

## 当前 A100 实测

输入 3913 tokens、1 张图片、生成 128 tokens，2 × A100 80GB、TP=2，缓存关闭：

| 路径 | TTFT P50 | 请求延迟 P50 | 首 token 后解码速度 |
| --- | ---: | ---: | ---: |
| baseline | 8.74 s | 29.21 s | 6.21 token/s |
| optimized | 22.13 s | 29.37 s | 17.53 token/s |
| CUDA Graph | 22.12 s | 29.18 s | 17.99 token/s |

Decode 相比 baseline 提升约 **2.8–2.9 倍**，但该轮首字阶段变慢，整次请求没有提速。首字阶段包含视觉处理和预填充，不能仅凭汇总计时确定某个算子是唯一原因。

后续同一请求的 cuBLAS + 融合 + chunk512 实测 TTFT 为 **2.73 秒**、请求延迟 **9.75 秒**，通过现有数值阈值；chunk128 在 decode 第 74 步超限。新加入的词表分片与向量化 GEMV 尚待服务器实测，不计入上述收益。

不同路径可能因 BF16/FP32 舍入顺序在后续 token 出现分歧。项目分别记录 CUDA 测试、trace cosine/RMSE 和 token 一致性；数值回归通过不等于识图业务准确率通过。

## 一次性验证全部优化

复用已导入 FP8 权重和请求，不安装依赖、不重新导入模型：

```bash
git pull --ff-only

AVI_MODEL=model-test-20260928-151515/fp8-tp2 \
AVI_REQUEST=model-test-20260928-151515/request \
AVI_GPUS=2,3 bash scripts/test-optimizations.sh
```

脚本编译一次，依次比较 baseline、现有 optimized、额外融合、cuBLAS prefill、融合 + cuBLAS、CUDA Graph、prefill chunk 128/256/512，以及单并发/并发 2。

结果保存在 `optimizations-*/matrix/summary.json`，包括 TTFT、延迟、吞吐、trace 数值差异和 token 一致性。

设置 `OPT_SUITE=deep` 可运行新一轮深度优化矩阵，包含词表分片、向量化 GEMV、chunk512/1024，以及基线预填充和逐层状态诊断。见 [测试说明](docs/deep-optimization.md)。

## 主要代码

- `src/engine.cpp`：模型执行、TP、Attention/GDN 状态和算子选择。
- `src/optimized.cu`：FP8 GEMV/WMMA、融合归一化、RoPE、GQA、GDN 和门控 kernel。
- `src/kernels.cu`：FP8 解码与基线 CUDA 实现。
- `tools/import_fp8.py`：block-FP8 checkpoint 校验、分片和导入。
- `tools/optimization_matrix.py`：正确性与性能矩阵。
- `tests/optimized_test.cpp`：CUDA kernel 数值回归。

## 原生 MTP 与进一步优化

新增原生 MTP 优化候选：FP8 MTP 导入、候选批量验证、GDN/卷积/KV 状态提交、草稿 CUDA Graph、TP 局部 argmax 通信、短序列 GQA，以及可选解码权重缓存和 GDN 分块预填充。代码已接入 C++/CUDA，新增路径尚待 A100 编译、正确性与性能验收。见 [原生 MTP 优化与一次性测试](docs/native-mtp.md)。

## 尚未实现

- Paged KV cache 与公共前缀分页复用。
- MTP 随机采样、完整前缀缓存，以及多个请求的合批验证。
- MTP 主模型批量验证 CUDA Graph capture。
- KV cache 量化和跨节点 Tensor Parallel。

这些优化需要新的状态布局或独立数值验收，在没有 A100 实测数据前不会标记为已完成。

## 说明

项目针对固定模型结构和固定硬件优化，不以兼容全部 Transformers 模型为目标。Python 只用于 checkpoint 导入、请求预处理、结果解码和测试；视觉编码、语言模型 forward 与逐 token 生成均在 C++/CUDA 进程执行。
