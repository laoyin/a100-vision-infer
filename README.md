# A100 Vision Infer

中文 | [English](README.en.md)

面向 **2 × NVIDIA A100-SXM4-80GB、TP=2** 与业务指定的 **Qwen3.8-27B LoRA 合并后 FP8 识图模型**的独立推理引擎。目标是在该固定硬件、模型和真实业务工作负载上超过 vLLM MTP。模型结构以 checkpoint 中的实际 `qwen3_5` config 为准。

项目参考 [ninfer](https://github.com/Neroued/ninfer) 的固定模型、离线权重布局、算子融合与显式状态管理思路。TP1/TP4 和小模型路径用于回归与诊断，不属于当前产品优化目标。完整边界与验收见 [固定目标](docs/target-contract.md)。

当前主要验证配置为 2 × A100-SXM4-80GB、TP=2、E4M3FN block-FP8（128 × 128）。权重保持 FP8，激活及 Tensor Core 计算使用 BF16，Gated DeltaNet recurrent state 使用 FP32。

> 📘 **交互式学习站点（GitHub Pages）**：<https://laoyin.github.io/a100-vision-infer/>
> 10 个可在浏览器直接打开的交互可视化页面，逐课讲透本引擎的核心机制；源码位于 [`learning/`](learning/)，推送后由 GitHub Actions 自动部署。

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

## 原生 MTP（投机解码）A100 实测

原生 MTP 已在 C++/CUDA 引擎内实现并完成 A100 验收：13 组原生配置全部运行成功，输出与原生基线一致，每组测试 5 次；vLLM baseline 进程退出码为 1，因此未生成跨引擎比较结果。2 × A100 80GB、TP=2，生成 128 tokens，P50：

| 原生配置 | 首 token 延迟 P50 | 总延迟 P50 |
| --- | ---: | ---: |
| 原生基线 | 3.075 s | 5.560 s |
| 权重解码缓存 | 2.077 s | 4.555 s |
| MTP2 + 权重缓存 | 2.089 s | 3.117 s |
| MTP3 + 权重缓存 | 2.101 s | 3.067 s |
| **MTP3 + 权重缓存 + 草稿 Graph** | **2.076 s** | **3.033 s** |
| 上述组合再加 GDN 分块 | 3.449 s | 4.404 s |

三个优化结论：

1. **权重缓存有效**：明显降低预填充耗时（TTFT 3.075→2.077 s），并让 MTP 批量验证获得收益。
2. **MTP 显著压低总延迟**：TTFT 基本不变的前提下，MTP3 + 权重缓存 + 草稿 Graph 把总延迟从 4.555 s 降到 3.033 s（约 −33%），为最优组合。
3. **GDN 分块预填充在当前形状下不划算**：叠加后回升到 3.449/4.404 s，故不作为默认路径。

实现与开关（FP8 MTP 导入、候选批量验证、GDN/卷积/KV 状态提交、草稿 CUDA Graph、TP 局部 argmax、短序列 GQA、权重缓存、GDN 分块）见 [原生 MTP 优化与一次性测试](docs/native-mtp.md)；原始数据在 `mtp-test/matrix/summary.json`。

## 尚未实现

- Paged KV cache 与公共前缀分页复用。
- MTP 随机采样、完整前缀缓存，以及多个请求的合批验证。
- MTP 主模型批量验证 CUDA Graph capture。
- KV cache 量化和跨节点 Tensor Parallel。

这些优化需要新的状态布局或独立数值验收，在没有 A100 实测数据前不会标记为已完成。

## 交互式学习站点

配套的可交互教学站点已部署到 GitHub Pages：<https://laoyin.github.io/a100-vision-infer/>。共 10 课，全部为自包含静态 HTML，无需服务器或联网；也可在本地直接打开 [`learning/index.html`](learning/index.html)：

1. [数据流](https://laoyin.github.io/a100-vision-infer/01-transformer-flow.html) —— 一次请求如何变成一串 token
2. [注意力](https://laoyin.github.io/a100-vision-infer/02-attention.html) —— QKV、softmax 与因果掩码
3. [KV Cache](https://laoyin.github.io/a100-vision-infer/03-kv-cache.html) —— 为什么 decode 是访存瓶颈
4. [RoPE / mRoPE](https://laoyin.github.io/a100-vision-infer/04-rope.html) —— 旋转位置编码与多模态扩展
5. [推理与采样](https://laoyin.github.io/a100-vision-infer/05-inference.html) —— prefill / decode 与贪心、温度采样
6. [张量并行](https://laoyin.github.io/a100-vision-infer/06-tensor-parallel.html) —— head 分片与 NCCL AllReduce
7. [FP8 量化](https://laoyin.github.io/a100-vision-infer/07-quantization.html) —— E4M3FN block-FP8 与 scale
8. [算子优化](https://laoyin.github.io/a100-vision-infer/08-optimized-cu.html) —— GEMV / WMMA / 融合 kernel
9. [混合注意力](https://laoyin.github.io/a100-vision-infer/09-hybrid-attention.html) —— Gated DeltaNet 线性注意力 + full attention
10. [投机解码](https://laoyin.github.io/a100-vision-infer/10-speculative.html) —— MTP 草稿、贪心验证与状态回滚

修改 `learning/` 下任意文件并推送到 `main`，[.github/workflows/pages.yml](.github/workflows/pages.yml) 会自动重新发布站点。

## 说明

项目针对固定模型结构和固定硬件优化，不以兼容全部 Transformers 模型为目标。Python 只用于 checkpoint 导入、请求预处理、结果解码和测试；视觉编码、语言模型 forward 与逐 token 生成均在 C++/CUDA 进程执行。
