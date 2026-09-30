# 固定 A100 / FP8 模型的性能研究与实施顺序

研究日期：2026-09-30。目标：2 × A100-SXM4-80GB、TP=2，现有 Qwen3.8-27B LoRA 合并 FP8 识图模型。模型结构以实际 qwen3_5 config 为准。本文件是研究结论和待实施计划，不代表下列内核已经接入或通过服务器验证。

## 1. 结论与实测依据

下一轮应优先替换 GDN 预填充和小矩阵计算的核心实现，再优化 MTP 整轮的同步、状态提交和 TP2 通信。保留当前 FP8 权重、BF16 激活和 FP32 recurrent state。固定模型优化的优势来自确定的形状、布局、执行顺序与业务输入分布；同样需要与成熟内核比较，不能仅凭 C++ 或专用引擎的定位认定更快。

最近已测代码是 098d5db；graph-wy-test-20260930-174938 的单图结果：

| 指标 | 实测 |
|---|---|
| 原生 verify-graph P50 | 2.834706 秒 |
| vLLM MTP3 P50 | 1.948760 秒 |
| 原生 TTFT | 约 1.949 秒 |
| 验证图相对原生 reference | 延迟降低约 0.33% |
| 原 WY 相对 reference | 延迟增加约 16.1% |
| 原生 / vLLM MTP3 draft 接受率 | 均约 57.3% |
| 输出 | 原生坐标 24，vLLM 为 23，尚未通过跨引擎一致性验收 |

原生需降低约 31.3% 的总延迟才能追平该次 vLLM 耗时。这只是差距计算，不是性能承诺，也不是质量合格后的速度比。

诊断请求的 frontend、vision、text prefill 分别约 0.375、0.392、1.202 秒；不是所有请求的固定耗时。gdn.scan、linear.text、tp.reduce 的 CUDA event 区间约 500、937、158 毫秒，但这些区间覆盖整次请求，可能包含 CPU 提交空隙，不能当成独占 GPU 活跃时间直接相加。

vLLM 的 mtp3.log 已明确记录：
- Selected MarlinFP8ScaledMMLinearKernel；
- Using Triton/FLA GDN prefill kernel；
- 文本及 ViT 使用 FLASH_ATTN；
- CUDA Graph 为 FULL_AND_PIECEWISE，量化 fp8，kv_cache_dtype=auto。

因此对手已有成熟的矩阵、GDN 和 Attention 内核。当前比较关闭 prefix/MM 缓存，KV dtype 与线上显式 fp8 配置也不同；冷请求比较和线上配置比较必须分别记录。

## 2. 优先级

| 顺序 | 工作 | 主要影响 | 当前缺口 | 成本 |
|---|---|---|---|---|
| P0 | 冻结业务数据与数值基准，采集 Nsight 时间线和关键矩阵形状 | 决定投入是否有效 | 单图、输出分歧；event 区间不是纯内核时间 | 中 |
| P1 | 引入 FLA 级别的 GDN chunk prefill，针对 K=V=128、HK=8/HV=24 固定优化 | TTFT / text prefill | WY 仍使用 ATen FP32 矩阵计算及三角求解 | 高 |
| P1 | 接入 FP8 Marlin W8A16，并按实际 M/N/K 与 BF16 cache/cuBLAS 比较 | MTP verify / decode / 部分 prefill | 自定义小 M 路径还未与 Marlin 同形状比较 | 高 |
| P2 | GPU 上完成候选归并、greedy 决策、接受长度及状态提交 | MTP 整轮延迟 | 候选回 CPU、MPI 广播和逐层 copy | 高 |
| P2 | ReplaySSM：记录短窗口输入，再批量重放已接受前缀 | MTP 状态流量 / 图池容量 | 逐位置保存完整 FP32 recurrent trajectory | 高 |
| P2 | TP2 小消息 one-shot all-reduce，融合 residual/RMSNorm | decode / verify 延迟 | 目前主要依赖 NCCL 与独立后处理 | 高 |
| P3 | 图像前端、视觉形状桶、固定执行计划和共享工作区 | 端到端 TTFT / CPU 空隙 | Python 文件 IPC、热路径查表、多次 ATen 调用 | 中至高 |
| 并发阶段 | 跨请求 MTP 批量 draft / verify / commit | C2 吞吐 | 当前逐会话 speculate；不是批量 MTP | 高 |

P1 是当前短输出识图请求的第一投入；长输出场景应提高 MTP 整轮的优先级。排序是结合当前测量的工程判断，不是论文给出的收益保证。

## 3. GDN：最直接的预填充缺口

本地 src/gdn_chunk.cpp 仍构造全局中间张量、转换 FP32、执行 ATen matmul 和 linalg_solve_triangular。最新的持久 WY kernel 只替换后段状态传播，没有把整条路径变成 Tensor Core chunk 实现。

FLA 的当前实现采用 chunk 流程，64-token 路径融合 KKT 与三角求解，随后生成 W/U、传播状态并计算输出；支持 grouped Q/K，不必预先物化 repeat_interleave。参见 [chunk_fwd.py](https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/gated_delta_rule/chunk_fwd.py)、[chunk.py](https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/gated_delta_rule/chunk.py)。

实施：
1. 先用服务器已有 Triton 环境建立同输入算子对照，锁定源码 commit 与依赖版本；不修改模型。
2. 固定 TP2 的 HK=8、HV=24、K=V=128；比较 chunk=32/64、tile、warps、寄存器与 spill。
3. 保留共享 Q/K head 索引，减少扩展和布局复制；预分配 W/U/输出工作区。
4. 选择现有 recurrent 路径处理小 T，chunk 路径处理长 prefill；用实测生成形状分派表。
5. 通过后再接入原生调用或移植必要 CUDA 内核，保留原路径作为数值参考。

注意：FLA 包含 BF16/TF32 等计算边界，数学等价不保证与现有 FP32 路径逐 bit 相同。必须比较 output、final FP32 state、分块续接与后续生成，并定位当前 24/23 分歧，不能直接降低验收标准。完整流水线也不是“所有步骤一个 kernel”。

理论依据：[Gated Delta Networks](https://arxiv.org/abs/2412.06464) 的硬件友好块并行算法。采用它的计算实现，不更换当前模型架构。

## 4. FP8 矩阵：复用成熟的 Ampere 路径

应参考 vLLM 的 FP8 分支，而不是直接使用原版 INT4 Marlin。[marlin_utils_fp8.py](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/quantization/utils/marlin_utils_fp8.py) 提供原 FP8 编码打包、块 scale 转换与 W8A16 调用；scale 精度和打包约定仍需核对。

实施：
- 独立导出所需 CUDA 内核与 pack/scale 规则，保留许可证和来源；不把 vLLM 服务框架作为原生运行时。
- 保留原始 FP8 codes，避免调用会重新量化权重的辅助函数。
- 测试实际 M=1、2、4、8 及业务 prefill 桶；N/K 来自固定 TP2 权重，不采用其他模型的推荐参数。
- 将 Marlin、当前共享 GEMV 和既有 BF16 解码缓存/cuBLAS 同输入比较；小 M 看带宽，大 M 看 Tensor Core 利用率。
- 按层角色和矩阵形状固定最优后端，记录显存占用与原始权重指纹。

现有引擎已经有 BF16 解码缓存。增加缓存开关本身不是新优化；应检查重复布局、生命周期及实际选中的后端。原版 [Marlin](https://github.com/IST-DASLab/marlin) 的 INT4 加速比不能套用到本项目 FP8。

## 5. MTP：从捕获某段图转为优化整轮

本地 src/speculative.cpp 的 gather_candidates 会把候选搬回 CPU，再做 MPI_Bcast；src/worker.cpp 也有 token 广播。完整优化应覆盖 draft、target verify、accept、commit 和发布，而不是只记录 graph replay 次数。

实施：
- 在 GPU 归并 TP2 候选，保持全词表合法 ID 范围与确定的 tie-break。
- 将 greedy 选择、连续匹配及 accepted length 留在 GPU；每轮只传必要的已提交 token 给 CPU。
- 使用稳定的 state/KV/metadata 地址，按真实 batch 和 MTP 窗口捕获并复用整轮可捕获的计算。
- 将逐层状态 copy 收敛到一次批量提交；EOS、预算、取消及 rejected suffix 使用同一提交边界。
- 当前 lm_head 已有局部候选归并；后续可评估全词表 head 与局部 argmax 融合，不能默认已有候选功能等于融合实现。

接受率在本次对照中相同，故不应先盲目增加 draft 长度。选择窗口要看每个已提交 token 的整轮成本。
论文参考：[Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192)。它说明验证机制，不保证本模型相对已经开启 MTP 的 vLLM 再获同等倍数。

## 6. ninfer 中值得新增借鉴的 ReplaySSM

ninfer 当前已有 MTP，并提供 [ReplaySSM 文档](https://github.com/Neroued/ninfer/blob/master/docs/maintainer/replayssm-gdn.md) 和 [实现入口](https://github.com/Neroued/ninfer/blob/master/src/ops/linear_attention/gated_delta_net/replay.cpp)：验证时保存驱动状态更新的紧凑输入，接受后仅重放已接受前缀，避免保存每个位置的完整 recurrent state。

根据本项目 TP2 尺寸自行计算：每 GPU 的一份 GDN state 为
48 × 24 × 128 × 128 × 4 bytes = 72 MiB。
MTP3 验证 T=4 时，仅 recurrent trajectory 即约 288 MiB。

若采用 BF16 K/V、FP32 gates、共享 Q/K head 和卷积列记录，则每 token 记录约 0.853 MiB，每 GPU 的 T=4 记录约 3.410 MiB；最终 checkpoint 仍需保留。这个估算依赖最终记录布局，尚不是本引擎实现占用。

值得评估的专用设计：接受长度已知后，将 48 层 × 24 value heads 的独立重放合成一个提交 kernel，增加并行宽度并减少逐层启动。

它用额外重放计算换取更少的状态写入。容量压缩不能直接推导速度提升。重放必须消费验证时的同一输入 bits，保持相同 FP32 运算顺序、舍入和归一化边界，并直接验证提交 state；文本偶然一致不足以证明状态正确。

## 7. TP2 通信与并发

参考 [vLLM custom_all_reduce.cuh](https://github.com/vllm-project/vllm/blob/main/csrc/custom_all_reduce.cuh) 的小消息路径，以及 [TensorRT-LLM 的 all-reduce/residual/RMSNorm 融合](https://nvidia.github.io/TensorRT-LLM/features/auto_deploy/transforms/post_load_fusion.html)。移植适用于 SM80 的实现，不能直接使用要求 Hopper/Blackwell 的新指令。

先确认 GPU 2、3 的 NVLink/P2P 拓扑，固定缓冲区与双缓冲序号。decode/短 verify 用小消息低延迟路径，大 prefill 保留 NCCL；阈值由测量决定。融合时保持 BF16 reduction、residual 和 norm 的数值边界，并验证图重放、不同会话和取消不会使两卡等待序号失配。

并发 MTP 应真正拼接不同请求的验证行，各请求独立维护 accepted length、conv、GDN state 和 KV frontier。[SGLang GDN backend](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/attention/linear/gdn_backend.py) 的状态池和验证/提交组织可作参考。当前串行 C2 结果不能代表它已实现，更不能证明超过同并发 vLLM。

## 8. 视觉与固定运行时

先拆分 frontend 中的图像解码、resize、normalize、tokenizer 和文件 IPC 时间。保留原 processor 的尺寸、插值、归一化与 patch 顺序；优先用常驻前端、共享内存/pinned staging、预分配缓冲减少数据交换。

视觉网络按实际分辨率和图片数建立有限形状桶，固定布局、工作区、权重指针，复用 CUDA Graph。文字侧也在初始化阶段绑定层参数，减少字符串、JSON 和权重映射查询。

[FlashAttention-2](https://arxiv.org/abs/2307.08691) 适用于 A100 的工作分配方法，但本项目已经有 flash_prefill/SDPA 路径。先核查实际选中 kernel、布局转换和 mask，不能把“再接一个 FlashAttention 开关”计作全新收益。

[FlashDecoding++](https://arxiv.org/abs/2311.01282) 的 flat GEMM、双缓冲和按形状选数据流值得借鉴；Attention decode 排在 48 层 GDN 和矩阵瓶颈之后，论文中的其他模型速度比不作为本项目预测。

## 9. 当前不作为主路线

| 路线 | 原因 |
|---|---|
| ninfer 公开速度直接作为目标值 | [公开测试](https://github.com/Neroued/ninfer/blob/master/docs/performance.md) 使用 RTX 5090 与 groupwise-int/NVFP4，硬件、权重和工作负载不同 |
| 直接接入 FlashInfer 当前 GDN/MTP kernel | [源码 API](https://github.com/flashinfer-ai/flashinfer/blob/main/flashinfer/gdn_decode.py) 明确要求 SM90；A100 为 SM80。状态组织思路仍可参考 |
| 原生 FP8 Tensor Core、NVFP4/TMA/WGMMA 专用快路径 | 不属于 A100 的执行能力 |
| INT4、稀疏化、裁剪视觉 token 或缩小图片 | 改变当前模型数值/输入语义，不满足本轮边界 |
| EAGLE/DFlash2 作为第一实现 | 需要新增匹配的 draft 权重或训练，先优化已有 MTP；不是当前 checkpoint 的直接替代 |
| 更换 GDN 架构或预训练新模型 | 不属于现有模型推理优化 |
| 继续叠加实验开关而不替换瓶颈 | 图/WY 实测已表明算子数量减少不等于端到端加速 |

ninfer 的 [固定 Program、工作区与稳定资源地址](https://github.com/Neroued/ninfer/blob/master/docs/maintainer/engine-architecture.md) 可借鉴；不需要照搬整个通用服务分层。

## 10. 下一轮交付与验收

第一轮：同形状算子对照 + FLA 级 GDN prefill + FP8 Marlin 分派。先缩短主要计算路径。
第二轮：GPU MTP 决策/批量提交 + ReplaySSM + TP2 小消息融合。缩短每个已提交 token 的成本。
第三轮：视觉/前端/固定执行计划 + 真正的跨请求 MTP。扩大到实际业务端到端和并发收益。

这是实施依赖顺序，可在同一代码交付中完成多个阶段，但不能省略各阶段消融。已提交 e0f0bd1 的共享 GEMV、GDN prepare、持久 WY 和图池仍需服务器验证；本研究不能代替该次测试。

验收使用相同模型、图片/模板/输出预算、采样、cache 语义与 GPU 拓扑；同时对照线上配置及调优后的 vLLM。报告 TTFT、decode、P50/P95、C1/C2 吞吐、显存和业务质量。当前严格一致性要求继续保留，输出分歧单独定位。

Nsight Systems 用于识别 CPU 空隙、同步和跨卡等待，Nsight Compute 仅抽样关键 kernel，检查带宽、Tensor Core、occupancy 和 spill。排除加载、首次编译、图捕获与 warmup；profiler 结果不混进性能排名。先采一条代表性请求和必要的小矩阵，不扩大服务器调试次数。

复用代码时记录上游 commit、版本、许可证与必要修改。这里链接 main/master 是研究入口，实施时必须固定版本；不能将当前上游行为自动视为服务器 vLLM 0.20.1 的全部行为。
