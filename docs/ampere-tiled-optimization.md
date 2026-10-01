# A100 固定模型：融合 GDN、FP8 W8A16 与 TileLang

本轮面向 2×A100-SXM4-80GB、TP=2、Qwen3.5 结构的 27B LoRA 合并 FP8 模型。原始 E4M3FN 编码及 block128 缩放保持不变，激活与矩阵乘使用 BF16，GDN 状态保持 FP32。所有新路径显式开启，服务器验收通过前不替换默认路径。

## 本轮实现

| 路径 | 实现 | 目的 |
|---|---|---|
| 原生融合 GDN | 同一 CUDA CTA 计算 KKT/QKT 和三角代入，组内 value heads 复用 Q/K 点积 | 省去 L 矩阵的全局显存读写，将上一轮 4 次 scan kernel launch 减为 3 次 |
| 原生 FP8 W8A16 | FP8 权重块在共享内存解码成 BF16，直接交给 SM80 WMMA，FP32 累加 | 覆盖 MTP 验证及其他 2～8 行线性层，避免整矩阵解码临时量 |
| Split-K | 1 或 4 个 K 分区，4 分区使用 FP32 中间值并归约 | 比较并行度收益与额外 launch/归约成本 |
| TileLang GDN | 专用 SM80 的融合块内点积与三角代入，保留已有状态传播 | 通过编译器选择矩阵布局，对照原生 CUDA |
| TileLang FP8 | 软件 E4M3FN 解码 + BF16 tiled GEMM，动态 N/K、2～8 行及 Split-K | 比较线程数、N 分块、流水线级数 |
| 离线调优与导出 | 数值验证后选择配置，导出独立 .so | worker 通过 C ABI 调用，请求期间不导入 Python/TileLang、不 JIT |
| 验收 | FP64 oracle、分块续算、部分尾块、Graph 重放、TP1/TP2、真实模型 | 检测数值偏差与端到端退化 |

GDN 生产特化为 HK=8、HV=24、K=V=128，chunk=32/64。另导出小模型与 oracle 所需的 1/3、1/2、2/4 头配置。

FP8 导入器已将 block128 缩放展开为按行 F32，新内核沿用该布局。只对 2～8 行、K 可被 128 整除的 FP8 线性层分派；TileLang 路径还要求 N 可被 16 整除。其他形状继续使用已有路径。单 token 解码、长预填充 GEMM 和已有 BF16 权重缓存不在这条新分派内。

GDN 的 FP32 状态传播沿用上一轮 TF32 高低位拆分计算，是有限精度实现，必须通过状态误差与最终输出验收。TileLang 不给 A100 增加原生 FP8 Tensor Core 能力。

## 一条命令测试

    cd /opt/a100-vision-infer
    git pull --ff-only

    HF_MODEL=/app/model/qwen3.8-27b/training_new_2026_08_18-10lun/checkpoint-4900-merged \
    AVI_GPUS=2,3 \
    bash scripts/test-ampere-tiled.sh native-mtp-20260929-150950

最后一个参数为已有导出目录，须包含 fp8-mtp-tp2/manifest.json、request/request.json、request.json。真实模型不重复导出，小模型在新结果目录内创建。

脚本顺序：原生编译与 CTest → TP 通信 → 可选 TileLang 编译/调优与原生 ABI 测试 → 小模型 MTP/Graph → GDN/线性层 benchmark → 同卡新跑 vLLM → 真实模型优化矩阵。

脚本只使用服务器已有 Python 包和 CUDA 工具链，不执行 pip/conda，不调整驱动/CUDA。

### TileLang 模式

- **AVI_TILELANG=auto**：默认。已有 TileLang ≥0.1.15 时编译；缺包则记录 skipped_missing_package，完成原生优化测试。
- **AVI_TILELANG=required**：必须完成 TileLang 验收。缺包或失败最终退出非零，仍尽可能收集原生结果。
- **AVI_TILELANG=off**：只测试本轮原生 CUDA 优化。
- **AVI_TILELANG_DIR=/absolute/path/to/kernels**：复用兼容环境中导出的模块，仍跑原生 ABI/数值检查，不必再次导入 TileLang。
- **TILELANG_QUICK=1**：每个特化只编译一个候选，仍做数值检查，用于首次编译排错。完整调优默认开启。

强制覆盖 TileLang：

    HF_MODEL=/path/to/complete-merged-checkpoint AVI_GPUS=2,3 AVI_TILELANG=required \
    bash scripts/test-ampere-tiled.sh native-mtp-20260929-150950

**当前开发机没有 Torch/CUDA/A100，新的 CUDA 和 TileLang 路径尚未在本地编译运行。** 编译、动态库加载和 GPU 数值结果以服务器验收为准。缺包时 auto 成功仅表示原生部分通过，不能视为 TileLang 已验证。

## 查看结果

默认结果目录：ampere-tiled-test-时间/。

| 文件 | 内容 |
|---|---|
| test.log | 当前阶段、错误位置 |
| tilelang-environment.json / tilelang-status.json | 实际版本、是否跳过或失败 |
| tilelang-export.log / kernels/tuning.json | 候选配置、数值检查、代表形状耗时、失败堆栈 |
| tilelang-ctest.log | C++ 直接加载导出库的数值与 Graph 测试 |
| gdn-benchmark.json | 寄存器扫描、上一轮 tensor-WY、原生融合、可选 TileLang 完整 GDN scan |
| linear-benchmark.json | shared FP8、缓存 BF16 GEMM、原生 W8A16、TileLang 固定模型矩阵对比 |
| matrix/summary.json | 每组 flags、首 token/总延迟、输出差异、与 vLLM 的可比结果 |

只有输入和输出 token 均匹配且 profile 通过，才生成跨引擎速度比。CUDA event interval 可能包含 CPU 提交间隙，不等同于纯 GPU 指令耗时或端到端收益。诊断 profile 排除在速度比较外。

Graph 捕获的调用不计为执行，重放累计图内新算子的调用数。矩阵逐请求检查计数相对预热和上一请求递增。失败 profile 保留日志；测试通过不代表超过 vLLM。

## 调优和导出边界

GDN 对 128/256 线程调优，以 512/2048 token 的 prepare 阶段计时。FP8 对 N tile=64/128、128/256 线程、1～3 级流水线候选调优，以 TP2 MLP 的 (N,K)=(17408,5120)、(5120,8704) 计时。按每个 rows/split 或 chunk/head 特化选择均值最小的有效候选，单算子调优不修改引擎默认值。

导出器核对真实 Cython host wrapper 的指针顺序、dtype、int32 动态维度和 stream 参数，再直接加载导出的库做独立数值测试。C++ 检查 manifest、库 SHA256、SM80 和初始化结果；模块保持加载至进程结束，确保 Graph 引用有效。导出库仍依赖兼容的 Linux/CUDA C++ 运行库，不是跨平台二进制。

本轮未接入 Marlin、替换整个 GDN 状态传播、实现 GPU 全流程 MTP 或多请求 MTP batching。这些仍为后续方向，当前优化收益待实测。

## 参考

- [TileLang](https://github.com/tile-ai/tilelang)：SM80 target、tiled GEMM、流水线与调优。
- [TileLang Cython wrapper](https://github.com/tile-ai/tilelang/blob/main/tilelang/jit/adapter/wrapper.py)：导出 ABI 检查依据。
- [Gated Delta Networks](https://arxiv.org/abs/2412.06464) 与 [FLA](https://github.com/fla-org/flash-linear-attention)：chunk/WY 算法参考。
- [DeepSeek TileKernels](https://github.com/deepseek-ai/TileKernels)：融合和访存组织参考；当前 NVIDIA 实现要求 SM90/SM100，本项目未直接搬用。
