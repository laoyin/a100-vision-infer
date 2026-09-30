# A100 GDN Tensor Core 预填充

本轮针对固定 Qwen3.8-27B FP8 模型的 K=V=128 GDN 预填充。目标是缩短 text prefill / TTFT。代码已实现，本地没有 nvcc、PyTorch/CUDA 或 A100；CUDA 编译、数值和性能尚待服务器验证。新路径默认关闭。

## 实现

入口：`--gdn-tensor-prefill --gdn-tensor-chunk 32|64`，默认 chunk=64。只替换至少 32 token 的预填充；decode 和 MTP target verification 继续使用原递推与状态轨迹路径。

新增 `src/gdn_tensor.cu`，每次扫描包含四个 CUDA kernel：

1. 分组 Q/K、V 和 gates 打包。Q/K 保持 key-head 数量，不生成三倍扩展输入；尾部 padding 的向量和 beta 为零，累计 gate 保持不变。
2. BF16 WMMA 计算 KKT 与 QKT，FP32 处理因果 gate。TP2 的 HK=8、HV=24 中，一份块内乘积供三个 value heads 复用。
3. 专用 unit-lower 三角求解，生成 W/U 与后续状态传播需要的变换；无需 ATen linalg_solve_triangular。
4. Tensor Core 计算 update、输出和跨块状态传播。同一 CTA 的 16 个 value columns 在整个扫描中保留 FP32 shared-memory state，消除逐块主机循环和 state copy。

这是基于 [FLA 块并行计算](https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/gated_delta_rule/chunk.py) 与 [Gated Delta Networks](https://arxiv.org/abs/2412.06464) 的原生 CUDA 实现，没有引入新 Python 依赖。KKT 与三角求解仍是两个 kernel，不声称已复制 FLA 的全部融合。原逐 token scan 为一个 kernel，新路径为四个；收益取决于矩阵并行效率和访存，不能只以 kernel 数判断。

FP8 codes/scales、模型权重与 BF16 激活来源不变。状态传播将 FP32 操作数分成两个 TF32 部分，计算四项乘积，以 FP32 累加并存储状态。它不是普通的单次 TF32 计算，也不保证逐 bit 等同原 FP32 递推。数值边界变化必须经过独立 oracle、状态和生成回归。

既有 `--fused-gdn-prepare` 增加分组输出模式，可直接交给新扫描，不再扩展 Q/K。原调用的默认输出模式保持兼容。新路径与图池、共享 FP8 GEMV 的组合单独消融，不修改部署默认值。

## 本地验证与服务器覆盖

本地 61 项 CPU 回归通过，含四项独立 GDN 数学/分解检查；Python 编译检查、Bash 语法检查通过。CPU 验证不等于 CUDA 内核运行验证。

服务器 CTest 增加：
- 独立 CPU scalar FP64 recurrence oracle，检查 BF16 output 与完整 FP32 final state；
- T=1/31/32/33/63/64/65/129/513，chunk=32/64；
- 非零初始状态、strided views、分组 heads 和不足一块的尾部；
- 在第 37 token 处分开扫描的续接；
- 实际 TP2 的 HK=8、HV=24、K=V=128，零 gate 和强衰减；
- CUDA Graph 重放以及修改输入缓冲后的结果；
- 分组 preparation 与原 preparation 的输入 bits / history 对照。

额外小模型使用 71-token 图文输入，保证新路径实际执行，并检查 TP1/TP2 下 MTP3 与验证图生成的 token 和停止原因严格一致。没有放宽现有验收。

## 一次服务器测试

复用之前已经导入的完整 FP8/MTP TP2 artifact 和业务请求：

```bash
cd /opt/a100-vision-infer
git pull --ff-only

HF_MODEL=/app/model/qwen3.8-27b/training_new_2026_08_18-10lun/checkpoint-4900-merged \
AVI_GPUS=2,3 \
bash scripts/test-gdn-tensor.sh native-mtp-20260929-150950
```

脚本不安装 Python 依赖，不修改驱动、CUDA toolkit、原始 checkpoint 或生产服务。复用现有 build.sh 的编译环境检测。默认 BUILD_JOBS=2、BENCH_REQUESTS=5、GDN_BENCH_REPEATS=9、PROFILE_TIMEOUT=1800。

执行顺序：编译/CTest → TP reduce → 原小模型 MTP 回归 → 新路径小模型回归 → GDN 算子微基准 → 全新 vLLM baseline/MTP1/2/3 → 八组原生消融。

微基准 `avi-gdn-bench` 在实际 TP2 的每卡 GDN 尺寸上比较 cooperative register scan 与 tensor chunk32/64，覆盖 T=512/2048/3914，包含非零 state 与数值检查。只测一层 scan，不能替代完整模型。CUDA event interval 可能包含主机提交间隙，同时记录 wall P50，不称为独占 GPU 活跃时间。

八组真实模型配置：reference、tensor32、tensor64、tensor+prepare、combined32、combined64、combined2048、diagnostic-tensor。组合使用已有共享 FP8 GEMV 与图池；诊断行不参与性能排名。测试顺序执行，未做交错 A/B。

每个测量请求的累计 `gdn_tensor_calls` 必须超过 warmup/上一请求计数，防止只有 warmup 执行新内核也被判为通过。图重放和复用计数继续验收。

## 返回结果

提供整个 `gdn-tensor-test-时间/` 目录，重点：

- `test.log`：失败阶段、编译和回归结果；
- `gdn-benchmark.json` / `gdn-benchmark.log`：各长度的数值检查及算子耗时；
- `matrix/summary.json`：真实模型 TTFT / latency、实际执行计数、严格 token 回归与 vLLM 对照；
- `matrix/diagnostic-tensor.json` / `.log`：阶段与 kernel 诊断。

算子基准数值失败返回 2；任何编译、小模型或测试失败保留目录并打印阶段。跨引擎坐标差异仍会使严格验收失败，不能为了输出速度比而跳过它。现有一次图文请求的测试也不能证明所有业务输入或并发均超过 vLLM。

## 本轮边界

本轮交付 GDN 预填充、分组乘积复用、算子微基准和完整测试入口。FP8 Marlin 接入、GPU MTP 决策、ReplaySSM、TP2 one-shot 通信和跨请求 MTP 仍按 [研究顺序](research-a100-fixed-model.md) 待实施；本轮不把它们列为已完成。
