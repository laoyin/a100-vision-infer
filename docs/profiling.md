# A100 TP2 首 token / 解码 profiling

这套工具定位性能瓶颈，不将 profiler 下的耗时当成速度成绩。正式比较继续使用 `scripts/test-dual-path.sh`。

## 运行

```bash
cd /opt/a100-vision-infer
git pull --ff-only

HF_MODEL=/app/model/qwen3.8-27b/training_new_2026_08_18-10lun/checkpoint-4900-merged \
AVI_BODY=/实际路径/business-request.json \
AVI_GPUS=2,3 \
bash scripts/profile-inference.sh native-mtp-20260929-150950
```

`AVI_BODY` 使用真实业务 messages 请求，图片为 data:image 内嵌数据；不设置则复用旧目录的 `request.json`。如果旧请求只能生成少量 token，解码采样会明确标记不足。不会改原始请求或模型。

默认重新编译、运行现有 CTest，然后针对同一图片/提示词运行四次独立模型会话：首 token × 参考/优化、512-token 解码样本 × 参考/优化。每次先预热一个请求，再打开采样，仅采一个测量请求。模型加载和预热不进入 CUDA capture。每个 MPI rank 单独运行 nsys，得到两个独立时间线；跨报告分析应保留原始时间信息，不能假定两个文件的相对零点完全相同。

首 token 样本预算为 1，用来定位预填充和首 token 的执行链。解码样本默认最多 512 token，包含该请求的预填充和随后解码，可能被预算截断；**不是完整长 JSON 质量验收，也不代表长上下文末尾性能**。自然 EOS 始终保留。修改 `PROFILE_TOKENS=1024` 可扩大样本，不建议一开始采完整几千/上万 token。

## 工具与环境

- `nsys` 必需。不存在则保存错误、退出，不会声称完成采样。
- `ncu` 默认可选：不存在记录 skipped；已经存在但执行失败会保留诊断并使最终退出非零。`PROFILE_NCU=off` 关闭，`PROFILE_NCU=required` 强制要求。
- 可用 `NSYS_BIN=/已有路径/nsys`、`NCU_BIN=/已有路径/ncu` 指定路径。
- 不安装 Python 包、工具，不修改 CUDA/驱动，不停止生产服务。默认只选择物理 GPU 2,3。
- `ncu` 若报告 `ERR_NVGPUCTRPERM`，表示当前账户/容器没有 GPU 性能计数器权限；保留日志交运维处理，脚本不会修改系统权限。
- `PROFILE_SKIP_BUILD=1` 仅用于已经编译当前提交的环境。旧 worker 缺少 profiler 协议会明确报错。
- `PROFILE_VARIANT=optimized` 仅采优化配置，减少服务器占用；默认 both 同时采参考与优化。
- `PROFILE_TIMEOUT` 默认每任务 1800 秒（nsys 任务额外留 600 秒供导出），不是强制采集时长。采集量主要由 token 预算限定。

## 采集内容

Nsight Systems 使用 CUDA API、CUDA Graph node、MPI、OS runtime 时间线，禁用 CPU sampling/context-switch 采集以减少权限要求。它跟踪两个原生 worker；Python 前端耗时在 benchmark.json 中单独记录，**没有 Python 调用栈**。不可把 worker 等待前端的空档当成 GPU kernel 变慢。

在已有 include 路径提供 `nvtx3/nvToolsExt.h` 时，还会标记 `vision.encode`、`text.step`、`gdn.attention`、线性层名称、`tp.reduce`、`mtp.round/draft/layer/verify_graph/candidates`。找不到 NVTX3 header 仍能编译并采 CUDA/MPI 数据，benchmark.json 的 `profiler.nvtx_enabled` 会为 false；不自动下载头文件。

不启用项目内 `--profile-kernels/--profile-stages`，避免它们的额外同步或关闭 Graph 改变待观察路径。Graph node tracing 本身也有开销，最终性能必须关 profiler 复测。

Nsight Compute 分别采 FP8 W8A16、GDN 块内求解、GDN 状态传播，每组最多两个匹配 kernel，导出计算/显存利用、launch、occupancy 数据。使用**独立单 GPU 算子 benchmark、固定代表形状**，不对 TP2/NCCL/Graph 内核做重放。这些不是自动从真实模型最慢 kernel 提取的样本；先看 nsys 再决定下一次针对哪些形状采样。

本脚本目前采原生引擎，不启动或采样线上 vLLM。已有双路径性能脚本负责同卡 vLLM 基准。不能把 ncu 的单算子指标直接换算成端到端或相对 vLLM 的提速。

## 日志

返回整个 `profile-test-时间/`，不要只发 summary：

- `summary.json`：工具路径、代码/模型/请求指纹，每个任务和两张卡是否有 kernel 数据、实际输出长度、参考/优化输出是否一致。
- `build.log`、`nsys-version.log`、`gpu-info.log`、`gpu-topology.log`。
- `first-token-reference/`、`first-token-optimized/`、`decode-sample-reference/`、`decode-sample-optimized/`：运行日志、完整参数、benchmark.json、nsys/rank-0 和 rank-1 的 `.nsys-rep`、SQLite 及统计 CSV。
- `ncu/`：各个 kernel 的 `.ncu-rep`、CSV、日志和命令。

`.nsys-rep` 可在本地 Nsight Systems GUI 打开；`.ncu-rep` 使用 Nsight Compute GUI。脚本会检查 SQLite 中确有 kernel 活动，空报告不算成功。缺少 NVTX 的统计表不影响 CUDA 数据有效性。

当前开发机没有 CUDA/Nsight，已做 CPU 协议/命令构造/空报告检测测试和语法检查；实际编译、注入、权限与采集兼容性仍需服务器验证。

参考：[Nsight Systems 用户指南](https://docs.nvidia.com/nsight-systems/UserGuide/index.html)、[Nsight Compute CLI](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html)。
