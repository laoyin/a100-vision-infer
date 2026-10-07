# 首 token 与长 JSON 双路径优化

固定目标仍为 2×A100-SXM4-80GB、TP=2、现有 qwen3_5 结构 27B LoRA 合并 FP8 权重。实际业务为识图后生成长 JSON；历史短输出样例仅用于功能验收，不能代表业务性能。

## 本轮实现

- `--fused-residual-norm`：目标模型每层及 MTP 层的 attention 输出残差相加与后置 RMSNorm 合成一个 kernel。保留 BF16 残差舍入、原归约次序和 one-centered norm 语义，同时覆盖预填充、草稿和验证。尚未融合通信或后一层的残差。
- `--gpu-candidates`：词表按 4096 项并行扫描，直接从 BF16 logits 生成局部最大值/全局 token ID，省去全词表 FP32 副本及多次转换；NCCL AllGather 后在 GPU 归并，去掉该段 rank-0 CPU 比较和 MPI 广播。相同分数选较小 ID，NaN/无穷最大值报错。仍有每轮 token 的小量 D2H、主机接受判断和状态提交，**不是完整 GPU MTP**。
- FP8 软件解码统一为精确位操作；正常数、次正规数、正负零和 E4M3FN NaN 均覆盖。用于普通/向量 GEMV、共享多 token GEMV、WMMA 和显式解码，保留 block128 scale 和 BF16 舍入；不重新量化模型。
- 联合对照上轮 GDN fused32/64、prefill chunk512/2048、FP8 Tensor Core 验证和 MTP2/3，分别测量首 token 路径与长输出。

新开关默认关闭；FP8 解码等价替换不另加开关。没有声称已超过 vLLM。此开发机没有 Torch/CUDA/A100，新 CUDA 编译、数值与速度必须在服务器验证。

## 测试命令

`AVI_BODY` 必须是真实业务的 OpenAI messages JSON，图片使用 data:image/...;base64,... 内嵌数据。仅支持提示词要求输出 JSON，本轮不比较 tools/response_format 约束解码。请求不会被脚本修改。

```bash
cd /opt/a100-vision-infer
git pull --ff-only

HF_MODEL=/app/model/qwen3.8-27b/training_new_2026_08_18-10lun/checkpoint-4900-merged \
AVI_BODY=/实际路径/business-request.json \
AVI_GPUS=2,3 \
MAX_TOKENS=8192 \
MAX_CONTEXT=20480 \
MIN_OUTPUT_TOKENS=1024 \
bash scripts/test-dual-path.sh native-mtp-20260929-150950
```

`MAX_TOKENS` 是生成上限，不会强制生成这么多；保留自然 EOS。根据真实业务设定上限及最低有效长度。输入 token 加输出预算必须落入 `MAX_CONTEXT`，越界会报错，不截断图片或提示词。不安装依赖、不改变服务器 CUDA、模型或生产服务。复用已有 FP8 artifact，仅重新预处理业务请求。

默认每 profile 测 3 次（另有预热），`BENCH_REQUESTS` 可调整，`PROFILE_TIMEOUT` 默认 7200 秒。每组独立启动模型，包含同卡新跑的 vLLM 基准；模型加载/预热不计入延迟，但测试总墙钟时间可能较长。本脚本不要求 TileLang；其单独验收继续使用 test-ampere-tiled.sh。

## 两组指标与验收

1. **first-token**：同一真实图片及提示词，仅生成一个 token。两引擎比较包含预处理的单 token 请求完成延迟，用于隔离首 token 路径。它包含完成处理开销，**不是 vLLM 的流式 TTFT**；原生真实首 token 事件仍单独记录。
2. **long-json**：同一输入完整自然生成。报告实际 token 数、完整请求耗时、原生首 token 后速率、MTP 统计、JSON 可解析性和自然结束。达到预算后截断、输出过短、无效 JSON 不允许计为成功加速。JSON 语法验证不能替代字段准确率验收。

严格比较相同 prompt token 和输出 token（保留已有 EOS 规范化）。每个 profile 必须同时匹配 vLLM MTP2/3 才报告相对最快 vLLM 的速度比。顶层汇总只将**同一 profile**在两组均更快视为“双路径更快”，不能挑两个不同配置的最佳值拼接。

GPU 测试覆盖 BF16 残差、RMS 归约、词表分块尾部/同分/非有限值、TP2 候选交换和输入变化后的 Graph 重放；小模型另外验证 TP1/TP2 输出一致性。实际业务测试仍是必要验收。

## 返回日志

整个 `dual-path-test-时间/` 目录：

- `test.log`：编译、阶段与错误。
- `workload.json`：业务请求/模型 manifest 指纹、代码版本和长度配置。
- `matrix-first-token/summary.json`：首 token 路径比较。
- `matrix-long-json/summary.json`：长 JSON 延迟、完整性、长度及原生解码指标。
- `vllm-first-token/`、`vllm-long-json/`：上游基准及 MTP 数据。
- `summary.json`：同一 profile 的双路径结果；缺失/不匹配不推算倍率。

尚未实施完整 GPU 接受/状态提交、FP8 Marlin 接入和 TP2 通信与 norm 融合。当前改动是可独立验收的一轮优化，不代表这些后续工作已经完成。
