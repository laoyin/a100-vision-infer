# A100 深度优化：共享 GEMV、GDN 融合和图池

本轮针对 graph-wy-test-20260930-174938：验证图仅减少约 0.33% 延迟，原 WY 延迟增加约 16.1%。这些数据不能证明超过 vLLM，新路径默认关闭，由同条件消融选择。

## 实现

- `--multi-token-gemv`：对 2..8 行的矩阵输入，一次加载权重为多 token 累加。支持 BF16 缓存矩阵和原始 FP8，四元素打包访存有对齐检查与标量尾部回退。单 token 与大块预填充保留原路径。
- `--multi-token-gemv-fp8`：小批量优先读取原始 FP8 权重，块缩放后按 BF16 边界舍入，再用 FP32 累加；减少权重读带宽。大块预填充仍可使用既有 BF16 解码缓存。没有重新量化模型。
- `--fused-gdn-prepare`：预填充融合卷积、SiLU、Q/K L2 归一化、head 扩展和门控；省去卷积输出及若干中间张量。第二个 kernel 更新卷积历史。保留卷积与激活的 BF16 舍入边界。
- `--gdn-wy-fused`：WY 的状态无关变换仍批量计算，但所有 chunk 的状态传递与输出改为一个持久 CUDA kernel；消除逐块 ATen matmul、copy 和 update 临时张量。仍需实测确认它优于逐 token 递推。
- `--mtp-verify-graph --reuse-verify-graph`：一个闲置会话图按精确 KV 容量复用，保留捕获的 KV/GDN 指针。新提示词清零 GDN 状态并覆盖有效 KV 前缀，其他容量淘汰闲置图。显式保留的状态和轨迹限制为 512 MiB，图私有工作区另需预留显存；超出限额不进入图池。
- 动态 GQA 的归并只遍历已使用的 KV 分区。
- `--audit-logits`：在 MTP 诊断行记录已提交前缀的候选分数。保持原验证 batch 形状，记录选择 ID 与 top-8；用于定位输出分歧，计时不参与性能排名。

## 今天服务器一次测试

```bash
cd /opt/a100-vision-infer
git pull --ff-only
HF_MODEL=/app/model/qwen3.8-27b/training_new_2026_08_18-10lun/checkpoint-4900-merged \
AVI_GPUS=2,3 \
bash scripts/test-native-fused.sh native-mtp-20260929-150950
```

复用之前导入的 FP8 TP2 模型和请求。不安装 Python 依赖，不修改模型或生产服务。默认 BUILD_JOBS=2、BENCH_REQUESTS=5、PROFILE_TIMEOUT=1800 秒。

脚本包括：编译和 CTest、TP 归约、TP1/TP2 小模型 MTP、图池的不同提示词/不同容量状态隔离、全新 vLLM 基线及 MTP1/2/3、十四组原生消融和分数诊断。C2 只报告原生吞吐，没有同并发 vLLM 对照。

## 查看结果

返回完整 `native-fused-test-时间/`：
- `test.log`：失败阶段和编译、回归结果。
- `matrix/summary.json`：各配置结果、图构建/复用/重放计数、严格验收和 failure_reasons。
- `logit-audit.json`：与 vLLM 基线的首次 token 分歧及原生候选分差。候选未进入 top-8 或首次分歧为首 token 时，分数可为空。
- `matrix/diagnostic.log`：完整原生候选分数；该分数不证明 vLLM 内部数值分歧的原因。

优化行必须与原生参考逐 token 一致。图池配置必须实际复用，验证图必须重放。跨引擎速度胜出仍要求输入一致、输出一致以及有效的 MTP2/3 对照。输出不一致不会放宽成通过。

本地仅具备 CPU 验证环境；CUDA 新内核的编译、数值、图池状态正确性及性能由上述服务器测试确认。GPU 测试覆盖奇数尾部、块尺度、非零卷积/递推状态、部分 WY chunk 和 GQA 前缀边界。
