# A100 FP8：验证图与 WY 预填充优化

本轮针对前次实测的 CPU 调度、GDN 预填充和 MTP 验证开销。原始 FP8 权重保留，A100 计算使用 BF16，GDN 状态使用 FP32。

## 实现

- `--mtp-verify-graph`：捕获完整投机窗口的目标模型前向、词表投影和局部 argmax。GPU 上的动态 KV 偏移支持同一会话重放；短窗口和受约束选择回退到 eager。拒绝后按已接受位置恢复 GDN 状态和卷积缓存。图按会话创建，实测包含创建成本。
- `--mtp-draft-graph`：草稿图新增词表投影和局部 argmax，减少图外 launch。跨 TP 的候选汇总保持图外执行。
- `--gdn-wy`：以 32 token 分块，跨块、跨 head 批量计算状态无关的 WY 变换，再顺序传递 FP32 状态。预填充使用，解码和 MTP 验证仍保留逐步状态轨迹。
- `--profile-kernels`：记录 text/vision/MTP/head 线性计算、TP 归约及 GDN 扫描的 CUDA event 区间。此模式关闭 CUDA Graph，仅供 C1 诊断；区间可包含主机调度空隙，并非纯 kernel 活跃时间。诊断行不参与速度排名。

## 一次完整服务器测试

```bash
cd /opt/a100-vision-infer
git pull --ff-only
HF_MODEL=/app/model/qwen3.8-27b/training_new_2026_08_18-10lun/checkpoint-4900-merged \
AVI_GPUS=2,3 \
bash scripts/test-graph-wy.sh native-mtp-20260929-150950
```

复用之前导入的 `fp8-mtp-tp2` 和请求；不安装 Python 依赖，不修改模型文件。脚本编译、运行 CTest 和 TP 归约测试，检查 TP1/TP2 小模型投机窗口、验证图重放及 worker 会话回归，再运行新的 vLLM 对照和九组原生消融。

输出为 `graph-wy-test-时间/`。请保留完整目录，重点查看 `test.log`、`matrix/summary.json`、`vllm/summary.json` 和失败项的同名日志。`BUILD_JOBS` 默认 2；`BENCH_REQUESTS` 默认 5；`PROFILE_TIMEOUT` 默认 1800 秒。

## 验收边界

本地 51 项 CPU 回归通过，包含独立 NumPy WY 数学验证。GPU 测试包含 WY 与逐 token 递推的输出/末状态比较，以及动态 GQA 与 eager 的输出/缓存比较；必须在服务器运行后才能确认。真实请求要求优化行与原生参考逐 token 一致；验证图行必须有实际重放。

只有与无投机 vLLM 基线一致、重复输出稳定、输入 token 完全相同的对照才计算速度比。仍要求 MTP2 和 MTP3 均有效才能给出胜出结论，缺失或不一致不会算作加速。C2 尚无同并发 vLLM 对照，不作超越结论。本轮未实现跨会话批量 MTP 验证，不能承诺全面超越 vLLM。
