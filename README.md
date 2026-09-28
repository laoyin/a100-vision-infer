# A100 Vision Infer — v0.3 实验版

参考 ninfer 的固定模型、离线转换、显式状态管理思路，面向 **2 × A100 80GB、TP=2、Qwen3.8-27B 微调模型**。

**已有原生模型执行代码；尚未在 A100 上编译或运行验证。已加入优化内核和常驻服务，但性能、正确性仍待服务器验收。先跑小模型检查，再测实际模型。**

## 本版实现

- C++ 模型执行和生成循环；LibTorch C++/ATen 提供 CUDA 张量、BF16 矩阵乘法与 SDPA。没有调用 Python 模型 forward/generate，没有封装 vLLM/SGLang。
- 原生视觉 patch projection、位置插值、视觉 Transformer 和 merger。
- 原生 Gated Attention、mRoPE、KV cache；自写 CUDA Gated DeltaNet 递归扫描及卷积状态管理。
- Attention/GDN 按 head 分片，MLP 列/行分片，NCCL 归约；TP=1/2/4，首要测试 TP=2。视觉塔、embedding 和输出头在各卡复制。
- 自有格式 FP8 E4M3FN、每输出行 FP32 scale；融合 FP8 GEMV / WMMA BF16 GEMM；可用 --baseline 回退分块解码对照。视觉权重保留 BF16，递归状态 FP32。支持 BF16 转换以作对照。
- 单图/多图、分块 prefill、常驻多请求调度、批量 decode、贪心/温度采样；离线 LoRA 合并、请求准备、文本解码工具。

Python 只用于转换、tokenizer/图片预处理、输出解码和独立验证。图像 encoder、语言模型和逐 token 生成都在 C++ 进程。LibTorch 是本版的 C++ 算子依赖，不是 Transformers 模型执行器。

## 当前服务器快捷测试（CUDA Toolkit 12.8）

适用于日志中 A100、nvcc 12.8、现有 PyTorch cu130 的服务器。保留系统驱动和 Toolkit，仅创建项目 `.venv-cu128`，安装 PyTorch 2.11.0+cu128 / torchvision 0.26.0+cu128。官方版本配对来源：https://pytorch.org/get-started/previous-versions/ 。不会安装系统包或修改现有训练环境。

```bash
git pull --ff-only
bash scripts/setup-cu128.sh
AVI_GPUS=2,3 bash scripts/test-server.sh
```

第一步需要联网下载依赖和足够磁盘空间；Python 必须支持 venv。缺少 OpenMPI/OpenSSL 开发包、CMake 或编译器时需要管理员先提供。测试脚本默认 GPU 2、3，应确认它们仍空闲；root 下自动设置 OpenMPI 所需环境变量。输出目录为带时间戳的 acceptance-*，保留 acceptance.log、pip-freeze.txt、commit.txt 和小模型结果。编译使用 --fresh 清除旧 CMake 配置，保留 build 目录文件。

小模型全部通过后，使用完整合并 BF16 模型测试真实图片：

```bash
HF_MODEL=/models/your-merged-bf16 TEST_IMAGE=/data/test.png AVI_GPUS=2,3 bash scripts/test-model.sh
```

这个步骤生成两份原生权重（BF16/FP8），需要充足磁盘空间；结果位于 model-test-*。参考模型对照默认单 GPU，极端图像可能超出其显存；此时保留日志，不要把 OOM 当作引擎数值通过。脚本只验收数值，业务字段仍需人工核对。

## 服务器运行步骤

在 Linux 服务器上传整个项目。需要 CUDA 版 PyTorch 及版本匹配的 torchvision（图片预处理使用）、匹配的 CUDA toolkit（包含 nvcc）、CMake >=3.24、G++、OpenMPI 开发包、NCCL 开发包。**驱动 580.126.09 不等于已经安装 CUDA 编译器。** 使用服务器现有 CUDA PyTorch 环境，或单独创建相同版本的环境。CPU-only PyTorch 不能构建。

Ubuntu 的常用系统依赖（NCCL 包通常需 NVIDIA 软件源）：

```bash
sudo apt-get install build-essential cmake libopenmpi-dev openmpi-bin libnccl-dev libssl-dev
python -m pip install -r requirements-tools.txt
python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.utils.cmake_prefix_path)'
nvcc --version
nvidia-smi topo -m
```

不要为了匹配本文随意升级训练环境。记录现有 PyTorch/CUDA 版本，优先在独立环境构建。

### 1. 编译与 CUDA 算子测试

```bash
bash scripts/build.sh
```

默认并行编译 2 个任务，避免头文件编译占满 CPU 内存。NCCL 在自定义位置时先设置 `NCCL_ROOT=/path/to/nccl`，目录下应有 include 与 lib/lib64。

### 2. 先跑小模型端到端检查

```bash
CUDA_VISIBLE_DEVICES=0,1 bash scripts/smoke.sh smoke-run-01
```

该脚本创建随机小型 Qwen3.5 同构模型和图片 patch 输入，运行 BF16/FP8、TP1/TP2、基线/优化/CUDA Graph 三种路径，并与 Transformers 的视觉特征和 prefill logits 比较。它不是准确率测试。任何检查失败都应先修复，不要直接跑 27B。输出目录必须是新目录。

MPI 默认按本机 rank 选择 GPU；每个进程要看到同一组 GPU。脚本针对普通 Linux 用户，不自动绕过 root 限制。

### 3. 合并你们的 LoRA（已有完整合并模型则跳过）

```bash
python tools/merge_lora.py \
  --base /models/Qwen3.8-27B \
  --adapter /models/your-adapter \
  --out /models/your-merged-bf16
```

CPU 合并需要充足 RAM，建议准备至少约 100GB 可用主存并监控峰值。检查生成的 `merge_audit.json`：你们视觉侧未冻结，不能遗漏视觉/aligner 的适配器。使用训练时完全一致的基础模型 revision 和 processor。合并脚本不替代合并前后业务等价性验证。

### 4. 转换原生权重

先留一份 BF16 原生包用于定位数值问题，再生成 FP8：

```bash
python tools/convert.py --model /models/your-merged-bf16 \
  --out /models/avi-bf16-tp2 --tp 2 --precision bf16

python tools/convert.py --model /models/your-merged-bf16 \
  --out /models/avi-fp8-tp2 --tp 2 --precision fp8
```

转换器逐张量工作，仍需几 GB 主存临时空间；磁盘要容纳按 rank 写出的模型包。输出目录不可已存在。只接受未量化的完整模型；不直接导入现有 W8A8/block-FP8 格式。视觉侧保留 BF16，MTP 显式跳过。

### 5. 准备识图请求

```bash
python tools/prepare_request.py \
  --model /models/your-merged-bf16 \
  --image /data/test.png \
  --prompt '请识别图片中的柜体、回路及关键字段，并输出 JSON。' \
  --max-pixels 4000000 --max-context 20480 --max-new-tokens 256 \
  --out /data/request-01
```

多图重复 `--image`。默认关闭 thinking 以便控制初始对照，可用 `--thinking` 打开。总预算包含图文输入和输出；不会把超过预算的内容静默截断。max_pixels 是预处理上限，不代表已验证的可用并发。

### 6. 双卡原生推理

先将下述模型路径换为 BF16 包跑对照，再用 FP8：

```bash
CUDA_VISIBLE_DEVICES=0,1 mpirun -np 2 ./build/avi-infer \
  --model /models/avi-fp8-tp2 \
  --request /data/request-01 \
  --output /data/result-01.json \
  --prefill-chunk 128 --trace

python tools/decode.py --model /models/your-merged-bf16 --result /data/result-01.json
```

输出必须是新文件。`--trace` 同时保存视觉特征和 prefill logits 的 FP32 二进制文件。仅限单机；TP4 需重新 `--tp 4` 转换并用 `mpirun -np 4` 启动。

### 7. 实际模型参考对照

```bash
python tools/compare_reference.py --model /models/your-merged-bf16 \
  --request /data/request-01 --native-output /data/result-01.json
```

参考对照默认单张 GPU 加载 BF16；大图/长序列可能需要更多显存或高效 FLA 依赖。先用小型真实图片检查，不修改正式业务的 400 万像素验收目标。当前阈值只是宽松数值 smoke 检查，不能证明计数、BOM、bbox 等业务质量，也不能替代逐 token/长序列验证。

## v0.2 服务与性能实现

- 融合 QKV、GDN 输入投影、MLP gate/up；融合 RMSNorm、SwiGLU、L2、mRoPE 和 decode 卷积。
- FP8 小批量 GEMV 直接读取压缩权重；prefill 使用 BF16 WMMA 分块。A100 上没有原生 FP8 Tensor Core 计算。
- GQA decode 使用 split-K 注意力，避免复制 GQA KV；GDN 扫描减少递归状态的全局显存访问。
- 常驻 C++ worker 保存模型，多请求轮询 prefill、合并 decode 投影和 MLP。每个请求有独立 KV/GDN 状态。
- 单请求 decode 可用 CUDA Graph；多请求使用批量执行路径。Graph 显式开启，默认不启用。
- 图像特征和完整 prompt 状态采用有容量上限的 LRU 缓存，内容哈希作为键。不是任意公共前缀匹配。
- 温度、top-k/top-p、重复惩罚、随机种子、取消、超时；主 rank 统一超时判定。
- HTTP/SSE、base64 图片输入、API key、JSON object 语法掩码。Python 仅处理 HTTP、tokenizer 和图片预处理。

启动服务（转换权重时需保留 tokenizer.json，旧包需重新转换以启用 JSON 语法）：

```bash
python -m pip install -r requirements-test.txt
CUDA_VISIBLE_DEVICES=0,1 python tools/serve.py \
  --model /models/avi-fp8-tp2 --hf-model /models/your-merged-bf16 \
  --tp 2 --max-concurrency 2 --max-context 20480 --prefill-chunk 128
```

默认仅监听 127.0.0.1；远程访问请显式设置 --host 和 AVI_API_KEY。接口为 `/v1/chat/completions`，图像使用 `image_url` 内容块中的 `data:image/png;base64,...`；支持 `stream: true` 和 `response_format: {"type":"json_object"}`。JSON 模式须关闭 thinking 和文本 stop。`avi_metrics.json_complete` 指示完成状态；达到长度上限或取消时可能只有 JSON 前缀，不能当作完整结果。语法不约束业务字段或值。

服务器完整小模型验收：

```bash
CUDA_VISIBLE_DEVICES=0,1 bash scripts/acceptance.sh acceptance-01
```

会保存环境、构建、数值检查、worker 集成测试日志。实际模型数值验收通过后，使用 `python tools/benchmark_http.py --help` 跑相同图片和 token 预算的并发压测。先固定并发 1/2；不能把训练 max_pixels 或显存容量直接换算成安全服务并发。

生成压测请求并执行：

```bash
python tools/make_http_request.py --image /data/test.png --prompt '识别图中设备并输出 JSON 对象' --json-object --out request-body.json
python tools/benchmark_http.py --body request-body.json --requests 20 --concurrency 2 --out benchmark-01.json
```

## v0.3 显存管理与验收补充

- 常驻 worker 启动后读取所有 rank 的可用显存，采用最小值，扣除 GPU 缓存上限与 workspace 预留，建立统一准入账本。显存不足的请求排队，单请求超出总预算直接报错。
- 每请求 KV 容量按输入 token + 最大输出 token 分配；GDN、输入驻留与 logits 也计入预算。取消、完成后释放额度。排队期间也执行超时与取消。
- 默认 `--workspace-mib 8192`，用于视觉/算子临时内存和 Graph 等开销。它是保守预留参数，不能保证覆盖所有输入的峰值；不要未经测量直接调小。
- 可用 `--host-prefix-cache-mib 4096` 开启每 rank 4 GiB 的锁页主存缓存。GPU 整段 prompt checkpoint 被淘汰时可转入主存，后续命中恢复 KV、卷积、GDN 和 logits。默认关闭；TP2 总锁页上限是每 rank 配置的两倍。没有异步重叠迁移和公共前缀匹配。
- GPU checkpoint 必须先能放入 `--prefix-cache-mib` 才能保存或降级；过大的 checkpoint 会跳过，不能仅增大 Host 上限。
- HTTP 启动会先检查模型文件是否缺失/截断。也可独立运行 `python tools/preflight.py --model /models/avi-fp8-tp2 --tp 2`，无需 GPU。文件长度检查不等于哈希或数值校验。
- `--trace` 现在保存逐步 decode logits。参考工具默认以原生 token 做 teacher forcing，对照最多 8 步；`--decode-check 0` 可关闭。开启 trace 的计时含诊断开销，不应作为性能结果。
- 小模型验收新增主存缓存命中恢复、错误请求拒绝后继续服务的检查。HTTP 本地测试使用替代 transport/tokenizer，仅检查协议与资源清理。

首次服务器安装测试依赖后执行 `scripts/acceptance.sh`；仅完成构建检查不足以验收。日志保留在新建的运行目录中，真实模型对照仍单独执行。

## 明确限制

- 本机没有 CUDA/LibTorch；C++/CUDA 主引擎尚未编译、未进行 A100 数值或性能验证。优化代码存在不等于实测加速，首次服务器运行仍可能暴露构建或数值问题。
- GDN 仍按时间顺序扫描，不是并行 chunk prefill 算法；视觉塔、embedding 和输出头仍各卡复制。没有请求抢占、paged KV、量化 KV、MTP、跨节点或多请求 CUDA Graph。
- 静态图片输入，不支持视频、JSON Schema 或工具调用。JSON 首次语法掩码在 CPU 构造，可能影响首 token 延迟。
- KV 按请求输入加输出预算分配；新增显存预算准入，但仍没有分页 KV。工作区是显式预留，不是精确峰值模型；极端图像、Graph 池和分配器碎片仍可能导致 OOM。
- 所有模型配置、processor 和微调产物须核对；训练版本记录见 configs/training-provenance.json。尚未验收真实识图准确率、长上下文、TP4 或 LoRA 合并等价性。

## 本地检查记录

2026-09-28：15 项 Python 单元/HTTP 接口测试、工具 AST 检查及纯 C++ JSON grammar 和显存预算测试，结果见 docs/implementation-status.md。加载器预期的 1184 个非 MTP 权重名称此前已与官方索引核对。CUDA 内核测试和服务器集成脚本已提供，尚未执行。源码和参考许可见 NOTICE。
