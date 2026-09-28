> 这是 v0.1 架构记录；当前功能及验收边界以 README.md 和 implementation-status.md 的 v0.2 记录为准。

# 架构与实施计划

本文件保留最初设计；v0.1 已写代码及未验证项见 [implementation-status.md](implementation-status.md)。

## 模型导出是第一道正确性关卡

训练是 BF16 LoRA，部署 FP8 是后续步骤。使用准确的基础模型 revision，加载全部适配器及 modules_to_save，导出合并的模型和处理器，再建立 BF16 参考，最后量化。

freeze_vit=false、freeze_aligner=false 和 all-linear 表明视觉侧可能也插入了 LoRA；不能将其描述为所有视觉权重全量训练。必须审计实际 adapter tensor keys、target_modules、modules_to_save 和加载时 missing/unexpected keys。合并前后比较同一图片的视觉特征、logits 与业务结果，不能静默遗漏视觉适配器。

保留 tokenizer、processor、chat template、图像缩放规则和特殊 token。训练 max_length=20480 不能直接解释成可额外生成 20480 tokens；服务应分别限制图片/文本输入和输出的总预算。max_pixels=4000000 的多图语义按实际 processor 和训练配置核对。

## FP8 在 SM80 上的精度契约

首版计划为 FP8 权重存储、BF16 激活/计算（W8A16），不是原生 W8A8。格式、scale 粒度、block shape 与混合精度排除层以 checkpoint 为准；不支持时明确失败，不能忽略 scale。

候选内核以分块解量化和 SM80 Tensor Core 乘法减少权重带宽。Marlin 路径作为技术参考，具体形状和 BF16 支持需要核验。不能永久展开全部权重到 BF16 后宣称具有 FP8 驻留优势。prefill 和 decode 分别测量，FP8 存储不保证所有负载更快。

KV 初版 BF16；DeltaNet 递归状态遵循参考实现精度。视觉、aligner、norm、门控和输出头是否量化分别决定，不能由 FP8 权重目标推断所有层与状态都应 FP8。

## 计划中的执行模块（当前状态见 implementation-status.md）

1. 权重加载与转换：safetensors、scale 校验、TP 分片及布局。
2. SM80 算子：BF16 GEMM、W8A16、norm/激活融合。
3. Qwen 混合执行图：Gated DeltaNet、完整注意力、FFN 和输出头。
4. 视觉链路：图像预处理、视觉编码、位置编码、图文 token 合并。
5. 状态管理：Attention KV 与 DeltaNet 状态共同分配、缓存、恢复、回收。
6. TP=2，再 TP=4：NCCL 通信、张量/状态分片，先不用自定义 collective。
7. 服务：批调度、取消、图片 token 预算、结构化输出、API。

对比单卡与 TP=2 的正确性，再以指定 TP=2 作为交付目标。Attention heads 可整除只是必要条件，仍须检查 DeltaNet、门控、卷积、输出头与所有分片算子。训练机 SXM4 不能证明推理机互联拓扑。

## 识图优化与验收

现有数据脚本涉及柜体、配电箱、回路、BOM 和 bbox，暂作为待确认业务方向。不得擅自降低图片分辨率；小字、线条和框坐标须保持可比性。保留裁剪/缩放坐标变换。图片 embedding 缓存键须包含内容、模型版本和预处理设置。

- 正确性：原模型+LoRA 对比合并 BF16；合并 BF16 对比 FP8；单卡对比 TP2/4。
- 质量：计数、字段、BOM、bbox IoU、JSON 合法率，阈值由业务确认。
- 性能：预处理、视觉编码、prefill、TTFT、decode、总耗时，P50/P95、吞吐和显存峰值。
- 负载：保留 400 万像素测试组；覆盖不同长宽比、多图、长输出及并发。
- 稳定性：请求取消、结束、缓存恢复与释放、显存预算、异常图片。

先完成单卡 BF16 文本与视觉数值对齐，再接 FP8 和 TP2，最后优化批处理、CUDA Graph 和缓存。TP4 与投机解码后置。没有 A100 实测前不发布性能提升或容量承诺。

## 参考来源

- https://github.com/Neroued/ninfer ：固定模型执行、权重转换和状态复用；其 SM120a 单卡实现不能直接移植到 A100 多卡。
- https://docs.vllm.ai/en/latest/features/quantization/llm_compressor/fp8/ ：Ampere FP8 W8A16/Marlin 路径与基线。
- https://github.com/sgl-project/sglang ：候选调度与混合状态管理参考，尚未源码审计。
- https://huggingface.co/docs/peft/main/developer_guides/checkpoint ：适配器保存与合并。
- https://developer.nvidia.com/blog/nvidia-ampere-architecture-in-depth/ ：硬件精度能力。

上游代码复用前固定 commit、审查文件与依赖许可、保留 attribution/NOTICE。训练软件版本记录不等于推理依赖锁定；需在实际服务器验证后生成锁文件。
