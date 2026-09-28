# v0.3 实现状态

日期：2026-09-28。此记录区分源码实现、本地测试和 GPU 验收。

新增实现：按请求容量分配 KV、多卡最小可用显存准入账本、排队超时/取消、可选锁页主存 checkpoint 降级与恢复、CPU 权重文件完整性预检、逐步 decode 参考对照。

本地结果：15 项 Python 测试通过，其中 HTTP 测试使用替代 worker/tokenizer；独立 C++ JSON grammar 和 memory budget 测试编译执行通过；Python AST 检查通过。

服务器脚本覆盖：BF16/FP8、TP1/2、基线/优化/Graph、prefill 与逐步 decode 数值、并发、缓存、主存恢复、取消、错误预算拒绝。以上 GPU 测试均尚未执行。TP4、实际 27B、400 万像素、长序列及业务质量仍待验收。

剩余主要功能：分页共享 KV、量化 KV、多请求 CUDA Graph、增量公共前缀、多卡通信重叠、并行 GDN prefill、MTP、JSON Schema。视频/MoE/Responses/Anthropic 接口不在当前已实现范围。

没有 CUDA 主引擎编译通过记录，也没有任何实测加速或显存峰值数据。准入预算的 workspace 参数必须结合服务器实测，不能作为永不 OOM 保证。主存恢复为阻塞复制，尚未优化传输重叠。

用户服务器已通过旧格式小模型验收（包括 TP2、Graph 和主存缓存）。本次新增 HF 128×128 block-FP8 导入、块缩放 CUDA 和混合投影：本地 NumPy TP 分片测试通过；尚未在服务器验证新格式。test-model.sh 已改为仅处理已有 FP8，不再生成 BF16 模型包。
