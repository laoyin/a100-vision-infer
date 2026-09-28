# GitHub 项目介绍

仓库名称：`a100-vision-infer`

## Description（可直接粘贴）

Experimental C++/CUDA vision-language inference engine for NVIDIA A100, targeting fine-tuned Qwen3.8-27B with FP8 weight storage, BF16 compute, and TP2/TP4.

## 中文介绍

A100 Vision Infer 是面向 NVIDIA A100 80GB 的实验性视觉语言模型推理引擎，针对 Qwen3.5 架构的 Qwen3.8-27B 及其微调模型开发。参考 NInfer 的固定模型、离线权重转换和显式状态管理思路，以 C++/CUDA 实现模型执行，使用 LibTorch/ATen 基础算子和 NCCL 多卡通信。

项目已包含 FP8 权重存储与 BF16 计算、TP2/TP4 执行路径、视觉编码、Gated DeltaNet、融合 CUDA 算子、分块 prefill、批量 decode、单请求 CUDA Graph、显存预算准入、GPU/CPU 前缀缓存，以及 HTTP/SSE 服务。Python 用于离线转换、图像预处理、tokenizer 和验证，不执行线上模型 forward/generate。

当前为开发中的实验版本。CPU 与接口测试已执行，CUDA 主引擎构建、A100 数值正确性、真实识图质量和性能仍待服务器验收，不承诺已达到生产可用或性能目标。仓库不包含模型权重和业务图片。

## Topics

`cuda` `cpp` `inference` `a100` `qwen` `vision-language-model` `tensor-parallelism` `fp8` `nccl`

## 创建仓库与首次推送

在 GitHub 创建同名空仓库，不勾选初始化 README、.gitignore 或许可证，以免与本地首个提交产生不同历史。原创代码许可证尚未选择；依赖与参考代码保留原许可证，见 NOTICE。

本地首个提交完成后，在 PowerShell 中执行（替换 YOUR_GITHUB_USERNAME）：

```powershell
Set-Location D:\a100-vision-infer
git remote add origin https://github.com/YOUR_GITHUB_USERNAME/a100-vision-infer.git
git push -u origin main
```

如果已经存在 origin，先运行 `git remote -v` 检查，确认后使用 `git remote set-url origin <仓库地址>`。GitHub 身份验证由本机 Git Credential Manager 或已配置的 SSH 完成，不要把访问令牌写进远程地址或提交文件。
