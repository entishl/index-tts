# IndexTTS2 性能优化与显存管理指南 (RTX 4070 / WSL2)

本文档记录了针对 RTX 4070 (12GB VRAM) 环境的深度优化配置。

## 1. 快速启动 (一键脚本)
项目根目录下的 `./start.sh` 已封装最佳运行参数：
```bash
uv run webui.py --fp16 --accel --cuda_kernel
```
*   `--fp16`: 开启半精度推理（40系显卡核心提速）。
*   `--accel`: 激活 **Flash Attention 2** 硬件加速（解决 GPT 生成瓶颈的关键）。
*   `--cuda_kernel`: 开启自定义 CUDA 算子优化。

## 2. 核心性能参数 (显存 vs 速度)
主要的显存开销来自于 `AccelInferenceEngine` 的预分配机制。

### 修改位置
文件路径：`indextts/gpt/model_v2.py`
大约在第 **454-455** 行左右：

```python
# 核心配置代码段
block_size=256,
num_blocks=16,  # <-- 这里的 16 是显存占用的关键
use_cuda_graph=True,
```

### 参数详解
*   **`num_blocks` (当前: 16)**:
    *   **作用**: 决定了显存预分配的 Token 总容量 (`num_blocks * block_size`)。
    *   **现状**: 16 * 256 = **4096** Token。这能覆盖极长文本，但会吃掉约 10GB+ 显存。
    *   **建议**: 
        *   若想释放显存 (降至 7-8GB)，可改为 **`num_blocks=8`** (容量 2048 Token)。
        *   对于常规 WebUI 生成（120-200 文本 Token），8 个 block 绰绰有余。

*   **`use_cuda_graph` (当前: True)**:
    *   **作用**: 录制静态计算图以消除 CPU 调度开销。
    *   **影响**: 开启后速度极快 (RTF < 0.5)，但会额外预留 1-2GB 静态显存。
    *   **建议**: 追求极致速度则保持 True；若显存告急发生 OOM，请改为 **False**。

## 3. WebUI 参数关联
在 WebUI 界面中看到的 **"Max tokens per generation segment"**:
*   **默认值**: 120 (建议范围 80-200)。
*   **关联性**: 这个值决定了单次生成的“任务量”。它生成的总 Token (文本+语音) 必须小于后端代码中的总容量 (`num_blocks * 256`)。
*   **匹配方案**: 
    *   若 UI 设为 120 -> 后端 `num_blocks` 设为 8 即可。
    *   若 UI 设为 200+ -> 后端 `num_blocks` 建议保持 12-16。

## 4. 环境依赖备忘
如需重新安装或迁移，必须确保以下组件已就绪：
1.  **CUDA Toolkit**: 需安装 `nvcc` (当前版本 12.4)。
2.  **Flash Attention 2**: `uv pip install flash-attn --no-build-isolation`。
3.  **DeepSpeed**: `uv sync --all-extras` (需在有 `nvcc` 的环境下执行以完成编译)。

---
*Created on: 2026-03-10*
