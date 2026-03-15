# Index-TTS (Optimized)

本仓库在原始 IndexTTS2 基础上做了少许工程化和性能优化，并新增了 API 中间层能力，提供一个完全 OpenAI 兼容的API服务，可直接对接 OpenClaw 等工具。

**性能优化**
- 集成 `flash-attn` 加速（可通过参数禁用）。
- 支持禁用情感分析模型（QwenEmotion），降低推理开销。
- 在 RTX 4070 上，RTF 最低可达 `0.4`。显存占用8-9G。

**API 中间层功能**
- 提供 OpenAI 兼容风格的 `/v1/audio/speech` 接口，便于快速对接。
- 支持真实参考音频与虚拟参考音色（`visual_voice.json`）。
- 支持情感控制与情感参考音频路径校验。
- 自动清理 API 生成的临时文件。
- 内置 `ffmpeg` 转码（默认输出 `mp3`，支持 `wav` 等格式）。

**环境要求与安装方式**
- **系统**：Linux 或 WSL2。
- **Python**：`>=3.10` (推荐使用 [uv](https://docs.astral.sh/uv/))。
- **CUDA**：需要 NVIDIA GPU。**安装加速库（Flash-Attn/DeepSpeed）必须预装 CUDA Toolkit (含 `nvcc`)**，建议版本 12.4。
- **依赖**：需要系统安装 `ffmpeg`。

安装步骤（以 Ubuntu/WSL2 为例）：
```bash
# 1. 安装系统开发工具与 FFmpeg
sudo apt update && sudo apt install -y ffmpeg build-essential

# 2. 同步环境与所有加速依赖 (自动创建 .venv)
# 注意：此过程会编译 flash-attn 和 deepspeed，请确保 nvcc 可用
uv sync --all-extras
```

---

**启动方式**

> **⚠️ 注意**：项目自带的 `.sh` 脚本中硬编码了 `CUDA_HOME=/usr/local/cuda-12.4`。如果您的 CUDA 安装路径或版本不同，请在运行前修改 `api.sh` 和 `serve.sh`。

**1. 启动 OpenAI 兼容 API + WebUI (推荐)**
```bash
./serve.sh
```
- 默认端口：`8000`
- WebUI 地址：`http://localhost:8000/ui`
- API 接口：完全兼容 OpenAI 风格的 `/v1/audio/speech`。

**2. 仅启动 API 服务**
```bash
./api.sh
```

**常用启动参数** (已在脚本中默认开启)：
- `--fp16` 启用 FP16 推理 (加速且省显存)。
- `--deepspeed` 启用 DeepSpeed 推理 (显著提升吞吐)。
- `--accel` 启用 **Flash Attention 2** 硬件加速。
- `--cuda_kernel` 启用自定义 CUDA 算子优化。
- `--disable_emo_text` 禁用情感分析模型 (降低开销，推荐在 API 模式下配合 `visual_voice.json` 使用)。

---

**参考音频管理**

本项目中的 `voice` (参考音色) 和 `emo_ref_path` (情感参考) 参数均支持直接填写文件名。系统会按以下顺序自动在本地目录中检索：
1.  `prompts/` 目录
2.  `examples/` 目录

**建议**：将您自己的参考音频（`.wav` 或 `.mp3`）放入 `prompts/` 文件夹中，然后在 API 请求中直接引用文件名即可。

---

**API 请求示例**

本服务提供 OpenAI 兼容接口，可直接对接 OpenClaw、One-API 等第三方工具。

基础请求（使用 `prompts/` 或 `examples/` 下的真实音频）：
```bash
curl -X POST http://127.0.0.1:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "indextts-v2",
    "input": "你好，这是一个测试。",
    "voice": "sample_prompt.wav",
    "response_format": "mp3"
  }' \
  --output speech.mp3
```

设置采样参数：
```bash
curl -X POST http://127.0.0.1:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "input": "你好，这是一个测试。",
    "voice": "sample_prompt.wav",
    "temperature": 0.8,
    "top_p": 0.8,
    "top_k": 30
  }' \
  --output speech.mp3
```

情感控制示例（向量）：
```bash
curl -X POST http://127.0.0.1:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "input": "今天心情很好。",
    "voice": "sample_prompt.wav",
    "emo_control_method": 2,
    "emo_vector": [0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.2]
  }' \
  --output speech.mp3
```

**visual_voice 与 JSON 用法**

文件位置：`visual_voice.json`。

核心概念：
官方的文本描述自动生成情感向量还处于试验阶段，效果难以预测。
作为对接openclaw等助手使用，其实我们对它的音色情感要求是一贯的。
因此，我们可以直接找到或生成自己想要的音色，并对它进行情感向量调整（可在webui中调试，然后写入到virtual_vocie.json中）。
理论上我们可以直接在客户端传入相应的情感向量参数，但是直接集成到中间层是一个更简单的办法。


- `virtual_voices` 的 key 是对外暴露的音色名，在 API 里作为 `voice` 使用。
- `base` 是真实参考音频文件名，必须位于 `prompts/` 或 `examples/`。
- `emo_vector` 是 8 维情感向量，顺序固定为：
  - 喜/Happy
  - 怒/Angry
  - 哀/Sad
  - 惧/Afraid
  - 厌恶/Disgusted
  - 低落/Melancholic
  - 惊喜/Surprised
  - 平静/Calm
- `emo_weight` 可选。
- 可在虚拟音色里直接覆盖采样参数，若未指定则使用请求参数或默认值。

可覆盖的采样参数：
- `do_sample`
- `top_p`
- `top_k`
- `temperature`
- `length_penalty`
- `num_beams`
- `repetition_penalty`
- `max_mel_tokens`
- `max_text_tokens_per_segment`

示例（节选）：
```json
{
  "virtual_voices": {
    "yyds": {
      "base": "yuki.wav",
      "emo_vector": [0.5, 0.1, 0.0, 0.0, 0.1, 0.0, 0.1, 0.1],
      "emo_weight": 0.65,
      "do_sample": true,
      "top_p": 0.8,
      "top_k": 30,
      "temperature": 0.8,
      "length_penalty": 0.0,
      "num_beams": 3,
      "repetition_penalty": 10.0,
      "max_mel_tokens": 1500,
      "max_text_tokens_per_segment": 120
    }
  }
}
```

使用虚拟音色：
```bash
curl -X POST http://127.0.0.1:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "input": "这是虚拟音色的示例。",
    "voice": "yyds"
  }' \
  --output speech.mp3
```

**说明**
- 原始官方仓库的 README 已重命名为 `official_README.md`，以便保留原始内容不做修改。
