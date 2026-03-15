#!/bin/bash

# 设置 CUDA 环境变量 (确保编译器和库路径正确)
export CUDA_HOME=/usr/local/cuda-12.4
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# 进入项目目录
cd "$(dirname "$0")"

echo "----------------------------------------------------------------"
echo "Starting IndexTTS2 Unified Server (API + WebUI)"
echo "Mode: FP16 + Flash Attention 2 + CUDA Kernel + DeepSpeed + Disable Emo Text"
echo "----------------------------------------------------------------"

# 使用 uv 运行统一服务，默认开启所有加速参数
# "$@" 允许您在运行脚本时传递额外参数，例如: ./serve.sh --port 8080
uv run serve.py --fp16 --accel --cuda_kernel --disable_emo_text --deepspeed "$@"
