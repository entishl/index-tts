import argparse
import os

import gradio as gr
import uvicorn

import api as api_mod
import webui as webui_mod


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="IndexTTS Unified Server (API + WebUI)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--port", type=int, default=8000, help="Port to run the server on")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to run the server on")
    parser.add_argument("--model_dir", type=str, default="./checkpoints", help="Model checkpoints directory")
    parser.add_argument("--fp16", action="store_true", default=False, help="Use FP16 for inference if available")
    parser.add_argument("--deepspeed", action="store_true", default=False, help="Use DeepSpeed to accelerate if available")
    parser.add_argument("--cuda_kernel", action="store_true", default=False, help="Use CUDA kernel for inference if available")
    parser.add_argument("--accel", action="store_true", default=False, help="Use Accel engine (Flash Attention 2) if available")
    parser.add_argument("--disable_emo_text", action="store_true", default=False, help="Disable text-based emotion control model (QwenEmotion)")
    parser.add_argument("--gui_seg_tokens", type=int, default=120, help="GUI: Max tokens per generation segment")
    parser.add_argument("--verbose", action="store_true", default=False, help="Enable verbose mode")
    parser.add_argument("--ui_path", type=str, default="/ui", help="Mount path for WebUI")
    return parser.parse_args(argv)


def main():
    args = parse_args()

    webui_mod.validate_model_dir(args.model_dir)
    os.environ["HF_HUB_CACHE"] = os.path.join(args.model_dir, "hf_cache")

    webui_mod.init_tts(args)
    api_mod.tts = webui_mod.tts

    demo = webui_mod.build_demo(args)
    demo.queue(20)

    app = api_mod.app
    app = gr.mount_gradio_app(app, demo, path=args.ui_path)

    print(f"Starting unified server on http://{args.host}:{args.port} (UI at {args.ui_path})")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
