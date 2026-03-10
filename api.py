import os
import sys
import time
import argparse
from typing import Optional, List

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
import uvicorn

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, "indextts"))

from indextts.infer_v2 import IndexTTS2

app = FastAPI(title="IndexTTS OpenAI-Compatible API")
tts = None

class SpeechRequest(BaseModel):
    model: str = Field(default="indextts-v2", description="The tts model to use.")
    input: str = Field(..., description="The text to synthesize.")
    voice: str = Field(
        default="sample_prompt.wav", 
        description="The reference audio filename located in the 'prompts/' or 'examples/' directory."
    )
    response_format: Optional[str] = Field(default="wav", description="The format to return the audio in. Default is wav.")
    speed: Optional[float] = Field(default=1.0, description="Speed of the generated audio.")
    
    # Custom IndexTTS params
    emo_control_method: Optional[int] = Field(default=0, description="0: same as reference, 1: use emo_ref_path, 2: vectors, 3: text")
    emo_ref_path: Optional[str] = Field(default=None, description="Filename for emotion reference audio in 'examples/' or 'prompts/'")
    emo_weight: Optional[float] = Field(default=0.65, description="Emotion control weight")
    emo_vector: Optional[List[float]] = Field(default=None, description="8-dim vector for emotion control [joy, anger, sadness, fear, disgust, depression, surprise, peace]")
    emo_text: Optional[str] = Field(default=None, description="Emotion description text")
    emo_random: Optional[bool] = Field(default=False, description="Randomize emotion sampling")
    
    max_text_tokens_per_segment: Optional[int] = Field(default=120)
    do_sample: Optional[bool] = Field(default=True)
    top_p: Optional[float] = Field(default=0.8)
    top_k: Optional[int] = Field(default=30)
    temperature: Optional[float] = Field(default=0.8)
    length_penalty: Optional[float] = Field(default=0.0)
    num_beams: Optional[int] = Field(default=3)
    repetition_penalty: Optional[float] = Field(default=10.0)
    max_mel_tokens: Optional[int] = Field(default=1500)

@app.post("/v1/audio/speech")
async def create_speech(req: SpeechRequest):
    global tts
    if tts is None:
        raise HTTPException(status_code=500, detail="TTS Model is not initialized.")
    if not req.input.strip():
        raise HTTPException(status_code=400, detail="Input text cannot be empty.")
        
    # locate reference voice
    prompt_path = os.path.join(current_dir, "prompts", req.voice)
    if not os.path.exists(prompt_path):
        prompt_path = os.path.join(current_dir, "examples", req.voice)
        if not os.path.exists(prompt_path):
            raise HTTPException(status_code=404, detail=f"Reference audio '{req.voice}' not found in 'prompts/' or 'examples/' directory.")
            
    # locate optional emotion reference voice
    emo_ref_mapped = None
    if req.emo_control_method == 1 and req.emo_ref_path:
        emo_ref_mapped = os.path.join(current_dir, "prompts", req.emo_ref_path)
        if not os.path.exists(emo_ref_mapped):
            emo_ref_mapped = os.path.join(current_dir, "examples", req.emo_ref_path)
            if not os.path.exists(emo_ref_mapped):
                raise HTTPException(status_code=404, detail=f"Emotion reference audio '{req.emo_ref_path}' not found in 'prompts/' or 'examples/'")

    output_dir = os.path.join(current_dir, "outputs", "api")
    os.makedirs(output_dir, exist_ok=True)
    out_filename = f"speech_{int(time.time()*1000)}.wav" 
    out_path = os.path.join(output_dir, out_filename)
    
    vec = req.emo_vector
    if req.emo_control_method == 2 and vec:
        if len(vec) != 8:
            raise HTTPException(status_code=400, detail="emo_vector must have exactly 8 elements.")
        vec = tts.normalize_emo_vec(vec, apply_bias=True)
    else:
        vec = None

    emo_text = req.emo_text if req.emo_text != "" else None

    kwargs = {
        "do_sample": req.do_sample,
        "top_p": req.top_p,
        "top_k": req.top_k,
        "temperature": req.temperature,
        "length_penalty": req.length_penalty,
        "num_beams": req.num_beams,
        "repetition_penalty": req.repetition_penalty,
        "max_mel_tokens": req.max_mel_tokens,
    }

    try:
        tts.infer(
            spk_audio_prompt=prompt_path,
            text=req.input,
            output_path=out_path,
            emo_audio_prompt=emo_ref_mapped,
            emo_alpha=req.emo_weight,
            emo_vector=vec,
            use_emo_text=(req.emo_control_method == 3),
            emo_text=emo_text,
            use_random=req.emo_random,
            max_text_tokens_per_segment=req.max_text_tokens_per_segment,
            **kwargs
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        
    return FileResponse(out_path, media_type="audio/wav", filename="speech.wav")

def main():
    parser = argparse.ArgumentParser(
        description="IndexTTS API Server",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--port", type=int, default=8000, help="Port to run the API on")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to run the API on")
    parser.add_argument("--model_dir", type=str, default="./checkpoints", help="Model checkpoints directory")
    parser.add_argument("--fp16", action="store_true", default=False, help="Use FP16 for inference if available")
    parser.add_argument("--deepspeed", action="store_true", default=False, help="Use DeepSpeed to accelerate if available")
    parser.add_argument("--cuda_kernel", action="store_true", default=False, help="Use CUDA kernel for inference if available")
    parser.add_argument("--accel", action="store_true", default=False, help="Use Accel engine (Flash Attention 2) if available")
    args = parser.parse_args()

    if not os.path.exists(args.model_dir):
        print(f"Model directory {args.model_dir} does not exist.")
        sys.exit(1)
        
    for file in ["bpe.model", "gpt.pth", "config.yaml", "s2mel.pth", "wav2vec2bert_stats.pt"]:
        if not os.path.exists(os.path.join(args.model_dir, file)):
            print(f"Required file {file} missing in {args.model_dir}. Please download it.")
            sys.exit(1)
            
    os.environ['HF_HUB_CACHE'] = os.path.join(args.model_dir, "hf_cache")
            
    print("Initializing IndexTTS2 model...")
    global tts
    tts = IndexTTS2(
        model_dir=args.model_dir,
        cfg_path=os.path.join(args.model_dir, "config.yaml"),
        use_fp16=args.fp16,
        use_deepspeed=args.deepspeed,
        use_cuda_kernel=args.cuda_kernel,
        use_accel=args.accel,
    )
    
    print(f"Starting API server on http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)

if __name__ == "__main__":
    main()
