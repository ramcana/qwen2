#!/usr/bin/env python3
"""
Minimal smoke test for Qwen-Image pipeline end-to-end functionality.
"""
import torch
from PIL import Image
from diffusers import DiffusionPipeline
from huggingface_hub import snapshot_download
import os


def main():
    print("🧪 Running Qwen Image smoke test...")
    
    # Check if model exists locally first, then try cache
    model_path = "./models/Qwen-Image"
    if os.path.exists(model_path):
        print(f"✅ Using local model: {model_path}")
        model_id = model_path
    else:
        # Try to use from HuggingFace cache
        try:
            cached_path = snapshot_download("Qwen/Qwen-Image", local_files_only=True)
            print(f"✅ Using cached model: {cached_path}")
            model_id = cached_path
        except:
            print("📥 Using remote model: Qwen/Qwen-Image")
            model_id = "Qwen/Qwen-Image"
    
    # Initialize pipeline
    print("🔧 Loading pipeline...")
    pipe = DiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True
    )
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = pipe.to(device)
    print(f"✅ Pipeline loaded on {device}")
    
    # Generate image (text-to-image, not editing)
    print("🚀 Generating image...")
    prompt = "A beautiful landscape with mountains and a clear blue sky"
    
    with torch.inference_mode():
        result = pipe(
            prompt=prompt,
            width=1024,
            height=1024,
            num_inference_steps=25,
            guidance_scale=4.0
        )
        generated_image = result.images[0]
    
    # Save result
    output_path = "generated_smoke_test.jpg"
    generated_image.save(output_path)
    print(f"💾 Saved result to: {output_path}")
    
    print("🎉 Smoke test completed successfully!")
    print(f"📊 Memory used: {torch.cuda.memory_allocated() / 1024**3:.2f} GB" if torch.cuda.is_available() else "CPU mode")


if __name__ == "__main__":
    main()