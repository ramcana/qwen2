#!/usr/bin/env python3
"""
Minimal Qwen Image Edit test focusing on basic functionality.
"""
import torch
import time
from PIL import Image
from diffusers import QwenImageEditPipeline
import os


def main():
    """Minimal test with basic settings."""
    print("🎯 Minimal Qwen Image Edit Test")
    print("=" * 40)
    
    # Clear GPU
    torch.cuda.empty_cache()
    
    print(f"🔥 GPU: {torch.cuda.get_device_name(0)}")
    
    # Load model
    model_path = "./models/Qwen-Image-Edit"
    print("📂 Loading pipeline...")
    
    start_time = time.time()
    
    # Load with absolute minimal settings
    pipe = QwenImageEditPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.float32,  # Try float32 instead of float16
    )
    
    pipe = pipe.to("cuda")
    
    load_time = time.time() - start_time
    print(f"✅ Loaded in {load_time:.1f}s")
    
    # Create tiny test image
    print("🎨 Creating test image...")
    img = Image.new("RGB", (128, 128), "white")
    
    # Minimal generation test
    print("⚡ Testing generation (1 step only)...")
    prompt = "blue"
    
    gen_start = time.time()
    
    # Absolute minimal settings
    with torch.inference_mode():
        result = pipe(
            prompt=prompt,
            image=img,
            num_inference_steps=1,  # Just 1 step
            guidance_scale=1.0,
            height=128,
            width=128,
        )
        edited_image = result.images[0]
    
    gen_time = time.time() - gen_start
    print(f"✅ Generated in {gen_time:.1f}s (1 step)")
    
    # Save result
    edited_image.save("test_minimal.jpg")
    print("💾 Saved test_minimal.jpg")
    
    # Analysis
    if gen_time > 30:
        print("❌ Still very slow even with 1 step!")
        print("💡 This suggests the model itself has performance issues")
        print("🔧 Possible solutions:")
        print("   1. Use a different, faster model")
        print("   2. Use model quantization")
        print("   3. Use a different pipeline")
    elif gen_time > 10:
        print("⚠️ Slow but workable")
        print("💡 For practical use, try 5-10 steps max")
    else:
        print("✅ Good performance!")
        print("💡 You can increase steps for better quality")
    
    print(f"\n📊 Performance Summary:")
    print(f"   Loading: {load_time:.1f}s")
    print(f"   Generation (1 step): {gen_time:.1f}s")
    print(f"   Estimated 20 steps: {gen_time * 20:.1f}s")


if __name__ == "__main__":
    main()