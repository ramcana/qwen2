#!/usr/bin/env python3
"""
Ultra-fast Qwen Image Edit test with aggressive GPU optimizations.
"""
import torch
import time
from PIL import Image
from diffusers import QwenImageEditPipeline
import os
import gc


def clear_gpu_memory():
    """Aggressively clear GPU memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gc.collect()


def main():
    """Ultra-fast test with minimal model loading."""
    print("⚡ Ultra-Fast Qwen Image Edit Test")
    print("=" * 40)
    
    # Clear everything first
    clear_gpu_memory()
    
    # Check GPU
    if not torch.cuda.is_available():
        print("❌ CUDA not available!")
        return
    
    print(f"🔥 GPU: {torch.cuda.get_device_name(0)}")
    
    # Use local model
    model_path = "./models/Qwen-Image-Edit"
    print(f"📂 Using model: {model_path}")
    
    # Load with minimal settings for speed
    print("⚡ Loading pipeline (minimal settings)...")
    start_time = time.time()
    
    # Try the most basic loading approach
    pipe = QwenImageEditPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
    )
    
    # Move to GPU
    pipe = pipe.to("cuda")
    
    # Enable basic optimizations only
    pipe.enable_attention_slicing()
    
    load_time = time.time() - start_time
    print(f"✅ Loaded in {load_time:.1f}s")
    
    # Check GPU memory
    allocated = torch.cuda.memory_allocated(0) / 1024**3
    print(f"🔥 GPU Memory: {allocated:.1f}GB allocated")
    
    # Create simple test
    print("🎨 Creating test image...")
    img = Image.new("RGB", (256, 256), "blue")  # Smaller image for speed
    
    # Ultra-fast generation settings
    print("⚡ Generating (ultra-fast settings)...")
    prompt = "red circle"
    
    gen_start = time.time()
    
    with torch.inference_mode():
        result = pipe(
            prompt=prompt,
            image=img,
            num_inference_steps=5,  # Minimal steps
            guidance_scale=1.0,     # Minimal guidance
            height=256,             # Small size
            width=256,
        )
        edited_image = result.images[0]
    
    gen_time = time.time() - gen_start
    print(f"✅ Generated in {gen_time:.1f}s")
    print(f"⚡ Speed: {gen_time/5:.2f}s per step")
    
    # Save result
    edited_image.save("test_fast.jpg")
    print("💾 Saved test_fast.jpg")
    
    # Performance check
    if gen_time/5 > 5:
        print("❌ Still too slow! Possible issues:")
        print("   - Model not using GPU properly")
        print("   - Memory fragmentation")
        print("   - Driver issues")
        print("💡 Try: sudo nvidia-smi -r")
    else:
        print("✅ Performance acceptable!")


if __name__ == "__main__":
    main()