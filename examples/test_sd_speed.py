#!/usr/bin/env python3
"""
Test Stable Diffusion speed for comparison.
"""
import torch
import time
from PIL import Image
from diffusers import StableDiffusionImg2ImgPipeline
import os


def main():
    """Test SD speed for comparison."""
    print("🚀 Stable Diffusion Speed Test (for comparison)")
    print("=" * 50)
    
    # Clear GPU
    torch.cuda.empty_cache()
    
    print(f"🔥 GPU: {torch.cuda.get_device_name(0)}")
    
    # Load a standard SD model for comparison
    print("📂 Loading Stable Diffusion 1.5...")
    
    start_time = time.time()
    
    try:
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16,
        )
        pipe = pipe.to("cuda")
        pipe.enable_attention_slicing()
        
        load_time = time.time() - start_time
        print(f"✅ SD loaded in {load_time:.1f}s")
        
        # Create test image
        img = Image.new("RGB", (512, 512), "blue")
        
        # Test generation
        print("⚡ Testing SD generation...")
        prompt = "a red circle"
        
        gen_start = time.time()
        
        with torch.inference_mode():
            result = pipe(
                prompt=prompt,
                image=img,
                num_inference_steps=20,
                guidance_scale=7.5,
                strength=0.8,
            )
            edited_image = result.images[0]
        
        gen_time = time.time() - gen_start
        print(f"✅ SD generated in {gen_time:.1f}s (20 steps)")
        print(f"⚡ SD speed: {gen_time/20:.2f}s per step")
        
        # Save result
        edited_image.save("test_sd_comparison.jpg")
        print("💾 Saved test_sd_comparison.jpg")
        
        # Compare with expected Qwen performance
        print(f"\n📊 Comparison:")
        print(f"   SD 1.5: {gen_time/20:.2f}s per step")
        print(f"   Qwen (observed): ~180s per step")
        print(f"   Qwen is ~{180/(gen_time/20):.0f}x slower than SD!")
        
        if gen_time/20 < 5:
            print("\n✅ Your GPU can run diffusion models fast!")
            print("❌ The issue is specifically with the Qwen model")
            print("💡 Qwen-Image-Edit might be:")
            print("   - Much larger than standard SD models")
            print("   - Not optimized for inference speed")
            print("   - Using inefficient attention mechanisms")
        
    except Exception as e:
        print(f"❌ Could not load SD for comparison: {e}")
        print("💡 This suggests a general diffusion model issue")


if __name__ == "__main__":
    main()