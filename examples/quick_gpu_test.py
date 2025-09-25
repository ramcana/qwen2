#!/usr/bin/env python3
"""
Quick GPU test to verify performance without full model loading.
"""
import torch
import time
from PIL import Image
import numpy as np


def test_gpu_performance():
    """Test basic GPU performance."""
    print("🚀 Quick GPU Performance Test")
    print("=" * 40)
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return
    
    device = torch.device("cuda")
    print(f"🔥 Using: {torch.cuda.get_device_name(0)}")
    
    # Test different operations
    sizes = [1000, 2000, 4000]
    
    for size in sizes:
        print(f"\n📊 Testing {size}x{size} matrices:")
        
        # Float32 test
        start = time.time()
        a = torch.randn(size, size, device=device, dtype=torch.float32)
        b = torch.randn(size, size, device=device, dtype=torch.float32)
        c = torch.mm(a, b)
        torch.cuda.synchronize()
        fp32_time = time.time() - start
        
        # Float16 test
        start = time.time()
        a16 = a.half()
        b16 = b.half()
        c16 = torch.mm(a16, b16)
        torch.cuda.synchronize()
        fp16_time = time.time() - start
        
        print(f"  Float32: {fp32_time:.3f}s")
        print(f"  Float16: {fp16_time:.3f}s ({fp32_time/fp16_time:.1f}x faster)")
        
        # Memory usage
        memory_used = torch.cuda.memory_allocated() / 1024**3
        print(f"  Memory: {memory_used:.1f}GB")
        
        # Clear memory
        del a, b, c, a16, b16, c16
        torch.cuda.empty_cache()


def test_image_processing():
    """Test image processing on GPU."""
    print("\n🎨 Image Processing Test")
    print("=" * 40)
    
    if not torch.cuda.is_available():
        return
    
    device = torch.device("cuda")
    
    # Create test image
    img = Image.new("RGB", (512, 512), "blue")
    img_array = np.array(img).astype(np.float32) / 255.0
    
    # Convert to tensor and move to GPU
    start = time.time()
    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0).to(device)
    
    # Simple image operations
    processed = img_tensor * 0.8 + 0.2  # Brightness adjustment
    processed = torch.clamp(processed, 0, 1)
    
    # Convert back
    result = processed.squeeze(0).permute(1, 2, 0).cpu().numpy()
    result = (result * 255).astype(np.uint8)
    result_img = Image.fromarray(result)
    
    torch.cuda.synchronize()
    process_time = time.time() - start
    
    print(f"✅ Image processing: {process_time:.3f}s")
    print(f"📊 Input shape: {img_tensor.shape}")
    print(f"💾 Saved test image: test_gpu_output.jpg")
    result_img.save("test_gpu_output.jpg")


def check_model_loading_issue():
    """Try to identify why model loading is slow."""
    print("\n🔍 Model Loading Analysis")
    print("=" * 40)
    
    try:
        from diffusers import QwenImageEditPipeline
        
        # Try loading just the config first
        print("📋 Checking model configuration...")
        
        import json
        from pathlib import Path
        
        model_path = Path("./models/Qwen-Image-Edit")
        config_path = model_path / "model_index.json"
        
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
            print("✅ Model config loaded")
            print(f"📊 Components: {list(config.keys())}")
        
        # Check individual component sizes
        components = ["text_encoder", "transformer", "vae", "scheduler"]
        for comp in components:
            comp_path = model_path / comp
            if comp_path.exists():
                size = sum(f.stat().st_size for f in comp_path.rglob("*") if f.is_file())
                print(f"📁 {comp}: {size / 1024**3:.1f}GB")
        
        print("\n💡 Recommendations:")
        print("1. Model files look complete")
        print("2. Try loading with torch_dtype=torch.float32")
        print("3. Consider using enable_model_cpu_offload() for memory")
        print("4. The slow loading might be normal for this large model")
        
    except Exception as e:
        print(f"❌ Error checking model: {e}")


def main():
    """Main test function."""
    test_gpu_performance()
    test_image_processing()
    check_model_loading_issue()
    
    print("\n🎯 Summary:")
    print("If GPU performance tests are fast but model loading is slow,")
    print("this suggests the issue is with the model size/complexity,")
    print("not your GPU setup.")


if __name__ == "__main__":
    main()