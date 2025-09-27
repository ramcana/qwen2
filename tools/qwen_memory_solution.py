#!/usr/bin/env python3
"""
Qwen-Image Memory Solution for 16GB GPU
Implements proper memory management and quantization
"""
import torch
import gc
import os
from pathlib import Path
import time

def clear_memory():
    """Aggressively clear GPU memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()

def check_gpu_memory():
    """Check GPU memory status"""
    if not torch.cuda.is_available():
        return {"available": False}
    
    total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    allocated = torch.cuda.memory_allocated() / (1024**3)
    cached = torch.cuda.memory_reserved() / (1024**3)
    
    return {
        "available": True,
        "total_gb": total,
        "allocated_gb": allocated,
        "cached_gb": cached,
        "free_gb": total - allocated,
        "usage_percent": (allocated / total) * 100
    }

def load_qwen_with_quantization():
    """Load Qwen-Image with 8-bit quantization"""
    print("🔧 Loading Qwen-Image with 8-bit quantization...")
    
    try:
        # Import required libraries
        from diffusers import QwenImagePipeline
        import transformers
        
        # Clear memory first
        clear_memory()
        
        # Check initial memory
        mem_info = check_gpu_memory()
        print(f"📊 Initial GPU memory: {mem_info['allocated_gb']:.2f}GB / {mem_info['total_gb']:.1f}GB")
        
        # Load with quantization and CPU offloading
        print("📁 Loading pipeline with optimizations...")
        
        pipe = QwenImagePipeline.from_pretrained(
            "./models/Qwen-Image",
            torch_dtype=torch.float16,
            device_map="auto",  # Automatic device mapping
            load_in_8bit=True,  # 8-bit quantization
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
        
        # Enable memory optimizations
        print("⚡ Applying memory optimizations...")
        
        # Enable attention slicing
        if hasattr(pipe, 'enable_attention_slicing'):
            pipe.enable_attention_slicing()
            print("   ✓ Attention slicing enabled")
        
        # Enable VAE slicing
        if hasattr(pipe, 'enable_vae_slicing'):
            pipe.enable_vae_slicing()
            print("   ✓ VAE slicing enabled")
        
        # Enable sequential CPU offload (most aggressive)
        if hasattr(pipe, 'enable_sequential_cpu_offload'):
            pipe.enable_sequential_cpu_offload()
            print("   ✓ Sequential CPU offload enabled")
        
        # Check memory after loading
        mem_info = check_gpu_memory()
        print(f"📊 Memory after loading: {mem_info['allocated_gb']:.2f}GB / {mem_info['total_gb']:.1f}GB ({mem_info['usage_percent']:.1f}%)")
        
        if mem_info['usage_percent'] > 90:
            print("⚠️ High memory usage detected!")
            return None
        
        print("✅ Pipeline loaded successfully with optimizations")
        return pipe
        
    except Exception as e:
        print(f"❌ Quantized loading failed: {e}")
        return None

def load_qwen_cpu_offload():
    """Load Qwen-Image with aggressive CPU offloading"""
    print("🔄 Loading Qwen-Image with CPU offloading...")
    
    try:
        from diffusers import QwenImagePipeline
        
        clear_memory()
        
        # Load pipeline
        pipe = QwenImagePipeline.from_pretrained(
            "./models/Qwen-Image",
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
        
        # Enable sequential CPU offload (keeps only active parts on GPU)
        if hasattr(pipe, 'enable_sequential_cpu_offload'):
            pipe.enable_sequential_cpu_offload()
            print("✓ Sequential CPU offload enabled")
        else:
            # Fallback to model CPU offload
            if hasattr(pipe, 'enable_model_cpu_offload'):
                pipe.enable_model_cpu_offload()
                print("✓ Model CPU offload enabled")
        
        # Enable all memory optimizations
        if hasattr(pipe, 'enable_attention_slicing'):
            pipe.enable_attention_slicing()
        if hasattr(pipe, 'enable_vae_slicing'):
            pipe.enable_vae_slicing()
        if hasattr(pipe, 'enable_vae_tiling'):
            pipe.enable_vae_tiling()
        
        # Check memory
        mem_info = check_gpu_memory()
        print(f"📊 GPU Memory: {mem_info['allocated_gb']:.2f}GB ({mem_info['usage_percent']:.1f}%)")
        
        return pipe
        
    except Exception as e:
        print(f"❌ CPU offload loading failed: {e}")
        return None

def test_safe_generation(pipe):
    """Test generation with safe parameters"""
    print("\n🧪 Testing Safe Generation...")
    
    if pipe is None:
        print("❌ No pipeline available")
        return False
    
    try:
        clear_memory()
        
        # Monitor memory before generation
        mem_before = check_gpu_memory()
        print(f"📊 Memory before: {mem_before['allocated_gb']:.2f}GB")
        
        # Generate with very conservative settings
        print("🎨 Generating with safe parameters...")
        start_time = time.time()
        
        with torch.inference_mode():
            result = pipe(
                "A simple mountain landscape",
                width=512,
                height=512,
                num_inference_steps=8,  # Very few steps
                guidance_scale=2.0,     # Low guidance
                output_type="pil"
            )
        
        generation_time = time.time() - start_time
        
        # Monitor memory after
        mem_after = check_gpu_memory()
        print(f"📊 Memory after: {mem_after['allocated_gb']:.2f}GB")
        
        # Save result
        output_file = "safe_generation_test.jpg"
        result.images[0].save(output_file)
        
        print(f"✅ Generation successful!")
        print(f"⏱️ Time: {generation_time:.1f}s")
        print(f"💾 Saved: {output_file}")
        
        return True
        
    except Exception as e:
        print(f"❌ Generation failed: {e}")
        return False

def main():
    """Main memory solution function"""
    print("🛡️ Qwen-Image Memory Solution for 16GB GPU")
    print("=" * 50)
    
    # Check initial system state
    mem_info = check_gpu_memory()
    if not mem_info["available"]:
        print("❌ No GPU available")
        return False
    
    print(f"🎮 GPU: {mem_info['total_gb']:.1f}GB total")
    print(f"📊 Current usage: {mem_info['allocated_gb']:.2f}GB ({mem_info['usage_percent']:.1f}%)")
    
    # Strategy 1: Try quantization first
    print("\n🔧 Strategy 1: 8-bit Quantization")
    pipe = load_qwen_with_quantization()
    
    if pipe is None:
        # Strategy 2: CPU offloading
        print("\n🔄 Strategy 2: CPU Offloading")
        pipe = load_qwen_cpu_offload()
    
    if pipe is None:
        print("\n❌ All loading strategies failed")
        print("💡 Recommendations:")
        print("1. Use a smaller model variant")
        print("2. Increase system RAM (64GB+)")
        print("3. Use cloud GPU with more VRAM")
        return False
    
    # Test generation
    success = test_safe_generation(pipe)
    
    if success:
        print("\n🎉 Memory solution successful!")
        print("💡 Safe generation parameters:")
        print("   - Resolution: 512x512")
        print("   - Steps: 8-12")
        print("   - Guidance: 2.0-3.0")
        print("   - Always clear memory between generations")
    else:
        print("\n⚠️ Generation test failed")
        print("💡 Model may still be too large for this GPU")
    
    return success

if __name__ == "__main__":
    main()