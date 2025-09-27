#!/usr/bin/env python3
"""
Apply memory optimizations to reduce GPU memory usage
"""
import torch
import gc
from diffusers import DiffusionPipeline
import time

def clear_memory():
    """Aggressively clear GPU memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()

def test_memory_optimized_generation():
    """Test generation with aggressive memory optimizations"""
    print("🔧 Memory-Optimized Generation Test")
    print("=" * 40)
    
    # Clear all memory first
    clear_memory()
    
    try:
        print("📁 Loading model with memory optimizations...")
        
        # Load with memory optimizations
        pipe = DiffusionPipeline.from_pretrained(
            "./models/Qwen-Image",
            torch_dtype=torch.float16,  # Essential for memory
            trust_remote_code=True,
            low_cpu_mem_usage=True,     # Reduce CPU memory during loading
        )
        
        # Move to GPU
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipe = pipe.to(device)
        
        # Apply all memory optimizations
        print("⚡ Applying memory optimizations...")
        
        # Enable attention slicing (reduces memory significantly)
        pipe.enable_attention_slicing()
        print("   ✓ Attention slicing enabled")
        
        # Enable VAE slicing (reduces VAE memory usage)
        if hasattr(pipe, 'enable_vae_slicing'):
            pipe.enable_vae_slicing()
            print("   ✓ VAE slicing enabled")
        
        # Enable VAE tiling for large images
        if hasattr(pipe, 'enable_vae_tiling'):
            pipe.enable_vae_tiling()
            print("   ✓ VAE tiling enabled")
        
        # Enable model CPU offload (keeps only active parts on GPU)
        if hasattr(pipe, 'enable_model_cpu_offload'):
            pipe.enable_model_cpu_offload()
            print("   ✓ Model CPU offload enabled")
        
        # Test with conservative settings
        print("\n🧪 Testing with memory-optimized settings...")
        
        # Clear memory before generation
        clear_memory()
        
        # Monitor memory
        if torch.cuda.is_available():
            memory_before = torch.cuda.memory_allocated() / (1024**3)
            print(f"📊 GPU Memory before: {memory_before:.2f} GB")
        
        # Generate with conservative settings
        start_time = time.time()
        
        with torch.inference_mode():
            result = pipe(
                "A simple mountain landscape",
                width=512,           # Smaller size
                height=512,          # Smaller size  
                num_inference_steps=10,  # Fewer steps
                guidance_scale=3.0,      # Lower guidance
                output_type="pil"
            )
        
        generation_time = time.time() - start_time
        
        # Monitor memory after
        if torch.cuda.is_available():
            memory_after = torch.cuda.memory_allocated() / (1024**3)
            print(f"📊 GPU Memory after: {memory_after:.2f} GB")
            
            # Check if we're within GPU limits
            total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            usage_percent = (memory_after / total_memory) * 100
            
            if usage_percent <= 100:
                print(f"✅ Memory usage within limits: {usage_percent:.1f}%")
            else:
                print(f"⚠️ Still exceeding GPU memory: {usage_percent:.1f}%")
        
        # Save result
        output_file = "memory_optimized_test.jpg"
        result.images[0].save(output_file)
        
        print(f"⏱️ Generation time: {generation_time:.1f}s")
        print(f"⚡ Speed: {10/generation_time:.2f} steps/sec")
        print(f"💾 Saved: {output_file}")
        
        return True
        
    except Exception as e:
        print(f"❌ Memory optimization test failed: {e}")
        return False

def test_sequential_cpu_offload():
    """Test with sequential CPU offload for maximum memory savings"""
    print("\n🔄 Testing Sequential CPU Offload (Maximum Memory Savings)")
    print("=" * 60)
    
    clear_memory()
    
    try:
        # Load model
        pipe = DiffusionPipeline.from_pretrained(
            "./models/Qwen-Image",
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
        
        # Enable sequential CPU offload (most aggressive memory saving)
        if hasattr(pipe, 'enable_sequential_cpu_offload'):
            pipe.enable_sequential_cpu_offload()
            print("✓ Sequential CPU offload enabled (maximum memory savings)")
        else:
            # Fallback to regular CPU offload
            if hasattr(pipe, 'enable_model_cpu_offload'):
                pipe.enable_model_cpu_offload()
                print("✓ Model CPU offload enabled")
        
        # Enable all other optimizations
        pipe.enable_attention_slicing()
        if hasattr(pipe, 'enable_vae_slicing'):
            pipe.enable_vae_slicing()
        
        # Test generation
        clear_memory()
        
        if torch.cuda.is_available():
            memory_before = torch.cuda.memory_allocated() / (1024**3)
            print(f"📊 GPU Memory: {memory_before:.2f} GB")
        
        start_time = time.time()
        
        with torch.inference_mode():
            result = pipe(
                "A peaceful forest scene",
                width=512,
                height=512,
                num_inference_steps=8,  # Even fewer steps
                guidance_scale=2.5      # Lower guidance
            )
        
        generation_time = time.time() - start_time
        
        if torch.cuda.is_available():
            memory_after = torch.cuda.memory_allocated() / (1024**3)
            total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            usage_percent = (memory_after / total_memory) * 100
            
            print(f"📊 GPU Memory after: {memory_after:.2f} GB ({usage_percent:.1f}%)")
        
        # Save result
        result.images[0].save("sequential_offload_test.jpg")
        
        print(f"⏱️ Generation time: {generation_time:.1f}s")
        print(f"💾 Saved: sequential_offload_test.jpg")
        
        return True
        
    except Exception as e:
        print(f"❌ Sequential offload test failed: {e}")
        return False

def main():
    """Main optimization function"""
    print("🚀 Memory Optimization Suite")
    print("=" * 50)
    
    # Test 1: Standard memory optimizations
    success1 = test_memory_optimized_generation()
    
    # Test 2: Maximum memory savings
    success2 = test_sequential_cpu_offload()
    
    if success1 or success2:
        print("\n🎉 Memory optimization tests completed!")
        print("\n💡 Recommendations:")
        print("1. Always use torch.float16 precision")
        print("2. Enable attention slicing for memory efficiency")
        print("3. Use 512x512 resolution for testing")
        print("4. Use 8-15 inference steps for speed")
        print("5. Enable CPU offload if memory is tight")
        
        print("\n🚀 Optimized settings for your 16GB GPU:")
        print("- Resolution: 512x512 or 768x768 max")
        print("- Steps: 10-15 for speed, 20-25 for quality")
        print("- Enable all memory optimizations")
        
    else:
        print("\n⚠️ Memory optimization failed")
        print("💡 Consider using CPU-only mode for very large models")

if __name__ == "__main__":
    main()