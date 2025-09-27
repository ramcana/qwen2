#!/usr/bin/env python3
"""
Performance optimization script for Qwen-Image
"""
import torch
import gc
from pathlib import Path
from huggingface_hub import snapshot_download
from diffusers import DiffusionPipeline
import time

def clear_gpu_memory():
    """Clear GPU memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

def test_optimized_generation():
    """Test generation with optimizations"""
    print("🚀 Testing Optimized Qwen-Image Generation")
    print("=" * 50)
    
    # Clear memory first
    clear_gpu_memory()
    
    try:
        # Load model with optimizations
        model_path = "./models/Qwen-Image"
        print(f"📁 Loading model: {model_path}")
        
        pipe = DiffusionPipeline.from_pretrained(
            model_path,
            torch_dtype=torch.float16,  # Use float16 for memory efficiency
            trust_remote_code=True
        )
        
        # Move to GPU
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipe = pipe.to(device)
        
        # Apply optimizations
        print("⚡ Applying performance optimizations...")
        
        # Enable memory efficient attention
        pipe.enable_attention_slicing()
        
        # Enable VAE slicing for memory efficiency
        if hasattr(pipe, 'enable_vae_slicing'):
            pipe.enable_vae_slicing()
        
        # Enable model CPU offload if memory is tight
        if hasattr(pipe, 'enable_model_cpu_offload'):
            pipe.enable_model_cpu_offload()
        
        print("✅ Optimizations applied")
        
        # Test generation with different settings
        test_configs = [
            {
                "name": "Fast (512x512)",
                "width": 512,
                "height": 512,
                "steps": 10,
                "guidance": 3.0
            },
            {
                "name": "Balanced (1024x1024)", 
                "width": 1024,
                "height": 1024,
                "steps": 20,
                "guidance": 4.0
            }
        ]
        
        for i, config in enumerate(test_configs):
            print(f"\n🧪 Test {i+1}: {config['name']}")
            
            start_time = time.time()
            
            # Clear memory before each test
            clear_gpu_memory()
            
            # Monitor memory before generation
            if torch.cuda.is_available():
                memory_before = torch.cuda.memory_allocated() / (1024**3)
                print(f"📊 Memory before: {memory_before:.2f} GB")
            
            # Generate
            prompt = f"A serene mountain landscape with clear blue sky, test {i+1}"
            
            with torch.inference_mode():
                result = pipe(
                    prompt=prompt,
                    width=config["width"],
                    height=config["height"],
                    num_inference_steps=config["steps"],
                    guidance_scale=config["guidance"]
                )
                
            generation_time = time.time() - start_time
            
            # Monitor memory after generation
            if torch.cuda.is_available():
                memory_after = torch.cuda.memory_allocated() / (1024**3)
                print(f"📊 Memory after: {memory_after:.2f} GB")
                print(f"📊 Memory delta: {memory_after - memory_before:.2f} GB")
            
            # Save result
            output_path = f"optimized_test_{i+1}.jpg"
            result.images[0].save(output_path)
            
            print(f"⏱️  Generation time: {generation_time:.1f}s")
            print(f"⚡ Speed: {config['steps']/generation_time:.2f} steps/sec")
            print(f"💾 Saved: {output_path}")
            
            # Clear memory after each test
            clear_gpu_memory()
        
        print("\n🎉 Performance testing complete!")
        
    except Exception as e:
        print(f"❌ Optimization test failed: {e}")
        return False
    
    return True

def benchmark_memory_usage():
    """Benchmark different memory optimization strategies"""
    print("\n🔬 Memory Optimization Benchmark")
    print("=" * 40)
    
    strategies = [
        {"name": "No optimizations", "attention_slicing": False, "vae_slicing": False, "cpu_offload": False},
        {"name": "Attention slicing", "attention_slicing": True, "vae_slicing": False, "cpu_offload": False},
        {"name": "VAE + Attention slicing", "attention_slicing": True, "vae_slicing": True, "cpu_offload": False},
        {"name": "Full optimization", "attention_slicing": True, "vae_slicing": True, "cpu_offload": True},
    ]
    
    for strategy in strategies:
        print(f"\n📋 Testing: {strategy['name']}")
        
        try:
            # Clear memory
            clear_gpu_memory()
            
            # Load pipeline
            pipe = DiffusionPipeline.from_pretrained(
                "./models/Qwen-Image",
                torch_dtype=torch.float16,
                trust_remote_code=True
            )
            
            # Move to device
            if not strategy["cpu_offload"]:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                pipe = pipe.to(device)
            
            # Apply optimizations
            if strategy["attention_slicing"]:
                pipe.enable_attention_slicing()
            if strategy["vae_slicing"] and hasattr(pipe, 'enable_vae_slicing'):
                pipe.enable_vae_slicing()
            if strategy["cpu_offload"] and hasattr(pipe, 'enable_model_cpu_offload'):
                pipe.enable_model_cpu_offload()
            
            # Measure memory
            if torch.cuda.is_available():
                memory_used = torch.cuda.memory_allocated() / (1024**3)
                print(f"   💾 GPU Memory: {memory_used:.2f} GB")
            
            # Quick generation test
            start_time = time.time()
            with torch.inference_mode():
                result = pipe(
                    "A simple test image",
                    width=512,
                    height=512,
                    num_inference_steps=5  # Very fast test
                )
            test_time = time.time() - start_time
            print(f"   ⏱️  Test time: {test_time:.1f}s")
            
            # Clean up
            del pipe
            clear_gpu_memory()
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")

def main():
    """Main optimization function"""
    print("🔧 Qwen-Image Performance Optimizer")
    print("=" * 50)
    
    # Test optimized generation
    success = test_optimized_generation()
    
    if success:
        # Benchmark memory strategies
        benchmark_memory_usage()
        
        print("\n💡 Optimization Recommendations:")
        print("1. Use float16 precision (already applied)")
        print("2. Enable attention slicing for memory efficiency")
        print("3. Use smaller resolutions for faster generation")
        print("4. Reduce inference steps for speed (10-15 steps)")
        print("5. Clear GPU memory between generations")
        
        print("\n🚀 Next steps:")
        print("- Run: make ui  # Start the web interface")
        print("- Try different prompts and settings")
        print("- Monitor memory usage with: python tools/performance_monitor.py")

if __name__ == "__main__":
    main()