#!/usr/bin/env python3
"""
Quick performance test for Qwen-Image with memory optimizations
"""
import torch
import gc
import time
from diffusers import DiffusionPipeline

def clear_memory():
    """Clear GPU memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

def test_generation_speed():
    """Test generation with different optimization levels"""
    print("⚡ Quick Performance Test")
    print("=" * 30)
    
    # Clear memory first
    clear_memory()
    
    try:
        # Load model
        print("📁 Loading model...")
        pipe = DiffusionPipeline.from_pretrained(
            "./models/Qwen-Image",
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipe = pipe.to(device)
        print(f"✅ Model loaded on {device}")
        
        # Test configurations
        tests = [
            {"name": "Fast", "steps": 10, "size": 512},
            {"name": "Quality", "steps": 25, "size": 1024}
        ]
        
        for test in tests:
            print(f"\n🧪 {test['name']} Test ({test['size']}x{test['size']}, {test['steps']} steps)")
            
            # Clear memory before test
            clear_memory()
            
            # Monitor memory
            if torch.cuda.is_available():
                memory_before = torch.cuda.memory_allocated() / (1024**3)
                print(f"📊 Memory before: {memory_before:.2f} GB")
            
            # Generate
            start_time = time.time()
            
            with torch.inference_mode():
                result = pipe(
                    "A beautiful mountain landscape",
                    width=test['size'],
                    height=test['size'],
                    num_inference_steps=test['steps'],
                    guidance_scale=4.0
                )
            
            generation_time = time.time() - start_time
            
            # Monitor memory after
            if torch.cuda.is_available():
                memory_after = torch.cuda.memory_allocated() / (1024**3)
                print(f"📊 Memory after: {memory_after:.2f} GB")
            
            # Save result
            output_file = f"perf_test_{test['name'].lower()}.jpg"
            result.images[0].save(output_file)
            
            # Calculate performance metrics
            steps_per_sec = test['steps'] / generation_time
            pixels_per_sec = (test['size'] * test['size']) / generation_time
            
            print(f"⏱️  Total time: {generation_time:.1f}s")
            print(f"⚡ Speed: {steps_per_sec:.2f} steps/sec")
            print(f"🖼️  Throughput: {pixels_per_sec/1000:.0f}K pixels/sec")
            print(f"💾 Saved: {output_file}")
        
        print(f"\n🎉 Performance test complete!")
        
        # Memory optimization tips
        if torch.cuda.is_available():
            total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            current_usage = torch.cuda.memory_allocated() / (1024**3)
            usage_percent = (current_usage / total_memory) * 100
            
            print(f"\n📊 GPU Memory Summary:")
            print(f"   Total: {total_memory:.1f} GB")
            print(f"   Used: {current_usage:.2f} GB ({usage_percent:.1f}%)")
            
            if usage_percent > 80:
                print("⚠️  High memory usage detected!")
                print("💡 Consider enabling memory optimizations:")
                print("   - Use smaller image sizes (512x512)")
                print("   - Reduce inference steps (10-15)")
                print("   - Enable attention slicing")
        
        return True
        
    except Exception as e:
        print(f"❌ Performance test failed: {e}")
        return False

def main():
    """Main function"""
    success = test_generation_speed()
    
    if success:
        print("\n🚀 Ready for production use!")
        print("💡 Next steps:")
        print("   - Start UI: make ui")
        print("   - Try different prompts")
        print("   - Experiment with settings")
    else:
        print("\n⚠️ Performance test failed")
        print("💡 Check GPU memory and try again")

if __name__ == "__main__":
    main()