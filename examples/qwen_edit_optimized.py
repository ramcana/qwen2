#!/usr/bin/env python3
"""
GPU-optimized Qwen Image Edit test with performance monitoring.
"""
import torch
import time
from PIL import Image
from diffusers import QwenImageEditPipeline
import os


def print_gpu_info():
    """Print detailed GPU information."""
    if torch.cuda.is_available():
        print(f"🔥 GPU: {torch.cuda.get_device_name(0)}")
        print(f"🔥 CUDA Version: {torch.version.cuda}")
        print(f"🔥 PyTorch Version: {torch.__version__}")
        
        # Memory info
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        cached = torch.cuda.memory_reserved(0) / 1024**3
        
        print(f"🔥 GPU Memory: {allocated:.1f}GB allocated, {cached:.1f}GB cached, {total_memory:.1f}GB total")
    else:
        print("❌ CUDA not available!")


def optimize_pipeline(pipe):
    """Apply GPU optimizations to the pipeline."""
    print("⚡ Applying GPU optimizations...")
    
    # Enable memory efficient attention
    try:
        pipe.enable_attention_slicing()
        print("✅ Attention slicing enabled")
    except:
        print("⚠️ Attention slicing not available")
    
    # Enable model CPU offload for memory efficiency
    try:
        pipe.enable_model_cpu_offload()
        print("✅ Model CPU offload enabled")
    except:
        print("⚠️ Model CPU offload not available")
    
    # Enable VAE slicing for memory efficiency
    try:
        if hasattr(pipe, 'vae'):
            pipe.vae.enable_slicing()
            print("✅ VAE slicing enabled")
    except:
        print("⚠️ VAE slicing not available")
    
    # Set components to eval mode and disable gradients
    try:
        if hasattr(pipe, 'text_encoder'):
            pipe.text_encoder.eval()
        if hasattr(pipe, 'transformer'):
            pipe.transformer.eval()
        if hasattr(pipe, 'vae'):
            pipe.vae.eval()
        print("✅ Components set to eval mode")
    except:
        print("⚠️ Could not set eval mode")
    
    print("✅ Pipeline optimized for inference")
    return pipe


def main():
    """Main optimized test function."""
    print("🚀 GPU-Optimized Qwen Image Edit Test")
    print("=" * 50)
    
    # Print initial GPU info
    print_gpu_info()
    
    # Clear GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("🧹 GPU cache cleared")
    
    # Check if model exists locally
    model_path = "./models/Qwen-Image-Edit"
    if os.path.exists(model_path):
        print(f"✅ Using local model: {model_path}")
        model_id = model_path
    else:
        print("📥 Using remote model: Qwen/Qwen-Image-Edit")
        model_id = "Qwen/Qwen-Image-Edit"
    
    # Initialize pipeline with optimal settings
    print("🔧 Loading pipeline with GPU optimizations...")
    start_time = time.time()
    
    pipe = QwenImageEditPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,  # Use float16 for RTX 4080
        low_cpu_mem_usage=True,     # Reduce CPU memory usage
    )
    
    # Move to GPU explicitly
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = pipe.to(device)
    
    # Apply optimizations
    pipe = optimize_pipeline(pipe)
    
    load_time = time.time() - start_time
    print(f"✅ Pipeline loaded in {load_time:.1f} seconds")
    
    # Print GPU info after loading
    print("\n📊 GPU Status After Loading:")
    print_gpu_info()
    
    # Create test image
    print("\n🎨 Creating test image...")
    img = Image.new("RGB", (512, 512), "skyblue")
    
    # Generate edited image with optimized settings
    print("🚀 Generating edited image (optimized)...")
    prompt = "Add a red kite flying in the sky"
    
    # Use optimized generation parameters
    generation_start = time.time()
    
    with torch.inference_mode():  # Disable gradients for inference
        with torch.cuda.amp.autocast():  # Use automatic mixed precision
            result = pipe(
                prompt=prompt, 
                image=img,
                num_inference_steps=20,  # Reduced steps for speed
                guidance_scale=4.0,
                generator=torch.Generator(device=device).manual_seed(42),  # Reproducible results
            )
            edited_image = result.images[0]
    
    generation_time = time.time() - generation_start
    print(f"✅ Generation completed in {generation_time:.1f} seconds")
    print(f"⚡ Speed: {generation_time/20:.2f} seconds per step")
    
    # Save result
    output_path = "edited_optimized.jpg"
    edited_image.save(output_path)
    print(f"💾 Saved result to: {output_path}")
    
    # Final GPU info
    print("\n📊 Final GPU Status:")
    print_gpu_info()
    
    # Performance summary
    print(f"\n🎉 Performance Summary:")
    print(f"   📥 Model loading: {load_time:.1f}s")
    print(f"   🎨 Image generation: {generation_time:.1f}s")
    print(f"   ⚡ Speed: {generation_time/20:.2f}s per step")
    
    if generation_time/20 > 10:
        print("⚠️ Generation is slower than expected!")
        print("💡 Try running: ./dev-stop.sh && ./dev-start.sh")
    else:
        print("✅ Performance looks good!")


if __name__ == "__main__":
    main()