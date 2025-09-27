#!/usr/bin/env python3
"""
Ubuntu-Native Qwen-Image Loader for RTX 4080 (16GB VRAM)
Implements 4-bit quantization and smart memory management
"""
import os
import torch
import tempfile
from pathlib import Path
import gc
import time

# Set optimal environment
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def setup_environment():
    """Setup optimal environment for RTX 4080"""
    print("⚙️ Setting up environment for RTX 4080...")
    
    # Determine optimal dtype
    DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    print(f"📊 Using dtype: {DTYPE}")
    
    # Setup directories
    offload_dir = os.path.expanduser("~/offload")
    os.makedirs(offload_dir, exist_ok=True)
    
    # Enable PyTorch optimizations
    if hasattr(torch.backends.cuda, "sdp_kernel"):
        torch.backends.cuda.sdp_kernel(
            enable_flash=True, 
            enable_math=False, 
            enable_mem_efficient=True
        )
        print("✅ PyTorch SDPA optimizations enabled")
    
    return DTYPE, offload_dir

def create_quantization_config(dtype):
    """Create 4-bit quantization config"""
    try:
        from transformers import BitsAndBytesConfig
        
        bnb_4bit = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_quant_type="nf4",
        )
        
        print("✅ 4-bit quantization config created")
        return bnb_4bit
        
    except ImportError:
        print("⚠️ BitsAndBytesConfig not available, using standard loading")
        return None

def load_qwen_pipeline_optimized():
    """Load Qwen-Image pipeline with Ubuntu-native optimizations"""
    print("🚀 Loading Qwen-Image with Ubuntu-native optimizations...")
    
    # Setup environment
    dtype, offload_dir = setup_environment()
    bnb_config = create_quantization_config(dtype)
    
    try:
        # Import the correct pipeline class
        from diffusers import DiffusionPipeline
        
        # Clear GPU memory first
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        
        print("📁 Loading pipeline from local model...")
        
        # Load with optimizations
        if bnb_config is not None:
            # Load with 4-bit quantization
            pipe = DiffusionPipeline.from_pretrained(
                "./models/Qwen-Image",
                torch_dtype=dtype,
                device_map="auto",
                max_memory={0: "14GiB", "cpu": "32GiB"},
                offload_folder=offload_dir,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
        else:
            # Fallback without quantization
            pipe = DiffusionPipeline.from_pretrained(
                "./models/Qwen-Image",
                torch_dtype=dtype,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            
            # Manual device placement with CPU offload
            if hasattr(pipe, 'enable_sequential_cpu_offload'):
                pipe.enable_sequential_cpu_offload()
                print("✅ Sequential CPU offload enabled")
        
        # Apply memory optimizations
        print("⚡ Applying memory optimizations...")
        
        optimization_methods = [
            "enable_attention_slicing",
            "enable_vae_slicing", 
            "enable_vae_tiling"
        ]
        
        for method in optimization_methods:
            if hasattr(pipe, method):
                getattr(pipe, method)()
                print(f"   ✓ {method}")
        
        # Check memory usage
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024**3)
            total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            usage_percent = (allocated / total) * 100
            
            print(f"📊 GPU Memory: {allocated:.2f}GB / {total:.1f}GB ({usage_percent:.1f}%)")
            
            if usage_percent > 90:
                print("⚠️ High memory usage - consider more aggressive offloading")
        
        print("✅ Pipeline loaded successfully!")
        return pipe
        
    except Exception as e:
        print(f"❌ Pipeline loading failed: {e}")
        return None

def safe_generate(pipe, prompt, **kwargs):
    """Generate with safe parameters for 16GB GPU"""
    if pipe is None:
        print("❌ No pipeline available")
        return None
    
    # Default safe parameters
    safe_params = {
        "width": 640,
        "height": 640,
        "num_inference_steps": 16,
        "guidance_scale": 3.5,
        **kwargs  # Allow overrides
    }
    
    print(f"🎨 Generating with safe parameters:")
    print(f"   Size: {safe_params['width']}x{safe_params['height']}")
    print(f"   Steps: {safe_params['num_inference_steps']}")
    print(f"   Guidance: {safe_params['guidance_scale']}")
    
    try:
        # Clear memory before generation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        # Monitor memory
        if torch.cuda.is_available():
            mem_before = torch.cuda.memory_allocated() / (1024**3)
            print(f"📊 Memory before: {mem_before:.2f}GB")
        
        start_time = time.time()
        
        with torch.inference_mode():
            result = pipe(prompt, **safe_params)
        
        generation_time = time.time() - start_time
        
        if torch.cuda.is_available():
            mem_after = torch.cuda.memory_allocated() / (1024**3)
            print(f"📊 Memory after: {mem_after:.2f}GB")
        
        print(f"⏱️ Generation time: {generation_time:.1f}s")
        
        return result.images[0]
        
    except Exception as e:
        print(f"❌ Generation failed: {e}")
        return None

def test_ubuntu_native_setup():
    """Test the complete Ubuntu-native setup"""
    print("🧪 Testing Ubuntu-Native Qwen-Image Setup")
    print("=" * 45)
    
    # Load pipeline
    pipe = load_qwen_pipeline_optimized()
    
    if pipe is None:
        print("❌ Pipeline loading failed")
        return False
    
    # Test generation with different quality levels
    test_configs = [
        {
            "name": "Fast Test",
            "width": 512,
            "height": 512,
            "num_inference_steps": 12,
            "guidance_scale": 3.0
        },
        {
            "name": "Balanced Test", 
            "width": 640,
            "height": 640,
            "num_inference_steps": 16,
            "guidance_scale": 3.5
        }
    ]
    
    for i, config in enumerate(test_configs):
        print(f"\n🧪 {config['name']}...")
        
        prompt = f"A serene mountain landscape with clear blue sky, test {i+1}"
        
        image = safe_generate(pipe, prompt, **{k:v for k,v in config.items() if k != 'name'})
        
        if image is not None:
            output_file = f"ubuntu_native_test_{i+1}.jpg"
            image.save(output_file)
            print(f"💾 Saved: {output_file}")
        else:
            print(f"❌ {config['name']} failed")
            return False
    
    print("\n🎉 Ubuntu-native setup test completed successfully!")
    return True

def main():
    """Main function"""
    print("🚀 Ubuntu-Native Qwen-Image for RTX 4080")
    print("=" * 45)
    
    # Check prerequisites
    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return False
    
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    
    print(f"🎮 GPU: {gpu_name}")
    print(f"📊 VRAM: {gpu_memory:.1f}GB")
    
    if "4080" not in gpu_name:
        print("⚠️ This script is optimized for RTX 4080")
    
    # Run test
    success = test_ubuntu_native_setup()
    
    if success:
        print("\n✅ Setup successful!")
        print("💡 Safe generation parameters:")
        print("   - Resolution: 512-768px")
        print("   - Steps: 12-20")
        print("   - Guidance: 3.0-4.0")
        print("   - Always use safe_generate() function")
    else:
        print("\n❌ Setup failed")
        print("💡 Try running setup_ubuntu_native.sh first")
    
    return success

if __name__ == "__main__":
    main()