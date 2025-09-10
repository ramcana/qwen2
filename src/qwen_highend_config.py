# High-End Hardware Configuration for Qwen-Image
# Optimized for AMD Threadripper PRO 5995WX + RTX 4080 + 128GB RAM
# Target: 15-60s generation time (not 500+s!)

import os
from typing import Any, Dict

import torch

# High-Performance Model Configuration
HIGH_END_MODEL_CONFIG: Dict[str, Any] = {
    "model_name": "Qwen/Qwen-Image",
    "torch_dtype": torch.bfloat16,  # Optimal for RTX 4080
    "device": "cuda",
    "use_safetensors": True,
    # HIGH-END MEMORY SETTINGS
    "low_cpu_mem_usage": False,  # DISABLED: You have 128GB RAM!
    "device_map": None,  # DISABLED: Keep everything on GPU for speed
    "max_memory": None,  # DISABLED: Use full 16GB VRAM
    # Performance optimizations
    "variant": None,
    "use_auth_token": None,
    "cache_dir": os.environ.get("QWEN_HOME", "./models/qwen-image"),
    "local_files_only": False,
    # Attention implementation will be determined at runtime by device helper
    "attn_implementation": None,  # Will be set by get_device_config()
}

# High-Performance Memory Configuration
HIGH_END_MEMORY_CONFIG: Dict[str, bool] = {
    # DISABLE all CPU offloading for maximum speed
    "enable_attention_slicing": False,  # DISABLED: You have enough VRAM
    "enable_cpu_offload": False,  # DISABLED: Keep on GPU
    "enable_sequential_cpu_offload": False,  # DISABLED: Major performance killer!
    "enable_model_cpu_offload": False,  # DISABLED: Keep on GPU
    # Enable performance optimizations
    "enable_xformers": True,  # Enable if available
    "enable_flash_attention": True,  # Enable if available
}

# High-Performance Generation Configuration
HIGH_END_GENERATION_CONFIG: Dict[str, Any] = {
    "width": 1664,
    "height": 928,
    "num_inference_steps": 50,
    "true_cfg_scale": 4.0,
    "max_batch_size": 1,  # Can be increased to 2-3 with 16GB VRAM
    # Performance settings
    "use_fp16": False,  # Use bfloat16 instead
    "force_zeros_for_empty_prompt": False,
    "requires_safety_checker": False,  # Skip for speed
}

# Environment variables for maximum performance
HIGH_END_ENVIRONMENT: Dict[str, str] = {
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True,max_split_size_mb:512",
    "CUDA_LAUNCH_BLOCKING": "0",  # Async execution
    "PYTORCH_NO_CUDA_MEMORY_CACHING": "0",  # Keep caching
    "OMP_NUM_THREADS": "32",  # Utilize Threadripper cores
    "MKL_NUM_THREADS": "32",  # Intel MKL threading
    "NUMBA_NUM_THREADS": "32",  # Numba threading
    # Memory management
    "PYTORCH_CUDA_MEMORY_FRACTION": "0.95",  # Use 95% of VRAM (15.2GB)
    "CUDA_MEMORY_POOL_SIZE": "15000",  # 15GB memory pool
}


def setup_high_end_environment():
    """Setup environment for high-end hardware"""
    import os

    print("🚀 Configuring environment for high-end hardware...")
    print("   • AMD Threadripper PRO 5995WX (64 cores)")
    print("   • RTX 4080 (16GB VRAM)")
    print("   • 128GB System RAM")

    for key, value in HIGH_END_ENVIRONMENT.items():
        os.environ[key] = value
        print(f"✅ {key}={value}")

    # Set CUDA memory fraction
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.95)
        print("✅ CUDA memory fraction set to 95%")


def get_high_end_config() -> Dict[str, Any]:
    """Get optimized configuration for high-end hardware"""

    if not torch.cuda.is_available():
        print("❌ CUDA not available! Check GPU drivers.")
        return HIGH_END_MODEL_CONFIG

    # Check hardware
    device_props = torch.cuda.get_device_properties(0)
    total_vram = device_props.total_memory / 1e9
    device_name = device_props.name

    print(f"🔍 GPU: {device_name}")
    print(f"🔍 VRAM: {total_vram:.1f}GB")

    if total_vram < 15:
        print(f"⚠️ Warning: Expected 16GB VRAM, found {total_vram:.1f}GB")
        print("   Performance may be suboptimal")

    # Get CPU info
    import os

    cpu_count = os.cpu_count()
    print(f"🔍 CPU Cores: {cpu_count}")

    if cpu_count and cpu_count < 32:
        print(f"⚠️ Warning: Expected 64+ cores, found {cpu_count}")
        # Adjust threading for lower core count
        HIGH_END_ENVIRONMENT["OMP_NUM_THREADS"] = str(min(16, cpu_count))
        HIGH_END_ENVIRONMENT["MKL_NUM_THREADS"] = str(min(16, cpu_count))
        HIGH_END_ENVIRONMENT["NUMBA_NUM_THREADS"] = str(min(16, cpu_count))

    return HIGH_END_MODEL_CONFIG


def apply_high_end_optimizations(pipeline):
    """Apply high-end optimizations to pipeline"""

    if not torch.cuda.is_available():
        return pipeline

    print("🚀 Applying high-end performance optimizations...")

    try:
        # Disable all memory-saving features (we have enough resources!)

        # 1. DISABLE attention slicing (we have enough VRAM)
        if hasattr(pipeline, "disable_attention_slicing"):
            pipeline.disable_attention_slicing()
            print("✅ Attention slicing DISABLED (using full VRAM)")

        # 2. DISABLE CPU offloading (keep everything on GPU)
        if hasattr(pipeline, "disable_model_cpu_offload"):
            pipeline.disable_model_cpu_offload()
            print("✅ CPU offload DISABLED (keeping on GPU)")

        # 3. Enable XFormers if available (faster attention)
        if hasattr(pipeline, "enable_xformers_memory_efficient_attention"):
            try:
                pipeline.enable_xformers_memory_efficient_attention()
                print("✅ XFormers enabled (faster attention)")
            except Exception:
                print("💡 XFormers not available")

        # 4. Optimize VAE if available
        if hasattr(pipeline, "vae") and hasattr(pipeline.vae, "enable_slicing"):
            pipeline.vae.disable_slicing()  # Disable VAE slicing for speed
            print("✅ VAE slicing DISABLED (using full VRAM)")

        # 5. Ensure all components are on GPU
        components = ["unet", "vae", "text_encoder"]
        for comp_name in components:
            if hasattr(pipeline, comp_name):
                component = getattr(pipeline, comp_name)
                if component is not None:
                    component = component.to("cuda")
                    setattr(pipeline, comp_name, component)
                    print(f"✅ {comp_name.upper()} locked to GPU")

        # 6. Set optimal compilation flags if available
        if hasattr(pipeline, "unet") and hasattr(
            pipeline.unet, "set_default_attn_processor"
        ):
            # Use default attention processor for speed
            pipeline.unet.set_default_attn_processor()
            print("✅ Default attention processor set")

        print("🚀 High-end optimizations applied successfully!")

    except Exception as e:
        print(f"⚠️ Some optimizations failed: {e}")

    return pipeline


def verify_high_end_setup():
    """Verify high-end setup is working correctly"""

    print("🔍 Verifying high-end setup...")

    # Check CUDA
    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return False

    # Check VRAM
    device_props = torch.cuda.get_device_properties(0)
    total_vram = device_props.total_memory / 1e9
    available_vram = (device_props.total_memory - torch.cuda.memory_allocated(0)) / 1e9

    print(f"✅ GPU: {device_props.name}")
    print(f"✅ Total VRAM: {total_vram:.1f}GB")
    print(f"✅ Available VRAM: {available_vram:.1f}GB")

    # Check CPU
    import os

    import psutil

    cpu_count = os.cpu_count()
    memory_gb = psutil.virtual_memory().total / 1e9

    print(f"✅ CPU Cores: {cpu_count}")
    print(f"✅ System RAM: {memory_gb:.0f}GB")

    # Check environment
    pytorch_version = torch.__version__
    print(f"✅ PyTorch: {pytorch_version}")

    # Performance check
    if total_vram >= 15 and memory_gb >= 100 and cpu_count >= 32:
        print("🚀 HIGH-END CONFIGURATION VERIFIED!")
        print("   Expected generation time: 15-60 seconds")
        return True
    else:
        print("⚠️ Hardware below high-end expectations")
        print("   Expected: 16GB VRAM, 128GB RAM, 64+ cores")
        print(
            f"   Found: {total_vram:.1f}GB VRAM, {memory_gb:.0f}GB RAM, {cpu_count} cores"
        )
        return False


# Quick performance test
def quick_performance_test():
    """Quick test to verify performance improvements"""

    if not torch.cuda.is_available():
        return False

    print("🧪 Running quick performance test...")

    # Clear cache
    torch.cuda.empty_cache()

    # Test tensor operations
    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)

    start_time.record()

    # Simulate model operations
    x = torch.randn(1, 4, 64, 64, device="cuda", dtype=torch.bfloat16)
    for _ in range(10):
        x = torch.nn.functional.conv2d(
            x, torch.randn(4, 4, 3, 3, device="cuda", dtype=torch.bfloat16), padding=1
        )
        x = torch.nn.functional.relu(x)

    end_time.record()
    torch.cuda.synchronize()

    elapsed_time = start_time.elapsed_time(end_time)
    print(f"✅ GPU compute test: {elapsed_time:.2f}ms")

    if elapsed_time < 100:  # Should be very fast on RTX 4080
        print("🚀 GPU performance looks good!")
        return True
    else:
        print("⚠️ GPU performance seems slow")
        return False
