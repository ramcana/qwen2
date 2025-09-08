# Qwen-Image-Edit Configuration
# Memory-optimized for RTX 4080 (16GB VRAM) to prevent CUDA OOM errors

from typing import Any, Dict

import torch

# Qwen-Image-Edit specific configuration
QWEN_EDIT_CONFIG: Dict[str, Any] = {
    "model_name": "Qwen/Qwen-Image-Edit",
    "torch_dtype": torch.bfloat16,    # Memory efficient
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "use_safetensors": True,
    
    # Memory optimization settings for loading
    "low_cpu_mem_usage": True,        # Required when using device_map
    "device_map": "balanced",         # Use "balanced" instead of "auto" for Qwen-Image-Edit
    "max_memory": {0: "12GB"},        # Reserve 4GB VRAM for other processes
    
    # Loading optimizations
    "variant": None,                  # Don't use fp16 variant to avoid loading issues
    "use_auth_token": None,          # No auth required for public model
    "cache_dir": "./models/qwen-image-edit",  # Local cache directory
}

# Memory management for pipeline loading
MEMORY_OPTIMIZATION_CONFIG: Dict[str, Any] = {
    # Enable attention slicing to reduce memory usage
    "enable_attention_slicing": True,
    
    # CPU offloading - use carefully to avoid device mismatch
    "enable_model_cpu_offload": False,      # Keep models on GPU for performance
    "enable_sequential_cpu_offload": True,  # Enable if you need more VRAM
    
    # Memory fraction settings
    "torch_cuda_memory_fraction": 0.75,     # Reserve 25% VRAM for other operations
    
    # Gradient checkpointing for training (if needed)
    "gradient_checkpointing": False,
}

# Safe loading function with memory checks
def get_memory_optimized_config() -> Dict[str, Any]:
    """Get memory-optimized configuration based on available VRAM"""
    
    if not torch.cuda.is_available():
        return {**QWEN_EDIT_CONFIG, "device": "cpu"}
    
    # Check available VRAM
    total_memory = torch.cuda.get_device_properties(0).total_memory
    allocated_memory = torch.cuda.memory_allocated(0)
    available_memory = total_memory - allocated_memory
    
    print(f"🔍 VRAM Status: {available_memory / 1e9:.1f}GB available of {total_memory / 1e9:.1f}GB total")
    
    config = QWEN_EDIT_CONFIG.copy()
    
    # Adjust max_memory based on available VRAM
    if available_memory < 10e9:  # Less than 10GB available
        print("⚠️ Limited VRAM detected, enabling aggressive optimization")
        config["max_memory"] = {0: "8GB"}
        config["device_map"] = "balanced"
        return config
    elif available_memory < 14e9:  # Less than 14GB available  
        print("💡 Moderate VRAM available, using balanced optimization")
        config["max_memory"] = {0: "11GB"}
        config["device_map"] = "balanced"
        return config
    else:
        print("✅ Sufficient VRAM available for standard loading")
        return config

def apply_memory_optimizations(pipeline):
    """Apply memory optimizations to loaded pipeline"""
    
    if not hasattr(pipeline, 'vae') or not torch.cuda.is_available():
        return pipeline
    
    try:
        # Enable attention slicing to reduce memory usage
        if hasattr(pipeline, 'enable_attention_slicing'):
            pipeline.enable_attention_slicing()
            print("✅ Attention slicing enabled")
        
        # Enable memory efficient attention if available
        if hasattr(pipeline, 'enable_xformers_memory_efficient_attention'):
            try:
                pipeline.enable_xformers_memory_efficient_attention()
                print("✅ XFormers memory efficient attention enabled")
            except Exception:
                print("💡 XFormers not available, using default attention")
        
        # Set memory fraction if needed
        if MEMORY_OPTIMIZATION_CONFIG["torch_cuda_memory_fraction"] < 1.0:
            torch.cuda.set_per_process_memory_fraction(
                MEMORY_OPTIMIZATION_CONFIG["torch_cuda_memory_fraction"]
            )
            print(f"✅ CUDA memory fraction set to {MEMORY_OPTIMIZATION_CONFIG['torch_cuda_memory_fraction']}")
        
        return pipeline
        
    except Exception as e:
        print(f"⚠️ Some memory optimizations failed: {e}")
        return pipeline

def clear_gpu_cache():
    """Clear GPU cache to free memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print("🧹 GPU cache cleared")

# Environment variable recommendations
ENVIRONMENT_VARS = {
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",  # Reduce fragmentation
    "CUDA_LAUNCH_BLOCKING": "0",  # Don't use blocking for performance
    "PYTORCH_NO_CUDA_MEMORY_CACHING": "0",  # Keep caching enabled
}

def set_environment_optimizations():
    """Set recommended environment variables"""
    import os
    
    for key, value in ENVIRONMENT_VARS.items():
        if key not in os.environ:
            os.environ[key] = value
            print(f"✅ Set {key}={value}")