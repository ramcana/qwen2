#!/usr/bin/env python3
"""
Emergency Device Fix Script for Qwen-Image Generator
Diagnoses and fixes persistent CUDA device mismatch issues
"""

import gc
import os
import sys
from datetime import datetime

import psutil
import torch

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def clear_all_cuda():
    """Aggressively clear all CUDA memory and reset"""
    print("🧹 Aggressive CUDA cleanup...")
    try:
        # Clear Python cache
        gc.collect()
        
        # Clear CUDA cache multiple times
        for i in range(3):
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        
        # Reset CUDA context
        if torch.cuda.is_available():
            try:
                torch.cuda.reset_peak_memory_stats()
                print("✅ CUDA memory stats reset")
            except:
                pass
        
        print("✅ CUDA cleanup completed")
        return True
    except Exception as e:
        print(f"❌ CUDA cleanup failed: {e}")
        return False

def diagnose_system():
    """Comprehensive system diagnosis"""
    print("🔍 System Diagnosis")
    print("=" * 50)
    
    # Python and PyTorch info
    print(f"Python executable: {psutil.Process().exe()}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"GPU {i}: {props.name}")
            print(f"  Total memory: {props.total_memory / 1e9:.1f} GB")
            print(f"  Allocated: {torch.cuda.memory_allocated(i) / 1e9:.2f} GB")
            print(f"  Cached: {torch.cuda.memory_reserved(i) / 1e9:.2f} GB")
    
    # Memory info
    memory = psutil.virtual_memory()
    print(f"System RAM: {memory.total / 1e9:.1f} GB total, {memory.available / 1e9:.1f} GB available")
    
    print("\n")

def test_device_operations():
    """Test basic CUDA operations to identify issues"""
    print("🧪 Testing Device Operations")
    print("=" * 50)
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return False
    
    try:
        # Test 1: Basic tensor creation
        print("Test 1: Basic tensor operations...")
        a = torch.randn(100, 100, device='cuda')
        b = torch.randn(100, 100, device='cuda')
        c = torch.mm(a, b)
        print(f"✅ Basic CUDA operations work. Result shape: {c.shape}")
        del a, b, c
        
        # Test 2: Mixed device operations (this often fails)
        print("Test 2: Mixed device operations...")
        cpu_tensor = torch.randn(50, 50)
        gpu_tensor = torch.randn(50, 50, device='cuda')
        
        try:
            # This should fail if there are device issues
            result = torch.mm(cpu_tensor, gpu_tensor)  # This will fail
            print("❌ Mixed device operation unexpectedly succeeded")
        except RuntimeError as e:
            if "Expected all tensors to be on the same device" in str(e):
                print("✅ Mixed device operation correctly failed (this is expected)")
            else:
                print(f"❌ Unexpected error: {e}")
        
        del cpu_tensor, gpu_tensor
        
        # Test 3: Device transfer
        print("Test 3: Device transfer operations...")
        cpu_tensor = torch.randn(100, 100)
        gpu_tensor = cpu_tensor.to('cuda')
        back_to_cpu = gpu_tensor.to('cpu')
        print("✅ Device transfers work correctly")
        del cpu_tensor, gpu_tensor, back_to_cpu
        
        # Test 4: Critical operations that fail in Qwen-Image
        print("Test 4: Critical operations (addmm, bmm)...")
        
        # Test addmm (the specific failing operation)
        a = torch.randn(10, device='cuda')
        b = torch.randn(10, 20, device='cuda')
        c = torch.randn(20, 15, device='cuda')
        result_addmm = torch.addmm(a[:15], b.t(), c)
        print("✅ addmm operation successful")
        
        # Test bmm
        batch1 = torch.randn(10, 3, 4, device='cuda')
        batch2 = torch.randn(10, 4, 5, device='cuda')
        result_bmm = torch.bmm(batch1, batch2)
        print("✅ bmm operation successful")
        
        del a, b, c, result_addmm, batch1, batch2, result_bmm
        
        # Clear after tests
        torch.cuda.empty_cache()
        print("✅ All device tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Device test failed: {e}")
        return False

def create_cpu_fallback_config():
    """Create a CPU-only configuration file"""
    print("📝 Creating CPU fallback configuration...")
    
    config_content = '''
# Emergency CPU-only configuration for Qwen-Image
# Use this if GPU issues persist

from typing import Any, Dict, Tuple
import torch

# Force CPU usage
MODEL_CONFIG: Dict[str, Any] = {
    "model_name": "Qwen/Qwen-Image",
    "torch_dtype": torch.float32,  # Use float32 for CPU
    "device": "cpu",  # Force CPU
    "use_safetensors": True,
}

# CPU-optimized memory settings
MEMORY_CONFIG: Dict[str, bool] = {
    "enable_attention_slicing": True,
    "enable_cpu_offload": False,  # Not needed for CPU-only
    "enable_sequential_cpu_offload": False,
}

# CPU-optimized generation settings (lower resolution for speed)
GENERATION_CONFIG: Dict[str, Any] = {
    "width": 512,  # Lower resolution for CPU
    "height": 512,
    "num_inference_steps": 20,  # Fewer steps for CPU
    "true_cfg_scale": 3.0,
    "max_batch_size": 1,
}

# Rest of the configuration...
ASPECT_RATIOS: Dict[str, Tuple[int, int]] = {
    "1:1": (512, 512),     # Square
    "16:9": (512, 288),    # Landscape (smaller for CPU)
    "9:16": (288, 512),    # Portrait  
    "4:3": (512, 384),     # Photo
    "3:4": (384, 512),     # Portrait photo
    "21:9": (512, 220),    # Ultra-wide
}

# Simplified prompt enhancement
PROMPT_ENHANCEMENT: Dict[str, Dict[str, str]] = {
    "en": {
        "quality_keywords": "high quality, detailed",
        "artistic_keywords": "professional",
        "technical_keywords": "clear, sharp"
    },
    "zh": {
        "quality_keywords": "高质量，细节丰富",
        "artistic_keywords": "专业",
        "technical_keywords": "清晰，锐利"
    }
}
'''
    
    with open('qwen_image_config_cpu_fallback.py', 'w') as f:
        f.write(config_content)
    
    print("✅ CPU fallback config created: qwen_image_config_cpu_fallback.py")
    print("💡 To use: rename to qwen_image_config.py (backup original first)")

def create_gpu_safe_config():
    """Create a GPU-safe configuration with conservative settings"""
    print("📝 Creating GPU-safe configuration...")
    
    config_content = '''
# GPU-safe configuration for Qwen-Image
# Conservative settings to avoid device mismatch

from typing import Any, Dict, Tuple
import torch

# Conservative GPU settings
MODEL_CONFIG: Dict[str, Any] = {
    "model_name": "Qwen/Qwen-Image",
    "torch_dtype": torch.float16,  # Use float16 instead of bfloat16
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "use_safetensors": True,
}

# Conservative memory settings - DISABLE potentially problematic optimizations
MEMORY_CONFIG: Dict[str, bool] = {
    "enable_attention_slicing": False,     # DISABLED - can cause device issues
    "enable_cpu_offload": False,           # DISABLED - major source of device conflicts
    "enable_sequential_cpu_offload": False, # DISABLED - also causes device issues
}

# Conservative generation settings
GENERATION_CONFIG: Dict[str, Any] = {
    "width": 512,   # Smaller to reduce memory pressure
    "height": 512,
    "num_inference_steps": 30,  # Reasonable quality
    "true_cfg_scale": 4.0,
    "max_batch_size": 1,
}
'''
    
    with open('qwen_image_config_gpu_safe.py', 'w') as f:
        f.write(config_content)
    
    print("✅ GPU-safe config created: qwen_image_config_gpu_safe.py")
    print("💡 This disables memory optimizations that can cause device conflicts")

def main():
    """Main emergency fix function"""
    print(f"🚨 Emergency Device Fix - {datetime.now()}")
    print("=" * 60)
    
    # Step 1: System diagnosis
    diagnose_system()
    
    # Step 2: Clear CUDA
    cuda_cleared = clear_all_cuda()
    
    # Step 3: Test operations
    device_works = test_device_operations()
    
    # Step 4: Recommendations
    print("📋 Recommendations")
    print("=" * 50)
    
    if not cuda_cleared:
        print("❌ CUDA cleanup failed - consider restarting Python/terminal")
    
    if not device_works:
        print("❌ Device operations failed - GPU may have issues")
        print("💡 Try these solutions in order:")
        print("   1. Restart the Python process")
        print("   2. Check nvidia-smi for GPU status")
        print("   3. Restart WSL: wsl --shutdown")
        print("   4. Use CPU fallback configuration")
        print("   5. Try GPU-safe configuration (disables problematic optimizations)")
        
        create_cpu_fallback_config()
        create_gpu_safe_config()
    else:
        print("✅ Device operations working - issue may be model-specific")
        print("💡 Try these solutions:")
        print("   1. Restart the UI with: ./restart_ui.sh")
        print("   2. Use GPU-safe configuration to disable problematic optimizations")
        
        create_gpu_safe_config()
    
    print("\n🔧 Advanced troubleshooting:")
    print("   - Check for memory leaks: nvidia-smi")
    print("   - Monitor system resources: htop")
    print("   - Check WSL2 GPU support: nvidia-smi.exe (from Windows)")
    print("   - Verify CUDA installation: nvcc --version")
    print("   - Check driver compatibility with PyTorch")
    
    print("\n🚀 Quick fixes to try:")
    print("   1. Replace current config with GPU-safe config")
    print("   2. Reduce image dimensions to 512x512")
    print("   3. Disable all memory optimizations")
    print("   4. Use CPU mode temporarily")

if __name__ == "__main__":
    main()