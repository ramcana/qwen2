#!/usr/bin/env python3
"""
Diagnose GPU performance issues with Qwen Image Edit.
"""
import torch
import time
from diffusers import QwenImageEditPipeline
import subprocess
import os


def run_gpu_benchmark():
    """Run a simple GPU benchmark."""
    print("🔥 GPU Benchmark Test")
    print("-" * 30)
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return
    
    device = torch.device("cuda")
    
    # Simple tensor operations benchmark
    print("Testing tensor operations...")
    start = time.time()
    
    # Create large tensors and do operations
    a = torch.randn(1000, 1000, device=device, dtype=torch.float16)
    b = torch.randn(1000, 1000, device=device, dtype=torch.float16)
    
    for i in range(100):
        c = torch.matmul(a, b)
        torch.cuda.synchronize()
    
    tensor_time = time.time() - start
    print(f"✅ Tensor ops: {tensor_time:.2f}s for 100 matrix multiplications")
    
    if tensor_time > 5:
        print("⚠️ Tensor operations are slow - GPU issue!")
    else:
        print("✅ Tensor operations are fast - GPU working")
    
    return tensor_time < 5


def check_model_files():
    """Check if model files are corrupted or incomplete."""
    print("\n📂 Model File Check")
    print("-" * 30)
    
    model_path = "./models/Qwen-Image-Edit"
    
    if not os.path.exists(model_path):
        print("❌ Model directory not found")
        return False
    
    # Check key files
    key_files = [
        "model_index.json",
        "scheduler/scheduler_config.json",
        "text_encoder/config.json",
        "transformer/config.json",
    ]
    
    missing = []
    for file in key_files:
        if not os.path.exists(os.path.join(model_path, file)):
            missing.append(file)
    
    if missing:
        print(f"❌ Missing files: {missing}")
        return False
    
    print("✅ All key model files present")
    
    # Check model sizes
    total_size = 0
    for root, dirs, files in os.walk(model_path):
        for file in files:
            if file.endswith('.safetensors'):
                filepath = os.path.join(root, file)
                size = os.path.getsize(filepath)
                total_size += size
    
    total_gb = total_size / (1024**3)
    print(f"📊 Model size: {total_gb:.1f} GB")
    
    if total_gb < 15:
        print("⚠️ Model seems incomplete (should be ~20GB)")
        return False
    
    print("✅ Model size looks correct")
    return True


def check_nvidia_driver():
    """Check NVIDIA driver and CUDA setup."""
    print("\n🔧 NVIDIA Driver Check")
    print("-" * 30)
    
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ nvidia-smi working")
            # Extract driver version
            lines = result.stdout.split('\n')
            for line in lines:
                if 'Driver Version' in line:
                    print(f"📋 {line.strip()}")
                    break
        else:
            print("❌ nvidia-smi failed")
            return False
    except:
        print("❌ nvidia-smi not found")
        return False
    
    # Check PyTorch CUDA
    print(f"🐍 PyTorch CUDA: {torch.version.cuda}")
    print(f"🔥 GPU: {torch.cuda.get_device_name(0)}")
    
    return True


def test_simple_diffusion():
    """Test a simple diffusion operation."""
    print("\n🧪 Simple Diffusion Test")
    print("-" * 30)
    
    try:
        # Test basic pipeline loading without full model
        print("Testing pipeline import...")
        from diffusers import DDPMScheduler
        scheduler = DDPMScheduler()
        print("✅ Diffusers working")
        
        # Test basic tensor operations on GPU
        print("Testing GPU tensor operations...")
        device = torch.device("cuda")
        x = torch.randn(1, 4, 64, 64, device=device, dtype=torch.float16)
        
        # Simulate some diffusion-like operations
        start = time.time()
        for i in range(10):
            noise = torch.randn_like(x)
            x_noisy = x + noise * 0.1
            torch.cuda.synchronize()
        
        op_time = time.time() - start
        print(f"✅ GPU operations: {op_time:.3f}s")
        
        if op_time > 1:
            print("⚠️ GPU operations are slow")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Diffusion test failed: {e}")
        return False


def main():
    """Run all diagnostics."""
    print("🔍 Qwen Image Edit GPU Diagnostics")
    print("=" * 50)
    
    # Run all checks
    gpu_ok = run_gpu_benchmark()
    model_ok = check_model_files()
    driver_ok = check_nvidia_driver()
    diffusion_ok = test_simple_diffusion()
    
    print("\n📋 Summary")
    print("=" * 20)
    print(f"GPU Performance: {'✅' if gpu_ok else '❌'}")
    print(f"Model Files: {'✅' if model_ok else '❌'}")
    print(f"NVIDIA Driver: {'✅' if driver_ok else '❌'}")
    print(f"Diffusion Test: {'✅' if diffusion_ok else '❌'}")
    
    if all([gpu_ok, model_ok, driver_ok, diffusion_ok]):
        print("\n✅ All checks passed - issue might be with pipeline optimization")
        print("💡 Try using smaller models or different settings")
    else:
        print("\n❌ Found issues - fix these first")
        
        if not gpu_ok:
            print("🔧 GPU Issue: Check NVIDIA drivers, restart system")
        if not model_ok:
            print("🔧 Model Issue: Re-download model with 'make models'")
        if not driver_ok:
            print("🔧 Driver Issue: Update NVIDIA drivers")
        if not diffusion_ok:
            print("🔧 Diffusion Issue: Reinstall diffusers library")


if __name__ == "__main__":
    main()