#!/usr/bin/env python3
"""
Device Test Script for Qwen2
Diagnoses CUDA and device setup issues
"""

import os
import sys

import torch

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_cuda_setup():
    """Test CUDA availability and configuration"""
    print("🔍 CUDA Device Test")
    print("=" * 50)
    
    # Basic CUDA check
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"PyTorch version: {torch.__version__}")
        print(f"GPU count: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"GPU {i}: {props.name}")
            print(f"   Memory: {props.total_memory / 1e9:.1f}GB")
            print(f"   Compute capability: {props.major}.{props.minor}")
        
        # Test tensor operations
        print("\n🧪 Testing tensor operations...")
        try:
            # Test basic operations
            device = "cuda:0"
            x = torch.randn(100, 100, device=device)
            y = torch.randn(100, 100, device=device)
            z = torch.mm(x, y)
            print(f"✅ Matrix multiplication on {device}: OK")
            
            # Test generator
            generator = torch.Generator(device=device).manual_seed(42)
            random_tensor = torch.randn(10, device=device, generator=generator)
            print(f"✅ Random generation on {device}: OK")
            
            # Clear cache
            torch.cuda.empty_cache()
            print("✅ CUDA cache cleared: OK")
            
        except Exception as e:
            print(f"❌ CUDA operations failed: {e}")
            return False
    else:
        print("⚠️ CUDA not available - will use CPU")
    
    return torch.cuda.is_available()

def test_model_components():
    """Test model component device handling"""
    print("\n🔍 Model Component Test")
    print("=" * 50)
    
    try:
        from src.qwen_generator import QwenImageGenerator

        # Create generator
        generator = QwenImageGenerator()
        print(f"Generator device: {generator.device}")
        
        # Don't actually load the model, just test the setup
        print("✅ Generator initialization: OK")
        
        return True
        
    except Exception as e:
        print(f"❌ Model component test failed: {e}")
        return False

def main():
    """Run all device tests"""
    print("🚀 Qwen2 Device Diagnostic Test")
    print("=" * 60)
    
    # Test CUDA
    cuda_ok = test_cuda_setup()
    
    # Test model components
    model_ok = test_model_components()
    
    print("\n📊 Test Summary")
    print("=" * 50)
    print(f"CUDA Setup: {'✅ PASS' if cuda_ok else '❌ FAIL'}")
    print(f"Model Setup: {'✅ PASS' if model_ok else '❌ FAIL'}")
    
    if cuda_ok and model_ok:
        print("\n🎉 All tests passed! Device setup is correct.")
        print("💡 If you still get device errors during generation:")
        print("   1. Restart the UI application")
        print("   2. Try with smaller image dimensions")
        print("   3. Enable CPU offload in config")
    else:
        print("\n⚠️ Some tests failed. Check the errors above.")
        print("💡 Troubleshooting:")
        if not cuda_ok:
            print("   - Verify NVIDIA drivers are installed")
            print("   - Check CUDA installation")
            print("   - Restart WSL2: wsl --shutdown")
        if not model_ok:
            print("   - Check virtual environment activation")
            print("   - Verify project dependencies")

if __name__ == "__main__":
    main()