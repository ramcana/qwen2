#!/usr/bin/env python3
"""
Test Qwen-Image-Edit Loading with Fixed Configuration
"""

import os
import sys

import torch

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def test_qwen_edit_loading():
    """Test the fixed Qwen-Image-Edit loading"""
    print("🧪 Testing fixed Qwen-Image-Edit loading...")
    print("=" * 50)
    
    try:
        # Check if diffusers is available
        from diffusers import QwenImageEditPipeline
        print("✅ QwenImageEditPipeline available")
        
        # Test basic loading (without device_map="auto")
        print("📥 Testing model loading with fixed configuration...")
        
        pipeline = QwenImageEditPipeline.from_pretrained(
            "Qwen/Qwen-Image-Edit",
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=False,  # Better for 128GB RAM
            resume_download=True,
            use_safetensors=True,
            local_files_only=True     # Only test if already downloaded
        )
        
        print("✅ Model loading successful!")
        
        # Test device movement
        if torch.cuda.is_available():
            print("🔄 Testing GPU movement...")
            pipeline = pipeline.to("cuda")
            print("✅ GPU movement successful!")
            
            # Check component devices
            components = ['unet', 'vae', 'text_encoder']
            for comp_name in components:
                if hasattr(pipeline, comp_name):
                    component = getattr(pipeline, comp_name)
                    if component is not None:
                        try:
                            device = str(next(component.parameters()).device)
                            print(f"   {comp_name.upper()}: {device}")
                        except Exception:
                            print(f"   {comp_name.upper()}: device check failed")
        
        del pipeline
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print("\n🎉 All tests passed! Qwen-Image-Edit should work correctly now.")
        return True
        
    except Exception as e:
        error_msg = str(e)
        print(f"❌ Test failed: {e}")
        
        if "local_files_only" in error_msg:
            print("\n💡 Model not downloaded yet. Run:")
            print("python tools/download_qwen_edit_hub.py")
        elif "auto not supported" in error_msg:
            print("\n❌ device_map issue still present!")
        else:
            print(f"\n💡 Error type: {type(e).__name__}")
        
        return False

def test_generator_integration():
    """Test with the actual generator class"""
    print("\n🔧 Testing QwenImageGenerator integration...")
    
    try:
        from qwen_generator import QwenImageGenerator
        
        generator = QwenImageGenerator()
        success = generator.load_model()
        
        if success:
            if generator.edit_pipe:
                print("✅ Enhanced features loaded successfully!")
            else:
                print("⚠️ Basic model loaded, enhanced features need download")
        else:
            print("❌ Model loading failed")
            
        return success
        
    except Exception as e:
        print(f"❌ Generator integration failed: {e}")
        return False

if __name__ == "__main__":
    print("🎯 Qwen-Image-Edit Fix Verification")
    print("=" * 50)
    
    # Test 1: Direct pipeline loading
    test1_success = test_qwen_edit_loading()
    
    # Test 2: Generator integration
    test2_success = test_generator_integration()
    
    print("\n" + "=" * 50)
    print("📋 Test Results:")
    print(f"   Direct loading: {'✅ PASS' if test1_success else '❌ FAIL'}")
    print(f"   Generator integration: {'✅ PASS' if test2_success else '❌ FAIL'}")
    
    if test1_success and test2_success:
        print("\n🎉 All tests passed! The fix is working correctly.")
    elif not test1_success:
        print("\n🔧 Run the download first: python tools/download_qwen_edit_hub.py")
    else:
        print("\n⚠️ Some issues remain. Check the error messages above.")