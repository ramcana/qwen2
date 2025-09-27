#!/usr/bin/env python3
"""
Practical solution: Use models that actually fit on RTX 4080 (16GB)
"""
import torch
import gc
from pathlib import Path

def clear_memory():
    """Clear GPU memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()

def check_gpu_status():
    """Check GPU status"""
    if not torch.cuda.is_available():
        return {"available": False}
    
    total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    allocated = torch.cuda.memory_allocated() / (1024**3)
    
    return {
        "available": True,
        "name": torch.cuda.get_device_name(0),
        "total_gb": total,
        "allocated_gb": allocated,
        "free_gb": total - allocated
    }

def test_sdxl_turbo():
    """Test SDXL-Turbo (fits on 16GB GPU)"""
    print("🧪 Testing SDXL-Turbo (8GB model)")
    print("=" * 35)
    
    try:
        from diffusers import AutoPipelineForText2Image
        import torch
        
        clear_memory()
        
        # Check memory before
        gpu_info = check_gpu_status()
        print(f"📊 GPU before: {gpu_info['allocated_gb']:.2f}GB / {gpu_info['total_gb']:.1f}GB")
        
        # Load SDXL-Turbo (much smaller model)
        print("📁 Loading SDXL-Turbo...")
        pipe = AutoPipelineForText2Image.from_pretrained(
            "stabilityai/sdxl-turbo",
            torch_dtype=torch.float16,
            variant="fp16"
        )
        pipe = pipe.to("cuda")
        
        # Check memory after loading
        gpu_info = check_gpu_status()
        print(f"📊 GPU after loading: {gpu_info['allocated_gb']:.2f}GB / {gpu_info['total_gb']:.1f}GB")
        
        # Generate test image
        print("🎨 Generating test image...")
        prompt = "A beautiful mountain landscape with clear blue sky"
        
        image = pipe(
            prompt=prompt,
            num_inference_steps=1,  # SDXL-Turbo only needs 1 step
            guidance_scale=0.0,     # No guidance needed
            width=512,
            height=512
        ).images[0]
        
        # Save result
        output_file = "sdxl_turbo_test.jpg"
        image.save(output_file)
        
        print(f"✅ SDXL-Turbo test successful!")
        print(f"💾 Saved: {output_file}")
        
        # Final memory check
        gpu_info = check_gpu_status()
        print(f"📊 Final GPU usage: {gpu_info['allocated_gb']:.2f}GB ({gpu_info['allocated_gb']/gpu_info['total_gb']*100:.1f}%)")
        
        return True
        
    except Exception as e:
        print(f"❌ SDXL-Turbo test failed: {e}")
        return False

def test_flux_schnell():
    """Test Flux.1-schnell (faster alternative)"""
    print("\n🧪 Testing Flux.1-schnell")
    print("=" * 25)
    
    try:
        from diffusers import FluxPipeline
        import torch
        
        clear_memory()
        
        print("📁 Loading Flux.1-schnell...")
        pipe = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-schnell",
            torch_dtype=torch.bfloat16
        )
        pipe.enable_model_cpu_offload()  # Save VRAM
        
        # Generate test
        print("🎨 Generating with Flux...")
        prompt = "A serene mountain landscape"
        
        image = pipe(
            prompt,
            guidance_scale=0.0,
            num_inference_steps=4,
            max_sequence_length=256,
            width=512,
            height=512
        ).images[0]
        
        output_file = "flux_schnell_test.jpg"
        image.save(output_file)
        
        print(f"✅ Flux test successful!")
        print(f"💾 Saved: {output_file}")
        
        return True
        
    except Exception as e:
        print(f"❌ Flux test failed: {e}")
        return False

def suggest_qwen_alternatives():
    """Suggest practical alternatives to local Qwen-Image"""
    print("\n💡 Qwen-Image Alternatives")
    print("=" * 30)
    
    print("🌐 Online Options (Recommended):")
    print("1. HuggingFace Spaces:")
    print("   - multimodalart/Qwen-Image-Fast")
    print("   - InstantX/Qwen-Image-ControlNet")
    print("   - Free, no local resources needed")
    
    print("\n🔧 API Options:")
    print("1. Replicate API")
    print("2. HuggingFace Inference API")
    print("3. Together AI")
    
    print("\n🖥️ Hardware Upgrade Options:")
    print("1. RTX 4090 (24GB VRAM)")
    print("2. RTX A6000 (48GB VRAM)")
    print("3. Cloud GPU (A100, H100)")
    
    print("\n📱 Local Alternatives (16GB Compatible):")
    print("1. SDXL-Turbo (8GB, very fast)")
    print("2. Stable Diffusion XL (12GB)")
    print("3. Flux.1-schnell (16GB with offloading)")

def create_working_config():
    """Create configuration for working alternatives"""
    print("\n📝 Creating Working Configuration")
    print("=" * 35)
    
    config = {
        "working_models_16gb": {
            "sdxl_turbo": {
                "model_id": "stabilityai/sdxl-turbo",
                "memory_usage": "~8GB",
                "speed": "Very Fast (1 step)",
                "quality": "Good",
                "recommended": True
            },
            "sdxl_base": {
                "model_id": "stabilityai/stable-diffusion-xl-base-1.0",
                "memory_usage": "~12GB",
                "speed": "Fast (20-30 steps)",
                "quality": "High",
                "recommended": True
            },
            "flux_schnell": {
                "model_id": "black-forest-labs/FLUX.1-schnell",
                "memory_usage": "~16GB (with CPU offload)",
                "speed": "Fast (4 steps)",
                "quality": "Very High",
                "recommended": False,
                "note": "Requires CPU offloading"
            }
        },
        "qwen_image_alternatives": {
            "online_spaces": [
                {
                    "name": "Qwen-Image-Fast",
                    "url": "https://huggingface.co/spaces/multimodalart/Qwen-Image-Fast",
                    "description": "8-step fast generation"
                },
                {
                    "name": "Qwen-Image-ControlNet",
                    "url": "https://huggingface.co/spaces/InstantX/Qwen-Image-ControlNet",
                    "description": "With ControlNet support"
                }
            ],
            "api_services": [
                "Replicate",
                "HuggingFace Inference API",
                "Together AI"
            ]
        },
        "hardware_requirements": {
            "current_gpu": "RTX 4080 (16GB)",
            "qwen_image_requirement": "24GB+ VRAM",
            "recommendation": "Use online services or upgrade GPU"
        }
    }
    
    import json
    config_path = Path("config/working_alternatives.json")
    config_path.parent.mkdir(exist_ok=True)
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"✅ Configuration saved: {config_path}")

def main():
    """Main practical solution"""
    print("🎯 Practical Solution for RTX 4080 (16GB)")
    print("=" * 45)
    
    # Check GPU
    gpu_info = check_gpu_status()
    if not gpu_info["available"]:
        print("❌ No GPU available")
        return False
    
    print(f"🎮 GPU: {gpu_info['name']}")
    print(f"📊 VRAM: {gpu_info['total_gb']:.1f}GB")
    print(f"📊 Current usage: {gpu_info['allocated_gb']:.2f}GB")
    
    # Test working alternatives
    print("\n🧪 Testing Working Alternatives...")
    
    # Test 1: SDXL-Turbo (most likely to work)
    sdxl_success = test_sdxl_turbo()
    
    # Test 2: Flux (if SDXL works)
    flux_success = False
    if sdxl_success:
        try:
            flux_success = test_flux_schnell()
        except:
            print("⚠️ Flux test skipped (optional)")
    
    # Provide alternatives
    suggest_qwen_alternatives()
    
    # Create config
    create_working_config()
    
    # Summary
    print(f"\n📋 Test Results:")
    print(f"   SDXL-Turbo: {'✅' if sdxl_success else '❌'}")
    print(f"   Flux-schnell: {'✅' if flux_success else '❌'}")
    
    if sdxl_success:
        print("\n🎉 Success! You have working alternatives:")
        print("1. Use SDXL-Turbo for fast local generation")
        print("2. Use online Qwen-Image spaces for Qwen-specific features")
        print("3. Consider GPU upgrade for full local Qwen-Image")
    else:
        print("\n⚠️ Local alternatives failed")
        print("💡 Recommended: Use online Qwen-Image spaces")
    
    return sdxl_success

if __name__ == "__main__":
    main()