#!/usr/bin/env python3
"""
Official Qwen-Image Example Script with Modern Architecture Support
Based on the official Hugging Face documentation with MMDiT optimizations
Demonstrates advanced text rendering capabilities and performance improvements
"""

import os
import sys
import time
from datetime import datetime

import torch
from diffusers import DiffusionPipeline, AutoPipelineForText2Image

# Add parent directory for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Add src directory for optimization components
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from pipeline_optimizer import PipelineOptimizer, OptimizationConfig
    from model_detection_service import ModelDetectionService
    OPTIMIZATION_AVAILABLE = True
except ImportError:
    OPTIMIZATION_AVAILABLE = False
    print("⚠️ Optimization components not available, using basic functionality")

def official_qwen_example_optimized():
    """Run the official Qwen-Image example with modern optimizations"""
    
    print("🎨 Official Qwen-Image Example with MMDiT Optimizations")
    print("=" * 60)
    print("Based on: https://huggingface.co/Qwen/Qwen-Image")
    print("Enhanced with: Modern architecture optimizations")
    print()
    
    model_name = "Qwen/Qwen-Image"
    
    # Try to detect and load optimized model
    pipe = None
    if OPTIMIZATION_AVAILABLE:
        try:
            print("🔍 Detecting optimal model configuration...")
            detector = ModelDetectionService()
            current_model = detector.detect_current_model()
            
            if current_model and current_model.download_status == "complete":
                print(f"📦 Found model: {current_model.name}")
                architecture = detector.detect_model_architecture(current_model)
                print(f"🏗️ Architecture: {architecture}")
                
                # Create optimized pipeline
                config = OptimizationConfig(
                    architecture_type=architecture,
                    enable_attention_slicing=False,  # Disabled for performance
                    enable_vae_slicing=False,        # Disabled for performance
                    enable_tf32=True,                # Enabled for RTX GPUs
                    enable_cudnn_benchmark=True      # Enabled for consistent inputs
                )
                
                optimizer = PipelineOptimizer(config)
                pipe = optimizer.create_optimized_pipeline(current_model.path, architecture)
                
                # Validate optimization
                validation = optimizer.validate_optimization(pipe)
                print(f"✅ Optimization status: {validation['overall_status']}")
                print(f"🚀 GPU optimizations: {validation['gpu_optimizations']}")
                
            else:
                print("⚠️ No optimal model found, falling back to basic loading")
        except Exception as e:
            print(f"⚠️ Optimization failed: {e}, falling back to basic loading")
    
    # Fallback to basic loading if optimization not available
    if pipe is None:
        print("📦 Loading Qwen-Image pipeline (basic configuration)...")
        if torch.cuda.is_available():
            torch_dtype = torch.float16  # Use float16 instead of bfloat16 for better compatibility
            device = "cuda"
            print("✅ Using CUDA with float16")
        else:
            torch_dtype = torch.float32
            device = "cpu"
            print("⚠️ Using CPU with float32")
        
        try:
            # Use AutoPipelineForText2Image for better text-to-image performance
            pipe = AutoPipelineForText2Image.from_pretrained(
                model_name, 
                torch_dtype=torch_dtype,
                trust_remote_code=True
            )
            pipe = pipe.to(device)
            print(f"✅ Pipeline loaded successfully on {device}")
        except Exception as e:
            print(f"❌ Failed to load pipeline: {e}")
            return
    
    # Official positive magic strings
    positive_magic = {
        "en": ", Ultra HD, 4K, cinematic composition.",  # for english prompt
        "zh": ", 超清，4K，电影级构图."  # for chinese prompt
    }
    
    # Official example prompt (complex text rendering)
    prompt = '''A coffee shop entrance features a chalkboard sign reading "Qwen Coffee 😊 $2 per cup," with a neon light beside it displaying "通义千问". Next to it hangs a poster showing a beautiful Chinese woman, and beneath the poster is written "π≈3.1415926-53589793-23846264-33832795-02384197". Ultra HD, 4K, cinematic composition'''
    
    negative_prompt = " "  # Empty string as per official docs
    
    # Official aspect ratios
    aspect_ratios = {
        "1:1": (1328, 1328),
        "16:9": (1664, 928),
        "9:16": (928, 1664),
        "4:3": (1472, 1140),
        "3:4": (1140, 1472),
        "3:2": (1584, 1056),
        "2:3": (1056, 1584),
    }
    
    # Use 16:9 ratio as in official example
    width, height = aspect_ratios["16:9"]
    
    print("🎯 Generating image with official example:")
    print(f"   Resolution: {width}x{height}")
    print(f"   Prompt: {prompt[:80]}...")
    print(f"   Using positive magic: {positive_magic['en']}")
    print()
    
    try:
        # Generate with optimized parameters
        print("🎨 Starting generation...")
        start_time = time.time()
        
        # Prepare generation arguments
        generation_kwargs = {
            "prompt": prompt + positive_magic["en"],
            "negative_prompt": negative_prompt,
            "width": width,
            "height": height,
            "num_inference_steps": 30,  # Reduced from 50 for better performance
            "generator": torch.Generator(device=device).manual_seed(42)
        }
        
        # Use appropriate CFG parameter (MMDiT uses true_cfg_scale, UNet uses guidance_scale)
        if hasattr(pipe, 'transformer') or 'qwen' in str(pipe.__class__).lower():
            generation_kwargs["true_cfg_scale"] = 3.5  # Optimized for MMDiT
        else:
            generation_kwargs["guidance_scale"] = 3.5  # Fallback for UNet
        
        image = pipe(**generation_kwargs).images[0]
        
        generation_time = time.time() - start_time
        
        # Save with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"official_qwen_example_{timestamp}.png"
        filepath = os.path.join("generated_images", filename)
        
        # Ensure directory exists
        os.makedirs("generated_images", exist_ok=True)
        
        image.save(filepath)
        
        print("✅ Image generated successfully!")
        print(f"📁 Saved as: {filepath}")
        print(f"⏱️ Generation time: {generation_time:.2f}s ({generation_time/30:.2f}s per step)")
        print()
        print("🎯 This example demonstrates:")
        print("   • Complex text rendering (English + Chinese)")
        print("   • Mathematical symbols and equations")
        print("   • Emoji support")
        print("   • Mixed typography in realistic scenes")
        print("   • Official 'positive magic' enhancement")
        print("   • MMDiT architecture optimizations")
        
        # Performance assessment
        time_per_step = generation_time / 30
        if time_per_step <= 2.0:
            print("🏆 Excellent performance! MMDiT optimizations working well")
        elif time_per_step <= 5.0:
            print("✅ Good performance with modern architecture")
        else:
            print("⚠️ Consider checking GPU optimization settings")
        
    except Exception as e:
        print(f"❌ Generation failed: {e}")
        return
    
    # Demonstrate other capabilities
    print()
    print("🌟 Additional Qwen-Image Capabilities:")
    print("   • Text rendering in 20+ languages")
    print("   • Multiple artistic styles (photorealistic, anime, impressionist)")
    print("   • Image editing and manipulation")
    print("   • Object detection and segmentation")
    print("   • Novel view synthesis")
    print("   • Super-resolution")
    print()
    print("📖 For more examples, check the official documentation:")
    print("   https://huggingface.co/Qwen/Qwen-Image")

def quick_text_test():
    """Quick test with simpler text rendering"""
    
    print("🚀 Quick Text Rendering Test")
    print("=" * 50)
    
    try:
        from src.qwen_generator import QwenImageGenerator
        
        generator = QwenImageGenerator()
        if not generator.load_model():
            print("❌ Failed to load model")
            return
        
        # Test prompt with both English and Chinese
        test_prompt = 'A modern café with a sign reading "AI Coffee Shop 人工智能咖啡店" and a menu board showing "Latte $4 拿铁咖啡"'
        
        print(f"🎯 Testing: {test_prompt}")
        
        image, message = generator.generate_image(
            prompt=test_prompt,
            width=1664,
            height=928,
            num_inference_steps=30,  # Faster for testing
            cfg_scale=4.0,
            seed=42,
            language="en"
        )
        
        if image:
            print("✅ Quick test successful!")
            print(message)
        else:
            print(f"❌ Quick test failed: {message}")
            
    except Exception as e:
        print(f"❌ Quick test error: {e}")

if __name__ == "__main__":
    print("🎨 Qwen-Image Official Examples")
    print("=" * 60)
    print()
    
    choice = input("Choose example:\n1. Official Hugging Face example (complex text)\n2. Quick text test with existing generator\n\nEnter choice (1-2): ").strip()
    
    if choice == "1":
        official_qwen_example_optimized()
    elif choice == "2":
        quick_text_test()
    else:
        print("Running both examples...")
        print()
        official_qwen_example_optimized()
        print("\n" + "="*60 + "\n")
        quick_text_test()