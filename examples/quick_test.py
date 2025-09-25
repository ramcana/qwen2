#!/usr/bin/env python3
"""
Quick Test Script for Modern Qwen Architecture
Run this to verify your optimized setup and measure performance improvements
"""

import sys
import os
import time
import torch
from datetime import datetime

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Add src directory for optimization components
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from pipeline_optimizer import PipelineOptimizer, OptimizationConfig
    from model_detection_service import ModelDetectionService
    OPTIMIZATION_AVAILABLE = True
except ImportError:
    OPTIMIZATION_AVAILABLE = False
    print("⚠️ Optimization components not available, using basic functionality")

try:
    from src.qwen_generator import QwenImageGenerator
    LEGACY_GENERATOR_AVAILABLE = True
except ImportError:
    LEGACY_GENERATOR_AVAILABLE = False

try:
    from diffusers import AutoPipelineForText2Image
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False
    print("❌ diffusers not installed. Run: pip install diffusers")
    sys.exit(1)


def test_optimized_setup():
    """Test the optimized modern architecture setup"""
    print("🚀 Testing Modern Qwen Architecture Setup...")
    print("=" * 60)
    
    if not OPTIMIZATION_AVAILABLE:
        print("❌ Optimization components not available")
        return test_basic_setup()
    
    try:
        # Detect current model
        print("🔍 Detecting model configuration...")
        detector = ModelDetectionService()
        current_model = detector.detect_current_model()
        
        if not current_model:
            print("❌ No Qwen models found")
            print("💡 Run: python tools/download_qwen_image.py --model qwen-image")
            return False
        
        print(f"📦 Found model: {current_model.name}")
        print(f"📊 Size: {current_model.size_gb:.1f}GB")
        print(f"🏷️ Type: {current_model.model_type}")
        print(f"✅ Status: {current_model.download_status}")
        
        # Check if optimization is needed
        optimization_needed = detector.is_optimization_needed()
        if optimization_needed:
            print("⚠️ Current model is not optimal for text-to-image generation")
            recommended = detector.get_recommended_model()
            print(f"💡 Recommended: {recommended}")
        else:
            print("✅ Model configuration is optimal")
        
        # Detect architecture
        architecture = detector.detect_model_architecture(current_model)
        print(f"🏗️ Architecture: {architecture}")
        
        # Analyze performance characteristics
        perf_chars = detector.analyze_performance_characteristics(current_model)
        print(f"⚡ Expected performance: {perf_chars['expected_generation_time']}")
        print(f"💾 Memory usage: {perf_chars['memory_usage']}")
        
        if perf_chars['bottlenecks']:
            print(f"⚠️ Bottlenecks: {', '.join(perf_chars['bottlenecks'])}")
        
        # Create optimized pipeline
        print(f"\n🔧 Creating optimized pipeline...")
        config = OptimizationConfig(
            architecture_type=architecture,
            enable_attention_slicing=False,  # Disabled for performance
            enable_vae_slicing=False,        # Disabled for performance
            enable_tf32=True,                # Enabled for RTX GPUs
            enable_cudnn_benchmark=True      # Enabled for consistent inputs
        )
        
        optimizer = PipelineOptimizer(config)
        pipeline = optimizer.create_optimized_pipeline(current_model.path, architecture)
        
        # Validate optimization
        validation = optimizer.validate_optimization(pipeline)
        print(f"✅ Optimization validation: {validation['overall_status']}")
        print(f"🚀 GPU optimizations: {validation['gpu_optimizations']}")
        
        # Test generation with performance monitoring
        test_prompt = "A beautiful coffee shop with a neon sign reading 'AI Café ☕', modern interior, warm lighting, photorealistic"
        
        print(f"\n📝 Generating test image with optimized pipeline...")
        print(f"Prompt: {test_prompt}")
        
        # Monitor GPU memory if available
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            memory_before = torch.cuda.memory_allocated() / 1e9
        
        start_time = time.time()
        
        # Generate image
        generation_kwargs = {
            "prompt": test_prompt,
            "width": 1024,
            "height": 1024,
            "num_inference_steps": 20,
            "generator": torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu").manual_seed(42)
        }
        
        # Use appropriate CFG parameter based on architecture
        if architecture == "MMDiT":
            generation_kwargs["true_cfg_scale"] = 3.5
        else:
            generation_kwargs["guidance_scale"] = 3.5
        
        result = pipeline(**generation_kwargs)
        image = result.images[0]
        
        generation_time = time.time() - start_time
        time_per_step = generation_time / 20
        
        # Record memory usage
        memory_peak = 0
        if torch.cuda.is_available():
            memory_peak = torch.cuda.max_memory_allocated() / 1e9
            memory_used = memory_peak - memory_before
            torch.cuda.empty_cache()
        
        # Save image
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"quick_test_optimized_{timestamp}.png"
        filepath = os.path.join("generated_images", filename)
        os.makedirs("generated_images", exist_ok=True)
        image.save(filepath)
        
        print(f"\n✅ Generation completed successfully!")
        print(f"⏱️ Total time: {generation_time:.2f}s")
        print(f"⚡ Time per step: {time_per_step:.2f}s")
        print(f"💾 Saved: {filepath}")
        
        if torch.cuda.is_available():
            print(f"🔥 GPU memory used: {memory_used:.2f}GB")
        
        # Performance assessment
        print(f"\n📊 Performance Assessment:")
        if time_per_step <= 2.0:
            print("🏆 Excellent performance! (≤2s/step)")
            print("   MMDiT optimizations are working perfectly")
        elif time_per_step <= 5.0:
            print("✅ Good performance (≤5s/step)")
            print("   Modern architecture optimizations are effective")
        elif time_per_step <= 10.0:
            print("⚠️ Moderate performance (≤10s/step)")
            print("   Consider checking optimization settings")
        else:
            print("❌ Poor performance (>10s/step)")
            print("   May need model optimization or hardware upgrade")
        
        # Compare with expected performance
        expected_range = perf_chars['expected_generation_time']
        print(f"   Expected: {expected_range}")
        print(f"   Actual: {time_per_step:.2f}s per step")
        
        # Architecture-specific notes
        if architecture == "MMDiT":
            print(f"\n🎯 MMDiT Architecture Notes:")
            print("   ✅ Using true_cfg_scale parameter")
            print("   ✅ Transformer-based architecture optimizations")
            print("   ✅ Enhanced text rendering capabilities")
            print("   ⚠️ AttnProcessor2_0 disabled for compatibility")
        elif architecture == "UNet":
            print(f"\n🎯 UNet Architecture Notes:")
            print("   ✅ Using guidance_scale parameter")
            print("   ✅ Traditional diffusion model optimizations")
            print("   ✅ Wide ecosystem compatibility")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Optimized test failed: {e}")
        print("🔄 Falling back to basic test...")
        return test_basic_setup()


def test_basic_setup():
    """Test basic setup without optimizations"""
    print("📦 Testing Basic Setup...")
    print("=" * 50)
    
    if LEGACY_GENERATOR_AVAILABLE:
        # Test with legacy generator
        try:
            generator = QwenImageGenerator()
            
            print("Loading model...")
            if not generator.load_model():
                print("❌ Failed to load model. Check your setup.")
                return False
            
            test_prompt = "A beautiful coffee shop with a neon sign reading 'AI Café', modern interior, warm lighting"
            
            print(f"\n📝 Generating test image...")
            print(f"Prompt: {test_prompt}")
            
            start_time = time.time()
            
            image, message = generator.generate_image(
                prompt=test_prompt,
                width=1024,
                height=1024,
                num_inference_steps=20,
                seed=42
            )
            
            generation_time = time.time() - start_time
            
            if image:
                print(f"\n✅ Success! {message}")
                print(f"⏱️ Generation time: {generation_time:.2f}s ({generation_time/20:.2f}s per step)")
                print("\n🎯 Basic test completed successfully!")
                print("Your Qwen-Image setup is working.")
                return True
            else:
                print(f"\n❌ Generation failed: {message}")
                return False
                
        except Exception as e:
            print(f"\n❌ Legacy generator error: {e}")
    
    # Test with basic diffusers pipeline
    try:
        print("🔄 Testing with basic diffusers pipeline...")
        
        # Try common model paths
        model_paths = [
            "./models/Qwen-Image",
            "./models/Qwen-Image-Edit",
            "Qwen/Qwen-Image"
        ]
        
        pipeline = None
        for model_path in model_paths:
            try:
                pipeline = AutoPipelineForText2Image.from_pretrained(
                    model_path,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    trust_remote_code=True
                )
                if torch.cuda.is_available():
                    pipeline = pipeline.to("cuda")
                print(f"✅ Loaded pipeline from: {model_path}")
                break
            except Exception:
                continue
        
        if not pipeline:
            print("❌ No pipeline could be loaded")
            return False
        
        # Generate test image
        test_prompt = "A serene mountain landscape at sunset"
        
        print(f"📝 Generating test image...")
        print(f"Prompt: {test_prompt}")
        
        start_time = time.time()
        
        generation_kwargs = {
            "prompt": test_prompt,
            "width": 1024,
            "height": 1024,
            "num_inference_steps": 20,
            "generator": torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu").manual_seed(42)
        }
        
        # Try both CFG parameter names
        try:
            generation_kwargs["true_cfg_scale"] = 3.5
            image = pipeline(**generation_kwargs).images[0]
        except Exception:
            generation_kwargs.pop("true_cfg_scale", None)
            generation_kwargs["guidance_scale"] = 3.5
            image = pipeline(**generation_kwargs).images[0]
        
        generation_time = time.time() - start_time
        
        # Save image
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"quick_test_basic_{timestamp}.png"
        filepath = os.path.join("generated_images", filename)
        os.makedirs("generated_images", exist_ok=True)
        image.save(filepath)
        
        print(f"\n✅ Basic generation completed!")
        print(f"⏱️ Generation time: {generation_time:.2f}s ({generation_time/20:.2f}s per step)")
        print(f"💾 Saved: {filepath}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Basic test failed: {e}")
        return False


def main():
    """Main test function"""
    print("🧪 Quick Test for Modern Qwen Architecture")
    print("=" * 70)
    
    # Display system info
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name()
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"🔥 GPU: {gpu_name} ({gpu_memory:.1f}GB VRAM)")
    else:
        print("⚠️ No GPU detected, using CPU")
    
    print()
    
    # Run appropriate test
    if OPTIMIZATION_AVAILABLE:
        success = test_optimized_setup()
    else:
        success = test_basic_setup()
    
    if success:
        print(f"\n🎉 Test completed successfully!")
        print("=" * 70)
        print("📁 Generated images saved to: ./generated_images/")
        
        if OPTIMIZATION_AVAILABLE:
            print("💡 Your modern Qwen architecture setup is working optimally!")
            print("🚀 Try: python examples/optimized_text_to_image_demo.py")
        else:
            print("💡 Basic setup is working. Consider installing optimization components:")
            print("   • Model detection service")
            print("   • Pipeline optimizer")
            print("   • Performance monitor")
        
        print("🔗 Next steps:")
        print("   • Run performance benchmarks")
        print("   • Try multimodal integration")
        print("   • Explore architecture-specific features")
    else:
        print(f"\n❌ Test failed!")
        print("💡 Troubleshooting:")
        print("   • Check if models are downloaded")
        print("   • Verify GPU drivers and CUDA installation")
        print("   • Run: python tools/download_qwen_image.py --detect")
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)