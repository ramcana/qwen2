#!/usr/bin/env python3
"""
Optimized Text-to-Image Generation Demo
Demonstrates MMDiT performance improvements with modern Qwen architecture
"""

import os
import sys
import time
import torch
from datetime import datetime
from pathlib import Path

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from pipeline_optimizer import PipelineOptimizer, OptimizationConfig
    from model_detection_service import ModelDetectionService
    from utils.performance_monitor import PerformanceMonitor
    OPTIMIZATION_AVAILABLE = True
except ImportError:
    OPTIMIZATION_AVAILABLE = False
    print("⚠️ Optimization components not available, using basic functionality")

try:
    from diffusers import AutoPipelineForText2Image, DiffusionPipeline
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False
    print("❌ diffusers not installed. Run: pip install diffusers")
    sys.exit(1)


class OptimizedTextToImageDemo:
    """Demo class for optimized text-to-image generation"""
    
    def __init__(self):
        self.pipeline = None
        self.optimizer = None
        self.performance_monitor = None
        self.model_info = None
        
        # Initialize components if available
        if OPTIMIZATION_AVAILABLE:
            self.detector = ModelDetectionService()
            self.performance_monitor = PerformanceMonitor()
        
        # Demo prompts showcasing different capabilities
        self.demo_prompts = {
            "text_rendering": {
                "prompt": 'A modern coffee shop with a neon sign reading "AI Café ☕" and a chalkboard menu showing "Latte $4.50, Cappuccino $4.00"',
                "description": "Text rendering capabilities"
            },
            "multilingual": {
                "prompt": 'A Japanese restaurant entrance with signs reading "寿司レストラン" and "Welcome 欢迎 환영합니다"',
                "description": "Multilingual text support"
            },
            "complex_scene": {
                "prompt": 'A futuristic cityscape at sunset with flying cars, neon advertisements, and holographic displays showing "2024 Tech Expo"',
                "description": "Complex scene generation"
            },
            "artistic_style": {
                "prompt": 'A serene mountain landscape in impressionist style, with soft brushstrokes and vibrant colors, peaceful lake reflection',
                "description": "Artistic style rendering"
            },
            "photorealistic": {
                "prompt": 'A professional portrait of a person working at a modern computer setup, natural lighting, high detail, photorealistic',
                "description": "Photorealistic generation"
            }
        }
    
    def detect_and_load_model(self) -> bool:
        """Detect and load the optimal model"""
        print("🔍 Detecting optimal model configuration...")
        
        if not OPTIMIZATION_AVAILABLE:
            print("⚠️ Using basic model loading without optimization")
            return self._load_basic_model()
        
        try:
            # Detect current model
            self.model_info = self.detector.detect_current_model()
            
            if not self.model_info:
                print("❌ No Qwen models found")
                print("💡 Run: python tools/download_qwen_image.py --model qwen-image")
                return False
            
            print(f"📦 Found model: {self.model_info.name}")
            print(f"🏗️ Architecture: {self.detector.detect_model_architecture(self.model_info)}")
            print(f"📊 Size: {self.model_info.size_gb:.1f}GB")
            print(f"✅ Status: {self.model_info.download_status}")
            
            # Check if optimization is recommended
            if self.detector.is_optimization_needed():
                print("⚠️ Current model is not optimal for text-to-image generation")
                recommended = self.detector.get_recommended_model()
                print(f"💡 Recommended: {recommended}")
                print("   Consider running: python tools/download_qwen_image.py --optimal")
            
            # Create optimized pipeline
            return self._load_optimized_model()
            
        except Exception as e:
            print(f"❌ Model detection failed: {e}")
            return self._load_basic_model()
    
    def _load_optimized_model(self) -> bool:
        """Load model with optimization"""
        try:
            print("🚀 Loading model with MMDiT optimizations...")
            
            # Create optimization configuration
            config = OptimizationConfig(
                architecture_type="MMDiT",
                optimal_steps=20,
                optimal_cfg_scale=3.5,
                enable_attention_slicing=False,  # Disabled for performance
                enable_vae_slicing=False,        # Disabled for performance
                enable_tf32=True,                # Enabled for RTX GPUs
                enable_cudnn_benchmark=True      # Enabled for consistent inputs
            )
            
            # Create optimizer
            self.optimizer = PipelineOptimizer(config)
            
            # Load optimized pipeline
            architecture = self.detector.detect_model_architecture(self.model_info)
            self.pipeline = self.optimizer.create_optimized_pipeline(
                self.model_info.path, 
                architecture
            )
            
            # Validate optimization
            validation = self.optimizer.validate_optimization(self.pipeline)
            print(f"✅ Optimization validation: {validation['overall_status']}")
            
            return True
            
        except Exception as e:
            print(f"❌ Optimized loading failed: {e}")
            print("🔄 Falling back to basic loading...")
            return self._load_basic_model()
    
    def _load_basic_model(self) -> bool:
        """Load model with basic configuration"""
        try:
            print("📦 Loading model with basic configuration...")
            
            # Try to find a model path
            model_path = None
            if self.model_info:
                model_path = self.model_info.path
            else:
                # Try common locations
                possible_paths = [
                    "./models/Qwen-Image",
                    "./models/Qwen-Image-Edit",
                    "Qwen/Qwen-Image"
                ]
                
                for path in possible_paths:
                    if os.path.exists(path) or "/" in path:  # Local path or HF repo
                        model_path = path
                        break
            
            if not model_path:
                print("❌ No model path found")
                return False
            
            print(f"📥 Loading from: {model_path}")
            
            # Load pipeline
            if torch.cuda.is_available():
                self.pipeline = AutoPipelineForText2Image.from_pretrained(
                    model_path,
                    torch_dtype=torch.float16,
                    trust_remote_code=True
                ).to("cuda")
                print("✅ Model loaded on GPU")
            else:
                self.pipeline = AutoPipelineForText2Image.from_pretrained(
                    model_path,
                    torch_dtype=torch.float32,
                    trust_remote_code=True
                )
                print("✅ Model loaded on CPU")
            
            return True
            
        except Exception as e:
            print(f"❌ Basic loading failed: {e}")
            return False
    
    def run_performance_benchmark(self):
        """Run performance benchmark with different settings"""
        if not self.pipeline:
            print("❌ No pipeline loaded")
            return
        
        print("\n🏃 Running Performance Benchmark")
        print("=" * 50)
        
        benchmark_configs = [
            {
                "name": "Ultra Fast",
                "steps": 15,
                "cfg_scale": 3.0,
                "resolution": (768, 768)
            },
            {
                "name": "Balanced",
                "steps": 20,
                "cfg_scale": 3.5,
                "resolution": (1024, 1024)
            },
            {
                "name": "High Quality",
                "steps": 30,
                "cfg_scale": 4.0,
                "resolution": (1024, 1024)
            }
        ]
        
        test_prompt = "A beautiful landscape with mountains and a lake, golden hour lighting"
        
        results = []
        
        for config in benchmark_configs:
            print(f"\n🧪 Testing {config['name']} settings...")
            print(f"   Steps: {config['steps']}, CFG: {config['cfg_scale']}, Resolution: {config['resolution']}")
            
            try:
                start_time = time.time()
                
                # Generate image
                generation_kwargs = {
                    "prompt": test_prompt,
                    "width": config["resolution"][0],
                    "height": config["resolution"][1],
                    "num_inference_steps": config["steps"],
                    "generator": torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu").manual_seed(42)
                }
                
                # Use appropriate CFG parameter based on architecture
                if self.model_info and "qwen" in self.model_info.name.lower():
                    generation_kwargs["true_cfg_scale"] = config["cfg_scale"]
                else:
                    generation_kwargs["guidance_scale"] = config["cfg_scale"]
                
                image = self.pipeline(**generation_kwargs).images[0]
                
                generation_time = time.time() - start_time
                time_per_step = generation_time / config["steps"]
                
                print(f"   ✅ Generated in {generation_time:.2f}s ({time_per_step:.2f}s/step)")
                
                results.append({
                    "config": config["name"],
                    "total_time": generation_time,
                    "time_per_step": time_per_step,
                    "steps": config["steps"],
                    "resolution": config["resolution"]
                })
                
                # Save benchmark image
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"benchmark_{config['name'].lower().replace(' ', '_')}_{timestamp}.png"
                filepath = os.path.join("generated_images", filename)
                os.makedirs("generated_images", exist_ok=True)
                image.save(filepath)
                print(f"   💾 Saved: {filepath}")
                
            except Exception as e:
                print(f"   ❌ Failed: {e}")
                results.append({
                    "config": config["name"],
                    "error": str(e)
                })
        
        # Display benchmark summary
        print(f"\n📊 Benchmark Summary")
        print("=" * 50)
        
        successful_results = [r for r in results if "error" not in r]
        if successful_results:
            fastest = min(successful_results, key=lambda x: x["time_per_step"])
            print(f"🏆 Fastest: {fastest['config']} ({fastest['time_per_step']:.2f}s/step)")
            
            for result in successful_results:
                efficiency = fastest["time_per_step"] / result["time_per_step"] * 100
                print(f"   {result['config']}: {result['time_per_step']:.2f}s/step ({efficiency:.0f}% efficiency)")
        
        failed_results = [r for r in results if "error" in r]
        if failed_results:
            print(f"\n❌ Failed configurations:")
            for result in failed_results:
                print(f"   {result['config']}: {result['error']}")
    
    def run_capability_showcase(self):
        """Showcase different generation capabilities"""
        if not self.pipeline:
            print("❌ No pipeline loaded")
            return
        
        print("\n🎨 Capability Showcase")
        print("=" * 50)
        
        for category, prompt_info in self.demo_prompts.items():
            print(f"\n🔹 {prompt_info['description']}")
            print(f"Prompt: {prompt_info['prompt'][:80]}...")
            
            try:
                start_time = time.time()
                
                # Generate with optimal settings
                generation_kwargs = {
                    "prompt": prompt_info["prompt"],
                    "width": 1024,
                    "height": 1024,
                    "num_inference_steps": 20,
                    "generator": torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu").manual_seed(42)
                }
                
                # Use appropriate CFG parameter
                if self.model_info and "qwen" in self.model_info.name.lower():
                    generation_kwargs["true_cfg_scale"] = 3.5
                else:
                    generation_kwargs["guidance_scale"] = 3.5
                
                image = self.pipeline(**generation_kwargs).images[0]
                
                generation_time = time.time() - start_time
                
                # Save image
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"showcase_{category}_{timestamp}.png"
                filepath = os.path.join("generated_images", filename)
                os.makedirs("generated_images", exist_ok=True)
                image.save(filepath)
                
                print(f"   ✅ Generated in {generation_time:.2f}s")
                print(f"   💾 Saved: {filepath}")
                
            except Exception as e:
                print(f"   ❌ Failed: {e}")
    
    def run_architecture_comparison(self):
        """Compare MMDiT vs traditional architectures"""
        print("\n⚡ Architecture Comparison")
        print("=" * 50)
        
        if not self.model_info:
            print("❌ No model information available")
            return
        
        architecture = "Unknown"
        if OPTIMIZATION_AVAILABLE:
            architecture = self.detector.detect_model_architecture(self.model_info)
        
        print(f"🏗️ Current Architecture: {architecture}")
        
        if architecture == "MMDiT":
            print("✅ MMDiT (Multimodal Diffusion Transformer) Benefits:")
            print("   • Modern transformer-based architecture")
            print("   • Optimized for text-to-image generation")
            print("   • Built-in attention optimizations")
            print("   • Better text rendering capabilities")
            print("   • Faster inference on modern GPUs")
            
            print("\n⚠️ MMDiT Considerations:")
            print("   • AttnProcessor2_0 not compatible (tensor unpacking issues)")
            print("   • torch.compile disabled due to tensor format differences")
            print("   • Uses true_cfg_scale instead of guidance_scale")
            
        elif architecture == "UNet":
            print("✅ UNet Architecture Benefits:")
            print("   • Well-established and stable")
            print("   • Compatible with AttnProcessor2_0 (Flash Attention)")
            print("   • torch.compile support available")
            print("   • Wide ecosystem compatibility")
            
            print("\n⚠️ UNet Limitations:")
            print("   • Older architecture design")
            print("   • May be less optimized for modern hardware")
            print("   • Potentially slower than MMDiT for text-to-image")
        
        else:
            print("❓ Unknown architecture - using generic optimizations")
        
        # Performance characteristics
        if OPTIMIZATION_AVAILABLE:
            perf_chars = self.detector.analyze_performance_characteristics(self.model_info)
            print(f"\n📊 Performance Characteristics:")
            print(f"   Expected Time: {perf_chars['expected_generation_time']}")
            print(f"   Memory Usage: {perf_chars['memory_usage']}")
            print(f"   Optimization Level: {perf_chars['optimization_level']}")
            
            if perf_chars['bottlenecks']:
                print(f"   Bottlenecks: {', '.join(perf_chars['bottlenecks'])}")
    
    def display_system_info(self):
        """Display system and model information"""
        print("💻 System Information")
        print("=" * 50)
        
        # GPU information
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name()
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"🔥 GPU: {gpu_name}")
            print(f"💾 VRAM: {gpu_memory:.1f}GB")
            
            # Memory usage
            if torch.cuda.is_initialized():
                allocated = torch.cuda.memory_allocated() / 1e9
                cached = torch.cuda.memory_reserved() / 1e9
                print(f"📊 VRAM Used: {allocated:.1f}GB allocated, {cached:.1f}GB cached")
        else:
            print("🔥 GPU: Not available (using CPU)")
        
        # Model information
        if self.model_info:
            print(f"\n📦 Model Information:")
            print(f"   Name: {self.model_info.name}")
            print(f"   Type: {self.model_info.model_type}")
            print(f"   Size: {self.model_info.size_gb:.1f}GB")
            print(f"   Status: {self.model_info.download_status}")
            print(f"   Optimal: {'Yes' if self.model_info.is_optimal else 'No'}")
        
        # Optimization status
        if self.optimizer:
            print(f"\n⚡ Optimization Status:")
            validation = self.optimizer.validate_optimization(self.pipeline)
            for key, value in validation.items():
                print(f"   {key.replace('_', ' ').title()}: {value}")


def main():
    """Main demo function"""
    print("🎨 Optimized Text-to-Image Generation Demo")
    print("=" * 60)
    print("Showcasing MMDiT performance improvements with modern Qwen architecture")
    print("")
    
    # Initialize demo
    demo = OptimizedTextToImageDemo()
    
    # Load model
    if not demo.detect_and_load_model():
        print("❌ Failed to load model")
        sys.exit(1)
    
    # Display system information
    demo.display_system_info()
    
    # Run demonstrations
    try:
        # Architecture comparison
        demo.run_architecture_comparison()
        
        # Performance benchmark
        demo.run_performance_benchmark()
        
        # Capability showcase
        demo.run_capability_showcase()
        
        print("\n🎉 Demo completed successfully!")
        print("=" * 60)
        print("📁 Generated images saved to: ./generated_images/")
        print("💡 Try different prompts with the optimized pipeline!")
        print("🚀 Next: python examples/multimodal_integration_demo.py")
        
    except KeyboardInterrupt:
        print("\n⏹️ Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()