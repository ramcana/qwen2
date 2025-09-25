#!/usr/bin/env python3
"""
Architecture Optimization Examples
Demonstrates various optimization techniques for modern Qwen architectures
"""

import os
import sys
import time
import torch
from datetime import datetime
from typing import Dict, List, Optional

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from pipeline_optimizer import PipelineOptimizer, OptimizationConfig
    from model_detection_service import ModelDetectionService
    OPTIMIZATION_AVAILABLE = True
except ImportError:
    OPTIMIZATION_AVAILABLE = False
    print("❌ Optimization components not available")
    sys.exit(1)

try:
    from diffusers import AutoPipelineForText2Image
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False
    print("❌ diffusers not installed. Run: pip install diffusers")
    sys.exit(1)


class ArchitectureOptimizationExamples:
    """Examples demonstrating architecture-specific optimizations"""
    
    def __init__(self):
        self.detector = ModelDetectionService()
        self.current_model = None
        self.architecture = None
        
        # Detect current model
        self.current_model = self.detector.detect_current_model()
        if self.current_model:
            self.architecture = self.detector.detect_model_architecture(self.current_model)
            print(f"📦 Detected model: {self.current_model.name}")
            print(f"🏗️ Architecture: {self.architecture}")
        else:
            print("❌ No model detected. Please download a model first.")
            sys.exit(1)
    
    def example_1_basic_vs_optimized(self):
        """Compare basic loading vs optimized loading"""
        print("\n🔄 Example 1: Basic vs Optimized Pipeline Loading")
        print("=" * 60)
        
        test_prompt = "A serene mountain landscape at sunset"
        
        # 1. Basic loading
        print("📦 Loading basic pipeline...")
        try:
            basic_start = time.time()
            basic_pipeline = AutoPipelineForText2Image.from_pretrained(
                self.current_model.path,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                trust_remote_code=True
            )
            if torch.cuda.is_available():
                basic_pipeline = basic_pipeline.to("cuda")
            basic_load_time = time.time() - basic_start
            
            print(f"✅ Basic pipeline loaded in {basic_load_time:.2f}s")
            
            # Generate with basic pipeline
            basic_gen_start = time.time()
            basic_kwargs = {
                "prompt": test_prompt,
                "width": 1024,
                "height": 1024,
                "num_inference_steps": 20,
                "generator": torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu").manual_seed(42)
            }
            
            if self.architecture == "MMDiT":
                basic_kwargs["true_cfg_scale"] = 3.5
            else:
                basic_kwargs["guidance_scale"] = 3.5
            
            basic_image = basic_pipeline(**basic_kwargs).images[0]
            basic_gen_time = time.time() - basic_gen_start
            
            # Save basic image
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            basic_path = os.path.join("generated_images", f"basic_example_{timestamp}.png")
            os.makedirs("generated_images", exist_ok=True)
            basic_image.save(basic_path)
            
            print(f"✅ Basic generation: {basic_gen_time:.2f}s ({basic_gen_time/20:.2f}s/step)")
            print(f"💾 Saved: {basic_path}")
            
        except Exception as e:
            print(f"❌ Basic pipeline failed: {e}")
            return
        
        # 2. Optimized loading
        print("\n🚀 Loading optimized pipeline...")
        try:
            optimized_start = time.time()
            
            # Create optimization configuration
            config = OptimizationConfig(
                architecture_type=self.architecture,
                enable_attention_slicing=False,  # Disabled for performance
                enable_vae_slicing=False,        # Disabled for performance
                enable_tf32=True,                # Enabled for RTX GPUs
                enable_cudnn_benchmark=True      # Enabled for consistent inputs
            )
            
            optimizer = PipelineOptimizer(config)
            optimized_pipeline = optimizer.create_optimized_pipeline(
                self.current_model.path, 
                self.architecture
            )
            optimized_load_time = time.time() - optimized_start
            
            print(f"✅ Optimized pipeline loaded in {optimized_load_time:.2f}s")
            
            # Validate optimizations
            validation = optimizer.validate_optimization(optimized_pipeline)
            print(f"🔧 Optimization status: {validation['overall_status']}")
            print(f"⚡ GPU optimizations: {validation['gpu_optimizations']}")
            
            # Generate with optimized pipeline
            optimized_gen_start = time.time()
            optimized_image = optimized_pipeline(**basic_kwargs).images[0]
            optimized_gen_time = time.time() - optimized_gen_start
            
            # Save optimized image
            optimized_path = os.path.join("generated_images", f"optimized_example_{timestamp}.png")
            optimized_image.save(optimized_path)
            
            print(f"✅ Optimized generation: {optimized_gen_time:.2f}s ({optimized_gen_time/20:.2f}s/step)")
            print(f"💾 Saved: {optimized_path}")
            
            # Compare results
            improvement = basic_gen_time / optimized_gen_time if optimized_gen_time > 0 else 1
            print(f"\n📊 Performance Comparison:")
            print(f"   Basic: {basic_gen_time:.2f}s")
            print(f"   Optimized: {optimized_gen_time:.2f}s")
            print(f"   Improvement: {improvement:.1f}x faster")
            
        except Exception as e:
            print(f"❌ Optimized pipeline failed: {e}")
    
    def example_2_memory_vs_performance_configs(self):
        """Compare memory-efficient vs performance-optimized configurations"""
        print("\n⚖️ Example 2: Memory-Efficient vs Performance-Optimized")
        print("=" * 60)
        
        test_prompt = "A futuristic city with flying cars and neon lights"
        configs = {
            "memory_efficient": OptimizationConfig(
                architecture_type=self.architecture,
                enable_attention_slicing=True,   # Enabled for memory efficiency
                enable_vae_slicing=True,         # Enabled for memory efficiency
                enable_tf32=True,
                enable_cudnn_benchmark=True,
                optimal_steps=25,
                optimal_cfg_scale=3.5
            ),
            "performance_optimized": OptimizationConfig(
                architecture_type=self.architecture,
                enable_attention_slicing=False,  # Disabled for performance
                enable_vae_slicing=False,        # Disabled for performance
                enable_tf32=True,
                enable_cudnn_benchmark=True,
                optimal_steps=20,
                optimal_cfg_scale=3.5
            )
        }
        
        results = {}
        
        for config_name, config in configs.items():
            print(f"\n🔧 Testing {config_name} configuration...")
            
            try:
                # Create pipeline
                optimizer = PipelineOptimizer(config)
                pipeline = optimizer.create_optimized_pipeline(
                    self.current_model.path, 
                    self.architecture
                )
                
                # Monitor memory
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                    memory_before = torch.cuda.memory_allocated() / 1e9
                
                # Generate image
                start_time = time.time()
                
                generation_kwargs = {
                    "prompt": test_prompt,
                    "width": 1024,
                    "height": 1024,
                    "num_inference_steps": config.optimal_steps,
                    "generator": torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu").manual_seed(42)
                }
                
                if self.architecture == "MMDiT":
                    generation_kwargs["true_cfg_scale"] = config.optimal_cfg_scale
                else:
                    generation_kwargs["guidance_scale"] = config.optimal_cfg_scale
                
                image = pipeline(**generation_kwargs).images[0]
                generation_time = time.time() - start_time
                
                # Record memory usage
                memory_peak = 0
                if torch.cuda.is_available():
                    memory_peak = torch.cuda.max_memory_allocated() / 1e9
                    torch.cuda.empty_cache()
                
                # Save image
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{config_name}_example_{timestamp}.png"
                filepath = os.path.join("generated_images", filename)
                image.save(filepath)
                
                results[config_name] = {
                    "generation_time": generation_time,
                    "time_per_step": generation_time / config.optimal_steps,
                    "memory_used": memory_peak - memory_before if torch.cuda.is_available() else 0,
                    "steps": config.optimal_steps,
                    "filepath": filepath
                }
                
                print(f"✅ {config_name}:")
                print(f"   Generation time: {generation_time:.2f}s")
                print(f"   Time per step: {generation_time / config.optimal_steps:.2f}s")
                print(f"   Memory used: {results[config_name]['memory_used']:.2f}GB")
                print(f"   Saved: {filepath}")
                
            except Exception as e:
                print(f"❌ {config_name} failed: {e}")
                results[config_name] = {"error": str(e)}
        
        # Compare results
        if all("error" not in result for result in results.values()):
            print(f"\n📊 Configuration Comparison:")
            memory_config = results["memory_efficient"]
            perf_config = results["performance_optimized"]
            
            speed_diff = memory_config["generation_time"] / perf_config["generation_time"]
            memory_diff = perf_config["memory_used"] / memory_config["memory_used"] if memory_config["memory_used"] > 0 else 1
            
            print(f"   Speed: Performance config is {speed_diff:.1f}x faster")
            print(f"   Memory: Memory config uses {memory_diff:.1f}x less memory")
            print(f"   Recommendation: Use performance config if you have 16GB+ VRAM")
    
    def example_3_architecture_specific_features(self):
        """Demonstrate architecture-specific optimization features"""
        print(f"\n🏗️ Example 3: {self.architecture} Architecture-Specific Features")
        print("=" * 60)
        
        if self.architecture == "MMDiT":
            self._demonstrate_mmdit_features()
        elif self.architecture == "UNet":
            self._demonstrate_unet_features()
        else:
            print(f"⚠️ Unknown architecture: {self.architecture}")
    
    def _demonstrate_mmdit_features(self):
        """Demonstrate MMDiT-specific features"""
        print("🎯 MMDiT (Multimodal Diffusion Transformer) Features:")
        print()
        
        # 1. Text rendering capabilities
        print("📝 1. Advanced Text Rendering")
        text_prompts = [
            'A coffee shop with a sign reading "AI Café ☕" and prices "$3.50 Latte, $4.00 Cappuccino"',
            'A Japanese restaurant with signs "寿司 Sushi" and "Welcome いらっしゃいませ"',
            'A math classroom with equations "E=mc²" and "π≈3.14159" on the blackboard'
        ]
        
        config = OptimizationConfig(
            architecture_type="MMDiT",
            enable_attention_slicing=False,
            enable_vae_slicing=False,
            enable_tf32=True,
            enable_cudnn_benchmark=True
        )
        
        optimizer = PipelineOptimizer(config)
        pipeline = optimizer.create_optimized_pipeline(self.current_model.path, "MMDiT")
        
        for i, prompt in enumerate(text_prompts):
            try:
                print(f"   Generating text example {i+1}...")
                
                image = pipeline(
                    prompt=prompt,
                    width=1024,
                    height=1024,
                    num_inference_steps=20,
                    true_cfg_scale=3.5,  # MMDiT uses true_cfg_scale
                    generator=torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu").manual_seed(42)
                ).images[0]
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"mmdit_text_example_{i+1}_{timestamp}.png"
                filepath = os.path.join("generated_images", filename)
                image.save(filepath)
                
                print(f"   ✅ Saved: {filename}")
                
            except Exception as e:
                print(f"   ❌ Failed: {e}")
        
        # 2. MMDiT-specific optimizations
        print(f"\n⚡ 2. MMDiT-Specific Optimizations:")
        print("   ✅ Uses true_cfg_scale parameter (not guidance_scale)")
        print("   ✅ Custom attention processor optimized for transformer architecture")
        print("   ✅ Built-in text rendering optimizations")
        print("   ⚠️ AttnProcessor2_0 disabled (causes tensor unpacking errors)")
        print("   ⚠️ torch.compile disabled (tensor format incompatibility)")
        
        # 3. Performance characteristics
        print(f"\n📊 3. Performance Characteristics:")
        perf_chars = self.detector.analyze_performance_characteristics(self.current_model)
        print(f"   Expected generation time: {perf_chars['expected_generation_time']}")
        print(f"   Memory usage: {perf_chars['memory_usage']}")
        print(f"   Optimization level: {perf_chars['optimization_level']}")
        
        if perf_chars['bottlenecks']:
            print(f"   Bottlenecks: {', '.join(perf_chars['bottlenecks'])}")
    
    def _demonstrate_unet_features(self):
        """Demonstrate UNet-specific features"""
        print("🎯 UNet Architecture Features:")
        print()
        
        # 1. Flash Attention support
        print("⚡ 1. Flash Attention Support")
        try:
            from diffusers.models.attention_processor import AttnProcessor2_0
            
            config = OptimizationConfig(
                architecture_type="UNet",
                enable_flash_attention=True,
                enable_tf32=True,
                enable_cudnn_benchmark=True
            )
            
            optimizer = PipelineOptimizer(config)
            pipeline = optimizer.create_optimized_pipeline(self.current_model.path, "UNet")
            
            # Test with Flash Attention
            if hasattr(pipeline, 'unet'):
                pipeline.unet.set_attn_processor(AttnProcessor2_0())
                print("   ✅ Flash Attention (AttnProcessor2_0) enabled")
            
            # Generate test image
            image = pipeline(
                prompt="A beautiful landscape with mountains and lakes",
                width=1024,
                height=1024,
                num_inference_steps=20,
                guidance_scale=7.5,  # UNet uses guidance_scale
                generator=torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu").manual_seed(42)
            ).images[0]
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"unet_flash_attention_{timestamp}.png"
            filepath = os.path.join("generated_images", filename)
            image.save(filepath)
            
            print(f"   ✅ Generated with Flash Attention: {filename}")
            
        except ImportError:
            print("   ⚠️ AttnProcessor2_0 not available")
        except Exception as e:
            print(f"   ❌ Flash Attention test failed: {e}")
        
        # 2. torch.compile support
        print(f"\n🔧 2. torch.compile Support")
        try:
            config = OptimizationConfig(
                architecture_type="UNet",
                use_torch_compile=True
            )
            
            optimizer = PipelineOptimizer(config)
            success = optimizer.apply_torch_compile_optimization(pipeline)
            
            if success:
                print("   ✅ torch.compile optimization applied")
            else:
                print("   ⚠️ torch.compile optimization not applied")
                
        except Exception as e:
            print(f"   ❌ torch.compile test failed: {e}")
        
        # 3. UNet-specific optimizations
        print(f"\n⚡ 3. UNet-Specific Optimizations:")
        print("   ✅ Uses guidance_scale parameter")
        print("   ✅ Compatible with AttnProcessor2_0 (Flash Attention)")
        print("   ✅ torch.compile support available")
        print("   ✅ Wide ecosystem compatibility")
        print("   ⚠️ May be less optimized than MMDiT for text-to-image tasks")
    
    def example_4_performance_tuning_guide(self):
        """Demonstrate performance tuning for different hardware configurations"""
        print("\n🎛️ Example 4: Performance Tuning Guide")
        print("=" * 60)
        
        # Detect GPU memory
        gpu_memory = 0
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            gpu_name = torch.cuda.get_device_name()
            print(f"🔥 Detected GPU: {gpu_name} ({gpu_memory:.1f}GB VRAM)")
        else:
            print("⚠️ No GPU detected, using CPU")
        
        # Get performance recommendations
        optimizer = PipelineOptimizer()
        recommendations = optimizer.get_performance_recommendations(gpu_memory)
        
        print(f"\n💡 Performance Recommendations:")
        print(f"   Memory Strategy: {recommendations['memory_strategy']}")
        print(f"   Expected Performance: {recommendations['expected_performance']}")
        
        if recommendations['warnings']:
            print(f"   Warnings: {', '.join(recommendations['warnings'])}")
        
        # Create recommended configuration
        optimal_settings = recommendations['optimal_settings']
        print(f"\n⚙️ Recommended Settings:")
        for setting, value in optimal_settings.items():
            print(f"   {setting}: {value}")
        
        # Test recommended configuration
        print(f"\n🧪 Testing Recommended Configuration:")
        try:
            config = OptimizationConfig(
                architecture_type=self.architecture,
                enable_attention_slicing=optimal_settings.get('enable_attention_slicing', False),
                enable_vae_slicing=optimal_settings.get('enable_vae_slicing', False),
                enable_cpu_offload=optimal_settings.get('enable_cpu_offload', False),
                torch_dtype=getattr(torch, optimal_settings.get('torch_dtype', 'float16')),
                optimal_width=int(optimal_settings.get('resolution', '1024x1024').split('x')[0]),
                optimal_height=int(optimal_settings.get('resolution', '1024x1024').split('x')[1])
            )
            
            optimizer = PipelineOptimizer(config)
            pipeline = optimizer.create_optimized_pipeline(self.current_model.path, self.architecture)
            
            # Test generation
            start_time = time.time()
            
            generation_kwargs = {
                "prompt": "A test image for performance tuning",
                "width": config.optimal_width,
                "height": config.optimal_height,
                "num_inference_steps": 20,
                "generator": torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu").manual_seed(42)
            }
            
            if self.architecture == "MMDiT":
                generation_kwargs["true_cfg_scale"] = 3.5
            else:
                generation_kwargs["guidance_scale"] = 3.5
            
            image = pipeline(**generation_kwargs).images[0]
            generation_time = time.time() - start_time
            
            # Save result
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"performance_tuned_{timestamp}.png"
            filepath = os.path.join("generated_images", filename)
            image.save(filepath)
            
            print(f"   ✅ Generation time: {generation_time:.2f}s ({generation_time/20:.2f}s/step)")
            print(f"   💾 Saved: {filename}")
            
            # Performance assessment
            time_per_step = generation_time / 20
            if time_per_step <= 2.0:
                print("   🏆 Excellent performance!")
            elif time_per_step <= 5.0:
                print("   ✅ Good performance")
            elif time_per_step <= 10.0:
                print("   ⚠️ Moderate performance")
            else:
                print("   ❌ Poor performance - consider hardware upgrade")
            
        except Exception as e:
            print(f"   ❌ Performance tuning test failed: {e}")
    
    def run_all_examples(self):
        """Run all optimization examples"""
        print("🎨 Architecture Optimization Examples")
        print("=" * 70)
        print(f"Model: {self.current_model.name}")
        print(f"Architecture: {self.architecture}")
        print()
        
        try:
            self.example_1_basic_vs_optimized()
            self.example_2_memory_vs_performance_configs()
            self.example_3_architecture_specific_features()
            self.example_4_performance_tuning_guide()
            
            print("\n🎉 All optimization examples completed!")
            print("=" * 70)
            print("📁 Generated images saved to: ./generated_images/")
            print("💡 Key takeaways:")
            print("   • Optimized pipelines provide significant performance improvements")
            print("   • Memory vs performance trade-offs depend on your hardware")
            print("   • Architecture-specific features enhance capabilities")
            print("   • Performance tuning should match your GPU specifications")
            
        except KeyboardInterrupt:
            print("\n⏹️ Examples interrupted by user")
        except Exception as e:
            print(f"\n❌ Examples failed: {e}")
            import traceback
            traceback.print_exc()


def main():
    """Main function"""
    if not OPTIMIZATION_AVAILABLE:
        print("❌ Optimization components not available")
        print("💡 Make sure all required modules are installed")
        sys.exit(1)
    
    examples = ArchitectureOptimizationExamples()
    
    print("Select examples to run:")
    print("1. Basic vs Optimized comparison")
    print("2. Memory vs Performance configurations")
    print("3. Architecture-specific features")
    print("4. Performance tuning guide")
    print("5. All examples")
    
    try:
        choice = input("\nEnter choice (1-5, default=5): ").strip() or "5"
        
        if choice == "1":
            examples.example_1_basic_vs_optimized()
        elif choice == "2":
            examples.example_2_memory_vs_performance_configs()
        elif choice == "3":
            examples.example_3_architecture_specific_features()
        elif choice == "4":
            examples.example_4_performance_tuning_guide()
        elif choice == "5":
            examples.run_all_examples()
        else:
            print("Invalid choice, running all examples...")
            examples.run_all_examples()
            
    except KeyboardInterrupt:
        print("\n⏹️ Interrupted by user")
    except Exception as e:
        print(f"\n❌ Failed: {e}")


if __name__ == "__main__":
    main()