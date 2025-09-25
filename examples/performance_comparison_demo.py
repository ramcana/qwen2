#!/usr/bin/env python3
"""
Performance Comparison Demo
Benchmarks and compares different Qwen architectures and optimization levels
"""

import os
import sys
import time
import json
import torch
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import statistics

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from model_detection_service import ModelDetectionService
    from pipeline_optimizer import PipelineOptimizer, OptimizationConfig
    from utils.performance_monitor import PerformanceMonitor
    OPTIMIZATION_AVAILABLE = True
except ImportError:
    OPTIMIZATION_AVAILABLE = False
    print("⚠️ Optimization components not available")

try:
    from diffusers import AutoPipelineForText2Image, DiffusionPipeline
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False
    print("❌ diffusers not installed. Run: pip install diffusers")
    sys.exit(1)


class PerformanceComparisonDemo:
    """Demo class for performance comparison and benchmarking"""
    
    def __init__(self):
        self.detector = None
        self.performance_monitor = None
        self.benchmark_results = []
        
        # Initialize components if available
        if OPTIMIZATION_AVAILABLE:
            self.detector = ModelDetectionService()
            self.performance_monitor = PerformanceMonitor()
        
        # Benchmark configurations
        self.benchmark_configs = {
            "ultra_fast": {
                "name": "Ultra Fast",
                "steps": 10,
                "cfg_scale": 2.5,
                "resolution": (512, 512),
                "description": "Minimum quality, maximum speed"
            },
            "fast": {
                "name": "Fast",
                "steps": 15,
                "cfg_scale": 3.0,
                "resolution": (768, 768),
                "description": "Good quality, fast generation"
            },
            "balanced": {
                "name": "Balanced",
                "steps": 20,
                "cfg_scale": 3.5,
                "resolution": (1024, 1024),
                "description": "Balanced quality and speed"
            },
            "quality": {
                "name": "High Quality",
                "steps": 30,
                "cfg_scale": 4.0,
                "resolution": (1024, 1024),
                "description": "High quality, slower generation"
            },
            "ultra_quality": {
                "name": "Ultra Quality",
                "steps": 50,
                "cfg_scale": 4.5,
                "resolution": (1280, 1280),
                "description": "Maximum quality, slowest generation"
            }
        }
        
        # Test prompts for benchmarking
        self.test_prompts = [
            "A serene mountain landscape at sunset",
            "A modern city skyline with glass buildings",
            "A portrait of a person reading a book",
            "A colorful abstract geometric pattern",
            "A peaceful garden with blooming flowers"
        ]
    
    def detect_available_models(self) -> Dict[str, any]:
        """Detect available models and their characteristics"""
        model_info = {
            "models": [],
            "recommended": None,
            "optimization_needed": False
        }
        
        if not OPTIMIZATION_AVAILABLE:
            print("⚠️ Model detection not available")
            return model_info
        
        try:
            # Detect current model
            current_model = self.detector.detect_current_model()
            if current_model:
                architecture = self.detector.detect_model_architecture(current_model)
                perf_chars = self.detector.analyze_performance_characteristics(current_model)
                
                model_info["models"].append({
                    "name": current_model.name,
                    "path": current_model.path,
                    "type": current_model.model_type,
                    "size_gb": current_model.size_gb,
                    "architecture": architecture,
                    "is_optimal": current_model.is_optimal,
                    "expected_performance": perf_chars["expected_generation_time"],
                    "bottlenecks": perf_chars["bottlenecks"]
                })
                
                print(f"📦 Found model: {current_model.name}")
                print(f"🏗️ Architecture: {architecture}")
                print(f"⚡ Expected performance: {perf_chars['expected_generation_time']}")
            
            # Check optimization needs
            model_info["optimization_needed"] = self.detector.is_optimization_needed()
            if model_info["optimization_needed"]:
                model_info["recommended"] = self.detector.get_recommended_model()
                print(f"💡 Optimization recommended: {model_info['recommended']}")
            
        except Exception as e:
            print(f"❌ Model detection failed: {e}")
        
        return model_info
    
    def create_pipeline_variants(self, model_path: str, architecture: str) -> Dict[str, any]:
        """Create different pipeline variants for comparison"""
        variants = {}
        
        if not OPTIMIZATION_AVAILABLE:
            print("⚠️ Creating basic pipeline only")
            try:
                pipeline = AutoPipelineForText2Image.from_pretrained(
                    model_path,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    trust_remote_code=True
                )
                if torch.cuda.is_available():
                    pipeline = pipeline.to("cuda")
                variants["basic"] = {
                    "pipeline": pipeline,
                    "name": "Basic Configuration",
                    "description": "Default settings without optimization"
                }
            except Exception as e:
                print(f"❌ Failed to create basic pipeline: {e}")
            return variants
        
        try:
            # 1. Unoptimized baseline
            print("📦 Creating unoptimized baseline...")
            baseline_config = OptimizationConfig(
                architecture_type=architecture,
                enable_attention_slicing=True,   # Memory-saving enabled
                enable_vae_slicing=True,         # Memory-saving enabled
                enable_tf32=False,               # Performance disabled
                enable_cudnn_benchmark=False     # Performance disabled
            )
            baseline_optimizer = PipelineOptimizer(baseline_config)
            variants["unoptimized"] = {
                "pipeline": baseline_optimizer.create_optimized_pipeline(model_path, architecture),
                "name": "Unoptimized Baseline",
                "description": "Memory-saving enabled, performance features disabled"
            }
            
            # 2. Memory-efficient configuration
            print("📦 Creating memory-efficient configuration...")
            memory_config = OptimizationConfig(
                architecture_type=architecture,
                enable_attention_slicing=True,
                enable_vae_slicing=True,
                enable_tf32=True,
                enable_cudnn_benchmark=True,
                optimal_steps=25,
                optimal_cfg_scale=3.5
            )
            memory_optimizer = PipelineOptimizer(memory_config)
            variants["memory_efficient"] = {
                "pipeline": memory_optimizer.create_optimized_pipeline(model_path, architecture),
                "name": "Memory Efficient",
                "description": "Balanced memory usage and performance"
            }
            
            # 3. Performance-optimized configuration
            print("📦 Creating performance-optimized configuration...")
            performance_config = OptimizationConfig(
                architecture_type=architecture,
                enable_attention_slicing=False,  # Disabled for performance
                enable_vae_slicing=False,        # Disabled for performance
                enable_tf32=True,                # Enabled for performance
                enable_cudnn_benchmark=True,     # Enabled for performance
                optimal_steps=20,
                optimal_cfg_scale=3.5
            )
            performance_optimizer = PipelineOptimizer(performance_config)
            variants["performance_optimized"] = {
                "pipeline": performance_optimizer.create_optimized_pipeline(model_path, architecture),
                "name": "Performance Optimized",
                "description": "Maximum performance, memory-saving disabled"
            }
            
            # 4. Ultra-performance configuration (if high VRAM available)
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
                if gpu_memory >= 16:  # 16GB+ VRAM
                    print("📦 Creating ultra-performance configuration...")
                    ultra_config = OptimizationConfig(
                        architecture_type=architecture,
                        enable_attention_slicing=False,
                        enable_vae_slicing=False,
                        enable_tf32=True,
                        enable_cudnn_benchmark=True,
                        optimal_steps=15,
                        optimal_cfg_scale=3.0
                    )
                    ultra_optimizer = PipelineOptimizer(ultra_config)
                    variants["ultra_performance"] = {
                        "pipeline": ultra_optimizer.create_optimized_pipeline(model_path, architecture),
                        "name": "Ultra Performance",
                        "description": "Maximum speed for high-VRAM GPUs"
                    }
            
        except Exception as e:
            print(f"❌ Failed to create pipeline variants: {e}")
        
        return variants
    
    def benchmark_pipeline_variant(self, variant_name: str, variant_info: Dict, config_name: str, config: Dict) -> Dict[str, any]:
        """Benchmark a specific pipeline variant with a configuration"""
        pipeline = variant_info["pipeline"]
        
        if not pipeline:
            return {"error": "Pipeline not available"}
        
        try:
            # Use a consistent test prompt
            test_prompt = "A beautiful landscape with mountains and a lake, golden hour lighting"
            
            # Prepare generation arguments
            generation_kwargs = {
                "prompt": test_prompt,
                "width": config["resolution"][0],
                "height": config["resolution"][1],
                "num_inference_steps": config["steps"],
                "generator": torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu").manual_seed(42)
            }
            
            # Use appropriate CFG parameter based on architecture
            if "qwen" in str(pipeline.__class__).lower() or hasattr(pipeline, 'transformer'):
                generation_kwargs["true_cfg_scale"] = config["cfg_scale"]
            else:
                generation_kwargs["guidance_scale"] = config["cfg_scale"]
            
            # Warm up (first generation is often slower)
            print(f"🔥 Warming up {variant_name} with {config_name}...")
            try:
                _ = pipeline(**generation_kwargs)
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
            except Exception as warmup_error:
                print(f"⚠️ Warmup failed: {warmup_error}")
            
            # Actual benchmark
            print(f"⏱️ Benchmarking {variant_name} with {config_name}...")
            
            start_time = time.time()
            
            # Monitor GPU memory if available
            gpu_memory_before = 0
            gpu_memory_peak = 0
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                gpu_memory_before = torch.cuda.memory_allocated() / 1e9
            
            # Generate image
            result = pipeline(**generation_kwargs)
            image = result.images[0]
            
            # Record timing and memory
            generation_time = time.time() - start_time
            time_per_step = generation_time / config["steps"]
            
            if torch.cuda.is_available():
                gpu_memory_peak = torch.cuda.max_memory_allocated() / 1e9
                torch.cuda.empty_cache()
            
            # Save benchmark image
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"benchmark_{variant_name}_{config_name}_{timestamp}.png"
            filepath = os.path.join("generated_images", "benchmarks", filename)
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            image.save(filepath)
            
            return {
                "variant": variant_name,
                "config": config_name,
                "total_time": generation_time,
                "time_per_step": time_per_step,
                "steps": config["steps"],
                "resolution": config["resolution"],
                "cfg_scale": config["cfg_scale"],
                "gpu_memory_before": gpu_memory_before,
                "gpu_memory_peak": gpu_memory_peak,
                "memory_used": gpu_memory_peak - gpu_memory_before,
                "image_path": filepath,
                "success": True
            }
            
        except Exception as e:
            print(f"❌ Benchmark failed for {variant_name} with {config_name}: {e}")
            return {
                "variant": variant_name,
                "config": config_name,
                "error": str(e),
                "success": False
            }
    
    def run_comprehensive_benchmark(self, model_path: str, architecture: str):
        """Run comprehensive benchmark across all variants and configurations"""
        print("\n🏃 Running Comprehensive Performance Benchmark")
        print("=" * 70)
        
        # Create pipeline variants
        variants = self.create_pipeline_variants(model_path, architecture)
        
        if not variants:
            print("❌ No pipeline variants available")
            return
        
        print(f"📊 Testing {len(variants)} pipeline variants with {len(self.benchmark_configs)} configurations")
        print(f"📈 Total benchmarks: {len(variants) * len(self.benchmark_configs)}")
        
        # Run benchmarks
        all_results = []
        
        for variant_name, variant_info in variants.items():
            print(f"\n🔧 Testing variant: {variant_info['name']}")
            print(f"   {variant_info['description']}")
            
            for config_name, config in self.benchmark_configs.items():
                result = self.benchmark_pipeline_variant(variant_name, variant_info, config_name, config)
                all_results.append(result)
                
                if result.get("success"):
                    print(f"   ✅ {config['name']}: {result['time_per_step']:.2f}s/step ({result['total_time']:.2f}s total)")
                else:
                    print(f"   ❌ {config['name']}: {result.get('error', 'Unknown error')}")
        
        # Store results
        self.benchmark_results = all_results
        
        # Analyze and display results
        self.analyze_benchmark_results()
    
    def analyze_benchmark_results(self):
        """Analyze and display benchmark results"""
        if not self.benchmark_results:
            print("❌ No benchmark results to analyze")
            return
        
        print("\n📊 Benchmark Analysis")
        print("=" * 70)
        
        # Filter successful results
        successful_results = [r for r in self.benchmark_results if r.get("success")]
        
        if not successful_results:
            print("❌ No successful benchmark results")
            return
        
        # Find best performers
        fastest_overall = min(successful_results, key=lambda x: x["time_per_step"])
        most_efficient = min(successful_results, key=lambda x: x.get("memory_used", float('inf')))
        
        print(f"🏆 Fastest Overall:")
        print(f"   {fastest_overall['variant']} + {fastest_overall['config']}")
        print(f"   {fastest_overall['time_per_step']:.2f}s/step ({fastest_overall['total_time']:.2f}s total)")
        
        print(f"\n💾 Most Memory Efficient:")
        print(f"   {most_efficient['variant']} + {most_efficient['config']}")
        print(f"   {most_efficient.get('memory_used', 0):.2f}GB memory used")
        
        # Performance by variant
        print(f"\n📈 Performance by Pipeline Variant:")
        variants = set(r["variant"] for r in successful_results)
        
        for variant in variants:
            variant_results = [r for r in successful_results if r["variant"] == variant]
            if variant_results:
                avg_time_per_step = statistics.mean(r["time_per_step"] for r in variant_results)
                best_time = min(r["time_per_step"] for r in variant_results)
                worst_time = max(r["time_per_step"] for r in variant_results)
                
                print(f"   {variant}:")
                print(f"     Average: {avg_time_per_step:.2f}s/step")
                print(f"     Best: {best_time:.2f}s/step")
                print(f"     Worst: {worst_time:.2f}s/step")
        
        # Performance by configuration
        print(f"\n⚙️ Performance by Configuration:")
        configs = set(r["config"] for r in successful_results)
        
        for config in configs:
            config_results = [r for r in successful_results if r["config"] == config]
            if config_results:
                avg_time_per_step = statistics.mean(r["time_per_step"] for r in config_results)
                best_variant = min(config_results, key=lambda x: x["time_per_step"])
                
                print(f"   {config}:")
                print(f"     Average: {avg_time_per_step:.2f}s/step")
                print(f"     Best variant: {best_variant['variant']} ({best_variant['time_per_step']:.2f}s/step)")
        
        # Speed improvement analysis
        if len(successful_results) > 1:
            slowest = max(successful_results, key=lambda x: x["time_per_step"])
            improvement_factor = slowest["time_per_step"] / fastest_overall["time_per_step"]
            
            print(f"\n🚀 Speed Improvement Analysis:")
            print(f"   Slowest: {slowest['variant']} + {slowest['config']} ({slowest['time_per_step']:.2f}s/step)")
            print(f"   Fastest: {fastest_overall['variant']} + {fastest_overall['config']} ({fastest_overall['time_per_step']:.2f}s/step)")
            print(f"   Improvement: {improvement_factor:.1f}x faster")
            
            # Check if we meet performance targets
            target_time_per_step = 5.0  # 5 seconds per step target
            if fastest_overall["time_per_step"] <= target_time_per_step:
                print(f"   ✅ Performance target met: {fastest_overall['time_per_step']:.2f}s ≤ {target_time_per_step}s")
            else:
                print(f"   ⚠️ Performance target not met: {fastest_overall['time_per_step']:.2f}s > {target_time_per_step}s")
    
    def export_benchmark_results(self):
        """Export benchmark results to JSON file"""
        if not self.benchmark_results:
            print("❌ No results to export")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"performance_benchmark_results_{timestamp}.json"
        filepath = os.path.join("generated_images", "benchmarks", filename)
        
        # Prepare export data
        export_data = {
            "timestamp": timestamp,
            "system_info": {
                "cuda_available": torch.cuda.is_available(),
                "gpu_name": torch.cuda.get_device_name() if torch.cuda.is_available() else "N/A",
                "gpu_memory_gb": torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0
            },
            "benchmark_results": self.benchmark_results,
            "summary": {
                "total_benchmarks": len(self.benchmark_results),
                "successful_benchmarks": len([r for r in self.benchmark_results if r.get("success")]),
                "failed_benchmarks": len([r for r in self.benchmark_results if not r.get("success")])
            }
        }
        
        # Add analysis if we have successful results
        successful_results = [r for r in self.benchmark_results if r.get("success")]
        if successful_results:
            fastest = min(successful_results, key=lambda x: x["time_per_step"])
            slowest = max(successful_results, key=lambda x: x["time_per_step"])
            
            export_data["analysis"] = {
                "fastest_configuration": {
                    "variant": fastest["variant"],
                    "config": fastest["config"],
                    "time_per_step": fastest["time_per_step"],
                    "total_time": fastest["total_time"]
                },
                "slowest_configuration": {
                    "variant": slowest["variant"],
                    "config": slowest["config"],
                    "time_per_step": slowest["time_per_step"],
                    "total_time": slowest["total_time"]
                },
                "improvement_factor": slowest["time_per_step"] / fastest["time_per_step"],
                "average_time_per_step": statistics.mean(r["time_per_step"] for r in successful_results)
            }
        
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            print(f"💾 Benchmark results exported to: {filepath}")
            
        except Exception as e:
            print(f"❌ Failed to export results: {e}")
    
    def run_quick_performance_test(self, model_path: str, architecture: str):
        """Run a quick performance test with optimal settings"""
        print("\n⚡ Quick Performance Test")
        print("=" * 50)
        
        try:
            if OPTIMIZATION_AVAILABLE:
                # Create optimized pipeline
                config = OptimizationConfig(
                    architecture_type=architecture,
                    enable_attention_slicing=False,
                    enable_vae_slicing=False,
                    enable_tf32=True,
                    enable_cudnn_benchmark=True
                )
                optimizer = PipelineOptimizer(config)
                pipeline = optimizer.create_optimized_pipeline(model_path, architecture)
            else:
                # Basic pipeline
                pipeline = AutoPipelineForText2Image.from_pretrained(
                    model_path,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    trust_remote_code=True
                )
                if torch.cuda.is_available():
                    pipeline = pipeline.to("cuda")
            
            # Quick test with balanced settings
            test_prompt = "A serene mountain landscape at sunset, photorealistic"
            
            print(f"🎨 Generating test image...")
            print(f"Prompt: {test_prompt}")
            
            start_time = time.time()
            
            generation_kwargs = {
                "prompt": test_prompt,
                "width": 1024,
                "height": 1024,
                "num_inference_steps": 20,
                "generator": torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu").manual_seed(42)
            }
            
            # Use appropriate CFG parameter
            if architecture == "MMDiT" or "qwen" in model_path.lower():
                generation_kwargs["true_cfg_scale"] = 3.5
            else:
                generation_kwargs["guidance_scale"] = 3.5
            
            result = pipeline(**generation_kwargs)
            image = result.images[0]
            
            generation_time = time.time() - start_time
            time_per_step = generation_time / 20
            
            # Save test image
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"quick_test_{timestamp}.png"
            filepath = os.path.join("generated_images", filename)
            os.makedirs("generated_images", exist_ok=True)
            image.save(filepath)
            
            print(f"✅ Generation completed!")
            print(f"⏱️ Total time: {generation_time:.2f}s")
            print(f"⚡ Time per step: {time_per_step:.2f}s")
            print(f"💾 Saved: {filepath}")
            
            # Performance assessment
            if time_per_step <= 2.0:
                print("🏆 Excellent performance! (≤2s/step)")
            elif time_per_step <= 5.0:
                print("✅ Good performance (≤5s/step)")
            elif time_per_step <= 10.0:
                print("⚠️ Moderate performance (≤10s/step)")
            else:
                print("❌ Poor performance (>10s/step)")
                print("💡 Consider optimization or hardware upgrade")
            
        except Exception as e:
            print(f"❌ Quick test failed: {e}")


def main():
    """Main demo function"""
    print("📊 Performance Comparison Demo")
    print("=" * 70)
    print("Benchmarking different Qwen architectures and optimization levels")
    print("")
    
    # Initialize demo
    demo = PerformanceComparisonDemo()
    
    # Detect available models
    model_info = demo.detect_available_models()
    
    if not model_info["models"]:
        print("❌ No models found for benchmarking")
        print("💡 Run: python tools/download_qwen_image.py --optimal")
        sys.exit(1)
    
    # Use the first available model
    model = model_info["models"][0]
    model_path = model["path"]
    architecture = model["architecture"]
    
    print(f"🎯 Benchmarking model: {model['name']}")
    print(f"🏗️ Architecture: {architecture}")
    print(f"📊 Expected performance: {model['expected_performance']}")
    
    # Choose benchmark type
    print("\nSelect benchmark type:")
    print("1. Quick performance test (1 generation)")
    print("2. Comprehensive benchmark (all variants and configurations)")
    print("3. Both")
    
    try:
        choice = input("\nEnter choice (1-3, default=1): ").strip() or "1"
        
        if choice in ["1", "3"]:
            demo.run_quick_performance_test(model_path, architecture)
        
        if choice in ["2", "3"]:
            demo.run_comprehensive_benchmark(model_path, architecture)
            demo.export_benchmark_results()
        
        print("\n🎉 Performance comparison completed!")
        print("=" * 70)
        print("📁 Results saved to: ./generated_images/benchmarks/")
        
        if model_info["optimization_needed"]:
            print(f"\n💡 Optimization Recommendation:")
            print(f"   Current setup is not optimal")
            print(f"   Consider downloading: {model_info['recommended']}")
            print(f"   Command: python tools/download_qwen_image.py --optimal")
        
        print("\n🚀 Next steps:")
        print("• Analyze benchmark results in generated_images/benchmarks/")
        print("• Try different optimization configurations")
        print("• Compare with other models or hardware setups")
        
    except KeyboardInterrupt:
        print("\n⏹️ Benchmark interrupted by user")
    except Exception as e:
        print(f"\n❌ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()