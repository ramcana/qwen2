#!/usr/bin/env python3
"""
Pipeline Performance Integration Example
Demonstrates integration of PerformanceMonitor with PipelineOptimizer
"""

import sys
import os
import time
from typing import Dict, Any

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils.performance_monitor import PerformanceMonitor, monitor_generation_performance
from pipeline_optimizer import PipelineOptimizer, OptimizationConfig


class MockPipeline:
    """Mock pipeline for demonstration purposes"""
    
    def __init__(self, model_name: str, architecture_type: str = "MMDiT"):
        self.model_name = model_name
        self.architecture_type = architecture_type
        self.device = "cuda"
        self.optimized = False
    
    def __call__(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Simulate generation"""
        num_steps = kwargs.get('num_inference_steps', 20)
        
        # Simulate different performance based on model and optimization
        if "Edit" in self.model_name:
            # Slow editing model
            step_time = 0.8 if not self.optimized else 0.4
        else:
            # Fast text-to-image model
            step_time = 0.15 if not self.optimized else 0.12
        
        # Simulate generation steps
        for step in range(num_steps):
            time.sleep(step_time)
        
        return {
            "images": [f"generated_image_{prompt[:10]}"],
            "num_steps": num_steps,
            "step_time": step_time
        }


def demonstrate_integrated_performance_monitoring():
    """Demonstrate integrated performance monitoring with pipeline optimization"""
    print("🚀 Pipeline Performance Integration Demo")
    print("="*60)
    
    # Test scenarios
    scenarios = [
        {
            "name": "Unoptimized Qwen-Image-Edit (Wrong Model)",
            "model_path": "Qwen/Qwen-Image-Edit",
            "architecture": "MMDiT",
            "optimized": False
        },
        {
            "name": "Optimized Qwen-Image (Correct Model)",
            "model_path": "Qwen/Qwen-Image", 
            "architecture": "MMDiT",
            "optimized": True
        }
    ]
    
    results = []
    
    for scenario in scenarios:
        print(f"\n📊 Testing: {scenario['name']}")
        print("-" * 50)
        
        # Create pipeline optimizer
        config = OptimizationConfig(
            architecture_type=scenario['architecture'],
            optimal_steps=20
        )
        optimizer = PipelineOptimizer(config)
        
        # Create mock pipeline (in real usage, this would be the actual pipeline)
        pipeline = MockPipeline(scenario['model_path'], scenario['architecture'])
        pipeline.optimized = scenario['optimized']
        
        # Monitor performance during generation
        with monitor_generation_performance(
            model_name=scenario['model_path'],
            target_time=5.0,
            architecture_type=scenario['architecture'],
            num_steps=20,
            resolution=(1024, 1024)
        ) as monitor:
            
            # Get optimal generation settings from optimizer
            gen_settings = optimizer.configure_generation_settings(scenario['architecture'])
            print(f"   Generation settings: {gen_settings}")
            
            # Simulate generation with performance monitoring
            print(f"   🎨 Generating with {scenario['architecture']} architecture...")
            
            # Monitor individual steps
            for step in range(20):
                monitor.start_step_timing()
                
                # Simulate step work
                if "Edit" in scenario['model_path'] and not scenario['optimized']:
                    time.sleep(0.8)  # Very slow
                elif "Edit" in scenario['model_path'] and scenario['optimized']:
                    time.sleep(0.4)  # Still slow but better
                else:
                    time.sleep(0.12 if scenario['optimized'] else 0.15)  # Fast
                
                step_time = monitor.end_step_timing()
                
                if step % 5 == 0:  # Print every 5th step
                    print(f"      Step {step + 1}/20: {step_time:.3f}s")
        
        # Collect results
        metrics = monitor.current_metrics
        results.append({
            "scenario": scenario['name'],
            "metrics": metrics,
            "settings": gen_settings
        })
        
        # Display results
        print(f"   ✅ Results:")
        print(f"      Total Time: {metrics.total_generation_time:.3f}s")
        print(f"      Per Step: {metrics.generation_time_per_step:.3f}s")
        print(f"      Target Met: {'✅' if metrics.target_met else '❌'}")
        print(f"      Performance Score: {metrics.performance_score:.1f}/100")
    
    # Compare results
    print(f"\n📈 Performance Comparison:")
    print("="*60)
    
    if len(results) >= 2:
        before_metrics = results[0]['metrics']  # Unoptimized
        after_metrics = results[1]['metrics']   # Optimized
        
        # Calculate improvement
        time_improvement = (before_metrics.total_generation_time - after_metrics.total_generation_time) / before_metrics.total_generation_time * 100
        improvement_factor = before_metrics.total_generation_time / after_metrics.total_generation_time
        
        print(f"Before Optimization (Wrong Model):")
        print(f"   Model: {before_metrics.model_name}")
        print(f"   Total Time: {before_metrics.total_generation_time:.3f}s")
        print(f"   Per Step: {before_metrics.generation_time_per_step:.3f}s")
        print(f"   Target Met: {'✅' if before_metrics.target_met else '❌'}")
        
        print(f"\nAfter Optimization (Correct Model):")
        print(f"   Model: {after_metrics.model_name}")
        print(f"   Total Time: {after_metrics.total_generation_time:.3f}s")
        print(f"   Per Step: {after_metrics.generation_time_per_step:.3f}s")
        print(f"   Target Met: {'✅' if after_metrics.target_met else '❌'}")
        
        print(f"\nImprovement:")
        print(f"   Speed Improvement: {time_improvement:.1f}%")
        print(f"   Improvement Factor: {improvement_factor:.1f}x")
        print(f"   Target Achievement: {before_metrics.target_met} → {after_metrics.target_met}")
    
    return results


def demonstrate_performance_validation():
    """Demonstrate performance validation with different configurations"""
    print(f"\n🎯 Performance Validation Demo")
    print("="*60)
    
    # Test different optimization configurations
    configs = [
        {
            "name": "Memory-Saving (Low VRAM)",
            "config": OptimizationConfig(
                enable_attention_slicing=True,
                enable_vae_slicing=True,
                optimal_steps=25
            ),
            "expected_performance": "slower"
        },
        {
            "name": "Performance-Optimized (High VRAM)",
            "config": OptimizationConfig(
                enable_attention_slicing=False,
                enable_vae_slicing=False,
                enable_tf32=True,
                optimal_steps=20
            ),
            "expected_performance": "faster"
        }
    ]
    
    validation_results = []
    
    for config_info in configs:
        print(f"\n🔧 Testing Configuration: {config_info['name']}")
        
        config = config_info['config']
        optimizer = PipelineOptimizer(config)
        
        # Create performance monitor
        monitor = PerformanceMonitor(target_generation_time=5.0)
        
        # Simulate generation with this configuration
        with monitor.monitor_generation(
            model_name="Qwen-Image",
            architecture_type="MMDiT",
            num_steps=config.optimal_steps,
            resolution=(1024, 1024)
        ):
            # Simulate performance based on configuration
            if config_info['expected_performance'] == "slower":
                # Memory-saving features slow down generation
                time.sleep(6.0)  # Slower than target
            else:
                # Performance optimizations speed up generation
                time.sleep(3.0)  # Faster than target
        
        # Validate performance
        metrics = monitor.current_metrics
        is_valid = monitor.validate_performance_target(metrics.total_generation_time)
        
        print(f"   Configuration Details:")
        print(f"      Attention Slicing: {config.enable_attention_slicing}")
        print(f"      VAE Slicing: {config.enable_vae_slicing}")
        print(f"      TF32 Enabled: {config.enable_tf32}")
        print(f"      Optimal Steps: {config.optimal_steps}")
        
        print(f"   Performance Results:")
        print(f"      Total Time: {metrics.total_generation_time:.3f}s")
        print(f"      Target Met: {'✅' if is_valid else '❌'}")
        print(f"      Performance Score: {metrics.performance_score:.1f}/100")
        
        validation_results.append({
            "config_name": config_info['name'],
            "config": config,
            "metrics": metrics,
            "valid": is_valid
        })
    
    # Summary
    print(f"\n📋 Validation Summary:")
    passed = sum(1 for r in validation_results if r['valid'])
    total = len(validation_results)
    print(f"   Configurations Passed: {passed}/{total}")
    
    if passed < total:
        print(f"   💡 Recommendations:")
        for result in validation_results:
            if not result['valid']:
                print(f"      - {result['config_name']}: Consider reducing memory-saving features")
    
    return validation_results


def demonstrate_real_time_monitoring():
    """Demonstrate real-time performance monitoring during generation"""
    print(f"\n⏱️ Real-Time Monitoring Demo")
    print("="*60)
    
    monitor = PerformanceMonitor(target_generation_time=5.0)
    
    print(f"\n🎨 Starting real-time monitored generation...")
    
    with monitor.monitor_generation(
        model_name="Qwen-Image",
        architecture_type="MMDiT",
        num_steps=15,
        resolution=(1024, 1024)
    ):
        print(f"   📊 Real-time step monitoring:")
        
        for step in range(15):
            monitor.start_step_timing()
            
            # Simulate variable step times (some steps are slower)
            if step in [5, 10]:  # Simulate slower steps
                time.sleep(0.2)
            else:
                time.sleep(0.1)
            
            step_time = monitor.end_step_timing()
            
            # Real-time feedback
            status = "🟢" if step_time < 0.15 else "🟡" if step_time < 0.25 else "🔴"
            print(f"      Step {step + 1:2d}/15: {step_time:.3f}s {status}")
            
            # Early warning if performance is degrading
            if step > 5:  # After a few steps
                avg_step_time = sum(monitor._step_times) / len(monitor._step_times)
                projected_total = avg_step_time * 15
                
                if projected_total > monitor.target_generation_time and step == 6:
                    print(f"      ⚠️  Warning: Projected total time {projected_total:.1f}s exceeds target!")
    
    # Final results
    metrics = monitor.current_metrics
    print(f"\n   📊 Final Results:")
    print(f"      Total Time: {metrics.total_generation_time:.3f}s")
    print(f"      Average Step Time: {metrics.generation_time_per_step:.3f}s")
    print(f"      Target Met: {'✅' if metrics.target_met else '❌'}")
    print(f"      Performance Score: {metrics.performance_score:.1f}/100")
    
    return monitor


def main():
    """Run all integration demos"""
    try:
        # Run integration demos
        integration_results = demonstrate_integrated_performance_monitoring()
        validation_results = demonstrate_performance_validation()
        realtime_monitor = demonstrate_real_time_monitoring()
        
        # Final summary
        print(f"\n🎉 Integration Demo Complete!")
        print("="*60)
        
        print(f"\n✅ Key Integration Features Demonstrated:")
        print(f"   🔧 Pipeline Optimizer + Performance Monitor integration")
        print(f"   📊 Before/after performance comparison")
        print(f"   🎯 Configuration-based performance validation")
        print(f"   ⏱️ Real-time step monitoring with early warnings")
        print(f"   📈 Comprehensive performance metrics collection")
        
        print(f"\n💡 Integration Benefits:")
        print(f"   - Automatic performance validation during optimization")
        print(f"   - Real-time feedback for performance issues")
        print(f"   - Detailed metrics for optimization decisions")
        print(f"   - Easy integration with existing pipeline code")
        print(f"   - MMDiT-specific performance monitoring")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Integration demo failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())