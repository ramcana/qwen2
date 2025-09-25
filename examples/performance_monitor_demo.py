#!/usr/bin/env python3
"""
Performance Monitor Demo
Demonstrates comprehensive performance monitoring for MMDiT architecture
"""

import sys
import os
import time
import torch
from typing import Dict, Any

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils.performance_monitor import (
    PerformanceMonitor,
    PerformanceMetrics,
    create_mmdit_performance_monitor,
    monitor_generation_performance
)
from pipeline_optimizer import PipelineOptimizer, OptimizationConfig


def simulate_model_loading(model_name: str, load_time: float = 2.0) -> str:
    """Simulate model loading with configurable time"""
    print(f"Loading model: {model_name}")
    time.sleep(load_time)
    return f"loaded_{model_name}"


def simulate_generation_step(step_num: int, step_time: float = 0.15) -> Dict[str, Any]:
    """Simulate a single generation step"""
    time.sleep(step_time)
    return {
        "step": step_num,
        "latents": f"latents_step_{step_num}",
        "time": step_time
    }


def simulate_mmdit_generation(monitor: PerformanceMonitor, num_steps: int = 20) -> Dict[str, Any]:
    """Simulate MMDiT generation with step-by-step monitoring"""
    print(f"🚀 Starting MMDiT generation with {num_steps} steps")
    
    results = []
    
    for step in range(num_steps):
        # Monitor each step
        monitor.start_step_timing()
        
        # Simulate generation work (faster for MMDiT)
        step_result = simulate_generation_step(step, step_time=0.12)
        
        step_time = monitor.end_step_timing()
        step_result["measured_time"] = step_time
        results.append(step_result)
        
        print(f"   Step {step + 1}/{num_steps}: {step_time:.3f}s")
    
    return {
        "steps": results,
        "total_steps": num_steps,
        "architecture": "MMDiT"
    }


def simulate_unet_generation(monitor: PerformanceMonitor, num_steps: int = 20) -> Dict[str, Any]:
    """Simulate UNet generation (slower) for comparison"""
    print(f"🐌 Starting UNet generation with {num_steps} steps")
    
    results = []
    
    for step in range(num_steps):
        # Monitor each step
        monitor.start_step_timing()
        
        # Simulate generation work (slower for UNet)
        step_result = simulate_generation_step(step, step_time=0.25)
        
        step_time = monitor.end_step_timing()
        step_result["measured_time"] = step_time
        results.append(step_result)
        
        print(f"   Step {step + 1}/{num_steps}: {step_time:.3f}s")
    
    return {
        "steps": results,
        "total_steps": num_steps,
        "architecture": "UNet"
    }


def demo_basic_performance_monitoring():
    """Demonstrate basic performance monitoring"""
    print("\n" + "="*60)
    print("🔍 DEMO: Basic Performance Monitoring")
    print("="*60)
    
    # Create monitor with 5-second target
    monitor = PerformanceMonitor(target_generation_time=5.0)
    
    # Measure model loading
    print("\n📥 Measuring model loading time...")
    model, load_time = monitor.measure_model_load_time(
        simulate_model_loading, "Qwen-Image", 1.5
    )
    print(f"Model loaded: {model} in {load_time:.3f}s")
    
    # Monitor generation with context manager
    print("\n🎨 Monitoring generation performance...")
    with monitor.monitor_generation(
        model_name="Qwen-Image",
        architecture_type="MMDiT",
        num_steps=20,
        resolution=(1024, 1024)
    ) as perf_monitor:
        # Simulate generation
        generation_result = simulate_mmdit_generation(perf_monitor, num_steps=20)
    
    # Display results
    print("\n📊 Performance Summary:")
    summary = monitor.get_performance_summary()
    current = summary["current_generation"]
    
    print(f"   Total Time: {current['total_time']:.3f}s")
    print(f"   Per Step: {current['per_step_time']:.3f}s")
    print(f"   Target Met: {'✅' if current['target_met'] else '❌'}")
    print(f"   Performance Score: {current['performance_score']:.1f}/100")
    
    return monitor.current_metrics


def demo_mmdit_vs_unet_comparison():
    """Demonstrate MMDiT vs UNet performance comparison"""
    print("\n" + "="*60)
    print("⚡ DEMO: MMDiT vs UNet Performance Comparison")
    print("="*60)
    
    # Test MMDiT performance
    print("\n🚀 Testing MMDiT (Qwen-Image) Performance:")
    mmdit_monitor = create_mmdit_performance_monitor(target_time=5.0)
    
    with mmdit_monitor.monitor_generation(
        model_name="Qwen-Image",
        architecture_type="MMDiT",
        num_steps=20,
        resolution=(1024, 1024)
    ):
        mmdit_result = simulate_mmdit_generation(mmdit_monitor, num_steps=20)
    
    mmdit_metrics = mmdit_monitor.current_metrics
    
    # Test UNet performance (for comparison)
    print("\n🐌 Testing UNet (Traditional) Performance:")
    unet_monitor = PerformanceMonitor(target_generation_time=5.0)
    
    with unet_monitor.monitor_generation(
        model_name="Stable-Diffusion",
        architecture_type="UNet",
        num_steps=20,
        resolution=(1024, 1024)
    ):
        unet_result = simulate_unet_generation(unet_monitor, num_steps=20)
    
    unet_metrics = unet_monitor.current_metrics
    
    # Compare performance
    print("\n📈 Performance Comparison:")
    comparison = mmdit_monitor.get_before_after_comparison(unet_metrics)
    
    print(f"   MMDiT (Qwen-Image):")
    print(f"     Total Time: {mmdit_metrics.total_generation_time:.3f}s")
    print(f"     Per Step: {mmdit_metrics.generation_time_per_step:.3f}s")
    print(f"     Target Met: {'✅' if mmdit_metrics.target_met else '❌'}")
    print(f"     Score: {mmdit_metrics.performance_score:.1f}/100")
    
    print(f"   UNet (Traditional):")
    print(f"     Total Time: {unet_metrics.total_generation_time:.3f}s")
    print(f"     Per Step: {unet_metrics.generation_time_per_step:.3f}s")
    print(f"     Target Met: {'✅' if unet_metrics.target_met else '❌'}")
    print(f"     Score: {unet_metrics.performance_score:.1f}/100")
    
    print(f"   Improvement:")
    print(f"     Speed Improvement: {comparison['improvement']['total_time_percent']:.1f}%")
    print(f"     Improvement Factor: {comparison['summary']['improvement_factor']:.1f}x")
    print(f"     Significant Improvement: {'✅' if comparison['summary']['significant_improvement'] else '❌'}")
    
    return mmdit_metrics, unet_metrics


def demo_performance_validation():
    """Demonstrate performance validation and diagnostics"""
    print("\n" + "="*60)
    print("🎯 DEMO: Performance Validation and Diagnostics")
    print("="*60)
    
    # Test with different performance scenarios
    scenarios = [
        ("Excellent Performance", 2.0, "MMDiT", "Qwen-Image"),
        ("Good Performance", 4.5, "MMDiT", "Qwen-Image"),
        ("Poor Performance", 8.0, "UNet", "Qwen-Image-Edit"),
        ("Very Poor Performance", 15.0, "UNet", "Qwen-Image-Edit")
    ]
    
    results = []
    
    for scenario_name, target_time, arch_type, model_name in scenarios:
        print(f"\n🧪 Testing: {scenario_name}")
        
        monitor = PerformanceMonitor(target_generation_time=5.0)
        
        # Simulate generation with specific timing
        with monitor.monitor_generation(
            model_name=model_name,
            architecture_type=arch_type,
            num_steps=20,
            resolution=(1024, 1024)
        ):
            # Simulate work to hit target time
            time.sleep(target_time)
        
        metrics = monitor.current_metrics
        
        print(f"   Time: {metrics.total_generation_time:.3f}s")
        print(f"   Target Met: {'✅' if metrics.target_met else '❌'}")
        print(f"   Score: {metrics.performance_score:.1f}/100")
        
        # Validate performance
        is_valid = monitor.validate_performance_target(metrics.total_generation_time)
        
        results.append({
            "scenario": scenario_name,
            "metrics": metrics,
            "valid": is_valid
        })
    
    # Summary
    print(f"\n📋 Validation Summary:")
    passed = sum(1 for r in results if r["valid"])
    total = len(results)
    print(f"   Scenarios Passed: {passed}/{total}")
    print(f"   Success Rate: {passed/total*100:.1f}%")
    
    return results


def demo_memory_monitoring():
    """Demonstrate memory usage monitoring"""
    print("\n" + "="*60)
    print("💾 DEMO: Memory Usage Monitoring")
    print("="*60)
    
    monitor = PerformanceMonitor(target_generation_time=5.0)
    
    print("\n🔍 Monitoring memory usage during generation...")
    
    with monitor.monitor_generation(
        model_name="Qwen-Image",
        architecture_type="MMDiT",
        num_steps=20,
        resolution=(1024, 1024)
    ):
        # Simulate memory-intensive generation
        if torch.cuda.is_available():
            # Allocate some GPU memory to simulate model usage
            dummy_tensor = torch.randn(1000, 1000, device='cuda')
            time.sleep(2.0)
            del dummy_tensor
            torch.cuda.empty_cache()
        else:
            time.sleep(2.0)
    
    # Display memory metrics
    metrics = monitor.current_metrics
    summary = monitor.get_performance_summary()
    
    print(f"\n💾 Memory Usage Summary:")
    if torch.cuda.is_available():
        print(f"   GPU Memory Used: {metrics.gpu_memory_used_gb:.2f}GB")
        print(f"   GPU Utilization: {metrics.gpu_memory_utilization_percent:.1f}%")
        print(f"   GPU Total: {metrics.gpu_memory_total_gb:.1f}GB")
        print(f"   GPU Name: {metrics.gpu_name}")
    else:
        print("   GPU: Not available")
    
    print(f"   System Memory Used: {metrics.system_memory_used_gb:.2f}GB")
    print(f"   System Utilization: {metrics.system_memory_utilization_percent:.1f}%")
    
    return metrics


def demo_json_export():
    """Demonstrate JSON export functionality"""
    print("\n" + "="*60)
    print("📄 DEMO: JSON Export Functionality")
    print("="*60)
    
    monitor = PerformanceMonitor(target_generation_time=5.0)
    
    # Generate some performance data
    with monitor.monitor_generation(
        model_name="Qwen-Image",
        architecture_type="MMDiT",
        num_steps=20,
        resolution=(1024, 1024)
    ):
        simulate_mmdit_generation(monitor, num_steps=20)
    
    # Export to JSON
    export_path = "performance_metrics_demo.json"
    monitor.export_metrics_to_json(export_path)
    
    print(f"📄 Metrics exported to: {export_path}")
    
    # Read and display sample of exported data
    import json
    try:
        with open(export_path, 'r') as f:
            data = json.load(f)
        
        print(f"\n📊 Exported Data Sample:")
        current_metrics = data["current_metrics"]
        print(f"   Model: {current_metrics['model_name']}")
        print(f"   Architecture: {current_metrics['architecture_type']}")
        print(f"   Total Time: {current_metrics['total_generation_time']:.3f}s")
        print(f"   Target Met: {current_metrics['target_met']}")
        print(f"   Export Timestamp: {data['export_timestamp']}")
        
    except Exception as e:
        print(f"❌ Error reading exported file: {e}")
    
    return export_path


def demo_convenience_functions():
    """Demonstrate convenience functions and context managers"""
    print("\n" + "="*60)
    print("🛠️ DEMO: Convenience Functions")
    print("="*60)
    
    print("\n🎯 Using convenience context manager:")
    
    # Use the convenience context manager
    with monitor_generation_performance(
        model_name="Qwen-Image",
        target_time=4.0,
        architecture_type="MMDiT",
        num_steps=15,
        resolution=(512, 512)
    ) as monitor:
        print(f"   Monitor created with {monitor.target_generation_time}s target")
        print(f"   Model: {monitor.current_metrics.model_name}")
        print(f"   Architecture: {monitor.current_metrics.architecture_type}")
        
        # Simulate generation
        for step in range(15):
            monitor.start_step_timing()
            time.sleep(0.1)  # 100ms per step
            step_time = monitor.end_step_timing()
            if step % 5 == 0:  # Print every 5th step
                print(f"   Step {step + 1}: {step_time:.3f}s")
    
    # Results are automatically captured
    print(f"\n✅ Generation completed:")
    print(f"   Total Time: {monitor.current_metrics.total_generation_time:.3f}s")
    print(f"   Average Step Time: {monitor.current_metrics.generation_time_per_step:.3f}s")
    print(f"   Target Met: {'✅' if monitor.current_metrics.target_met else '❌'}")
    
    return monitor


def main():
    """Run all performance monitoring demos"""
    print("🚀 Performance Monitor Demo Suite")
    print("Demonstrating comprehensive performance monitoring for MMDiT architecture")
    
    try:
        # Run all demos
        demo_basic_performance_monitoring()
        mmdit_metrics, unet_metrics = demo_mmdit_vs_unet_comparison()
        validation_results = demo_performance_validation()
        memory_metrics = demo_memory_monitoring()
        export_path = demo_json_export()
        convenience_monitor = demo_convenience_functions()
        
        # Final summary
        print("\n" + "="*60)
        print("🎉 DEMO SUITE COMPLETE")
        print("="*60)
        
        print(f"\n📊 Overall Results:")
        print(f"   MMDiT Performance: {mmdit_metrics.total_generation_time:.3f}s")
        print(f"   UNet Performance: {unet_metrics.total_generation_time:.3f}s")
        print(f"   Speed Improvement: {unet_metrics.total_generation_time / mmdit_metrics.total_generation_time:.1f}x")
        
        validation_passed = sum(1 for r in validation_results if r["valid"])
        print(f"   Validation Tests Passed: {validation_passed}/{len(validation_results)}")
        
        if torch.cuda.is_available():
            print(f"   GPU Memory Monitored: ✅")
            print(f"   GPU: {memory_metrics.gpu_name}")
        else:
            print(f"   GPU Memory Monitored: ❌ (No GPU available)")
        
        print(f"   JSON Export: ✅ ({export_path})")
        print(f"   Convenience Functions: ✅")
        
        print(f"\n💡 Key Features Demonstrated:")
        print(f"   ✅ Comprehensive timing measurement")
        print(f"   ✅ Step-by-step performance tracking")
        print(f"   ✅ Memory usage monitoring")
        print(f"   ✅ Performance validation against targets")
        print(f"   ✅ MMDiT vs UNet architecture comparison")
        print(f"   ✅ Before/after performance comparison")
        print(f"   ✅ JSON export for analysis")
        print(f"   ✅ Context managers for easy integration")
        print(f"   ✅ Diagnostic information and warnings")
        
        print(f"\n🎯 Performance Monitoring System Ready!")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())