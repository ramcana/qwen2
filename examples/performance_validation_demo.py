#!/usr/bin/env python3
"""
Performance Validation and Benchmarking Demo
Demonstrates comprehensive performance validation with multimodal support
"""

import sys
import os
import time
import logging
from pathlib import Path
from typing import List, Dict, Any

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils.performance_validator import (
    PerformanceValidator,
    BenchmarkSuite,
    create_performance_validator,
    create_default_benchmark_suite,
    validate_performance_improvement
)
from utils.performance_monitor import PerformanceMetrics

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MockImageGenerator:
    """Mock image generator for demonstration purposes"""
    
    def __init__(self, model_name: str, base_generation_time: float = 0.05):
        """
        Initialize mock generator
        
        Args:
            model_name: Name of the model
            base_generation_time: Base time per generation in seconds
        """
        self.model_name = model_name
        self.base_generation_time = base_generation_time
        self.generation_count = 0
    
    def generate_image(self, prompt: str, num_steps: int = 20, 
                      resolution: tuple = (1024, 1024)) -> str:
        """
        Mock image generation
        
        Args:
            prompt: Text prompt
            num_steps: Number of inference steps
            resolution: Image resolution
            
        Returns:
            Mock generated image identifier
        """
        self.generation_count += 1
        
        # Simulate generation time based on parameters
        complexity_factor = (resolution[0] * resolution[1]) / (1024 * 1024)  # Resolution factor
        step_factor = num_steps / 20  # Step count factor
        
        generation_time = self.base_generation_time * complexity_factor * step_factor
        time.sleep(generation_time)
        
        return f"{self.model_name}_generated_image_{self.generation_count}_{resolution[0]}x{resolution[1]}"
    
    def __call__(self, prompt: str) -> str:
        """Make generator callable"""
        return self.generate_image(prompt)


class MockQwen2VLProcessor:
    """Mock Qwen2-VL processor for multimodal demonstrations"""
    
    def __init__(self, base_processing_time: float = 0.02):
        """
        Initialize mock processor
        
        Args:
            base_processing_time: Base processing time in seconds
        """
        self.base_processing_time = base_processing_time
        self.processing_count = 0
    
    def process_text_and_image(self, text_prompt: str, image_path: str = None) -> str:
        """
        Mock multimodal processing
        
        Args:
            text_prompt: Text prompt
            image_path: Optional image path
            
        Returns:
            Mock processing result
        """
        self.processing_count += 1
        
        # Simulate processing time
        processing_time = self.base_processing_time
        if image_path:
            processing_time *= 1.5  # Image processing takes longer
        
        time.sleep(processing_time)
        
        if image_path:
            return f"multimodal_result_{self.processing_count}_text_and_image"
        else:
            return f"text_result_{self.processing_count}_text_only"
    
    def __call__(self, text_prompt: str, image_path: str = None) -> str:
        """Make processor callable"""
        return self.process_text_and_image(text_prompt, image_path)


def demonstrate_basic_performance_validation():
    """Demonstrate basic performance validation functionality"""
    logger.info("🚀 Starting Basic Performance Validation Demo")
    
    # Create performance validator
    validator = create_performance_validator(
        target_improvement=10.0,  # 10x improvement target
        target_time=2.0           # 2 second target
    )
    
    # Create mock generators
    slow_generator = MockImageGenerator("Slow-Model", base_generation_time=0.2)  # 200ms
    fast_generator = MockImageGenerator("Fast-Model", base_generation_time=0.02)  # 20ms
    
    logger.info("📊 Running speed improvement validation...")
    
    # Test speed improvement
    validation_result = validator.run_speed_improvement_validation(
        before_function=slow_generator,
        after_function=fast_generator,
        test_prompt="A beautiful landscape with mountains and lakes"
    )
    
    # Display results
    logger.info("✅ Speed Improvement Validation Results:")
    logger.info(f"   Improvement Factor: {validation_result['improvement_factor']:.1f}x")
    logger.info(f"   Target Factor: {validation_result['target_improvement_factor']:.1f}x")
    logger.info(f"   Status: {validation_result['validation_status']}")
    logger.info(f"   Time Saved: {validation_result['time_saved_seconds']:.3f}s")
    
    return validation_result


def demonstrate_end_to_end_testing():
    """Demonstrate end-to-end performance testing"""
    logger.info("🎯 Starting End-to-End Performance Testing Demo")
    
    # Create validator
    validator = create_performance_validator(target_time=1.0)
    
    # Create optimized generator
    optimized_generator = MockImageGenerator("Optimized-MMDiT", base_generation_time=0.03)
    
    # Test prompts
    test_prompts = [
        "A serene mountain landscape at sunset",
        "A futuristic city with flying cars",
        "A peaceful garden with blooming flowers",
        "An abstract geometric pattern in vibrant colors",
        "A portrait of a wise old wizard"
    ]
    
    logger.info(f"🧪 Running end-to-end test with {len(test_prompts)} prompts...")
    
    # Run end-to-end test
    e2e_result = validator.run_end_to_end_performance_test(
        generator_function=optimized_generator,
        test_prompts=test_prompts,
        model_name="Optimized-MMDiT-Model"
    )
    
    # Display results
    logger.info("✅ End-to-End Test Results:")
    logger.info(f"   Success Rate: {e2e_result.success_rate:.1f}%")
    logger.info(f"   Average Generation Time: {e2e_result.total_time:.3f}s")
    logger.info(f"   Performance Score: {e2e_result.performance_score:.1f}/100")
    logger.info(f"   Target Met: {'Yes' if e2e_result.target_met else 'No'}")
    logger.info(f"   Error Count: {e2e_result.error_count}")
    
    return e2e_result


def demonstrate_gpu_utilization_monitoring():
    """Demonstrate GPU utilization and memory monitoring"""
    logger.info("🔥 Starting GPU Utilization Monitoring Demo")
    
    # Create validator
    validator = create_performance_validator()
    
    # Create generator that simulates GPU-intensive work
    gpu_intensive_generator = MockImageGenerator("GPU-Intensive", base_generation_time=0.05)
    
    logger.info("📈 Running GPU utilization benchmark for 5 seconds...")
    
    # Run GPU utilization benchmark
    gpu_result = validator.run_gpu_utilization_benchmark(
        generator_function=gpu_intensive_generator,
        test_prompt="High-resolution detailed artwork",
        duration_seconds=5.0
    )
    
    # Display results
    logger.info("✅ GPU Utilization Benchmark Results:")
    logger.info(f"   GPU Utilization: {gpu_result.gpu_utilization_percent:.1f}%")
    logger.info(f"   GPU Memory Used: {gpu_result.gpu_memory_used_gb:.2f}GB")
    logger.info(f"   GPU Memory Peak: {gpu_result.gpu_memory_peak_gb:.2f}GB")
    logger.info(f"   Memory Efficiency: {gpu_result.gpu_memory_efficiency:.1f}%")
    logger.info(f"   Success Rate: {gpu_result.success_rate:.1f}%")
    
    # Display generation rate info from warnings
    for warning in gpu_result.warnings:
        if "Generations per second" in warning or "Total generations" in warning:
            logger.info(f"   {warning}")
    
    return gpu_result


def demonstrate_multimodal_benchmarking():
    """Demonstrate multimodal performance benchmarking"""
    logger.info("🎭 Starting Multimodal Performance Benchmarking Demo")
    
    # Create validator
    validator = create_performance_validator()
    
    # Create mock Qwen2-VL processor
    qwen2vl_processor = MockQwen2VLProcessor(base_processing_time=0.03)
    
    # Test prompts and images
    text_prompts = [
        "Describe this image in detail",
        "What are the main colors in this scene?",
        "Generate a creative story based on this image",
        "Analyze the composition and lighting"
    ]
    
    image_paths = [
        "mock_image_1.jpg",
        "mock_image_2.jpg",
        "mock_image_3.jpg"
    ]
    
    logger.info(f"🧠 Running multimodal benchmark with {len(text_prompts)} prompts...")
    
    # Run multimodal benchmark
    multimodal_result = validator.run_multimodal_performance_benchmark(
        qwen2vl_function=qwen2vl_processor,
        text_prompts=text_prompts,
        image_paths=image_paths
    )
    
    # Display results
    logger.info("✅ Multimodal Benchmark Results:")
    logger.info(f"   Model: {multimodal_result.model_name}")
    logger.info(f"   Architecture: {multimodal_result.architecture_type}")
    logger.info(f"   Success Rate: {multimodal_result.success_rate:.1f}%")
    logger.info(f"   Text Processing Time: {multimodal_result.text_processing_time:.3f}s")
    logger.info(f"   Image Analysis Time: {multimodal_result.image_analysis_time:.3f}s")
    logger.info(f"   Total Time: {multimodal_result.total_time:.3f}s")
    logger.info(f"   Multimodal Enabled: {multimodal_result.multimodal_enabled}")
    
    return multimodal_result


def demonstrate_regression_test_suite():
    """Demonstrate comprehensive regression test suite"""
    logger.info("🧪 Starting Regression Test Suite Demo")
    
    # Create validator
    validator = create_performance_validator(target_time=1.0)
    
    # Create test generator
    test_generator = MockImageGenerator("Regression-Test-Model", base_generation_time=0.04)
    
    # Create custom benchmark suite
    suite = BenchmarkSuite(
        name="Demo Regression Test Suite",
        target_improvement_factor=5.0,
        target_generation_time=1.0,
        test_resolutions=[(512, 512), (768, 768), (1024, 1024)],
        test_step_counts=[10, 20],
        test_batch_sizes=[1, 2],
        include_multimodal_tests=True
    )
    
    logger.info(f"🔬 Running regression test suite: {suite.name}")
    logger.info(f"   Resolutions: {suite.test_resolutions}")
    logger.info(f"   Step counts: {suite.test_step_counts}")
    logger.info(f"   Batch sizes: {suite.test_batch_sizes}")
    
    # Run regression test suite
    suite_results = validator.run_regression_test_suite(
        generator_function=test_generator,
        benchmark_suite=suite
    )
    
    # Display results
    summary = suite_results["summary"]
    logger.info("✅ Regression Test Suite Results:")
    logger.info(f"   Total Tests: {summary['total_tests']}")
    logger.info(f"   Passed Tests: {summary['passed_tests']}")
    logger.info(f"   Failed Tests: {summary['failed_tests']}")
    logger.info(f"   Pass Rate: {summary['pass_rate']:.1f}%")
    logger.info(f"   Overall Status: {summary['overall_status']}")
    
    # Show individual test results summary
    test_results = suite_results["test_results"]
    logger.info(f"📊 Individual Test Performance:")
    for result in test_results[:5]:  # Show first 5 tests
        logger.info(f"   {result['test_name']}: {result['total_time']:.3f}s "
                   f"({'PASS' if result['target_met'] else 'FAIL'})")
    
    if len(test_results) > 5:
        logger.info(f"   ... and {len(test_results) - 5} more tests")
    
    return suite_results


def demonstrate_architecture_comparison():
    """Demonstrate MMDiT vs UNet architecture comparison"""
    logger.info("⚡ Starting Architecture Comparison Demo")
    
    # Create validator
    validator = create_performance_validator()
    
    # Create generators for different architectures
    unet_generator = MockImageGenerator("UNet-Model", base_generation_time=0.15)      # Slower
    mmdit_generator = MockImageGenerator("MMDiT-Model", base_generation_time=0.03)    # Faster
    
    test_prompt = "A detailed fantasy landscape with dragons"
    
    logger.info("🏗️ Testing UNet architecture...")
    
    # Test UNet performance
    with validator.benchmark_generation(
        test_name="unet_architecture_test",
        model_name="UNet-Stable-Diffusion",
        architecture_type="UNet",
        num_steps=20,
        resolution=(1024, 1024)
    ) as (unet_result, _):
        unet_image = unet_generator(test_prompt)
        unet_result.success_rate = 100.0 if unet_image else 0.0
    
    logger.info("🚀 Testing MMDiT architecture...")
    
    # Test MMDiT performance
    with validator.benchmark_generation(
        test_name="mmdit_architecture_test",
        model_name="MMDiT-Qwen-Image",
        architecture_type="MMDiT",
        num_steps=20,
        resolution=(1024, 1024)
    ) as (mmdit_result, _):
        mmdit_image = mmdit_generator(test_prompt)
        mmdit_result.success_rate = 100.0 if mmdit_image else 0.0
    
    # Compare results
    unet_result = next(r for r in validator.benchmark_results if r.architecture_type == "UNet")
    mmdit_result = next(r for r in validator.benchmark_results if r.architecture_type == "MMDiT")
    
    improvement_factor = unet_result.total_time / mmdit_result.total_time if mmdit_result.total_time > 0 else 0
    
    logger.info("✅ Architecture Comparison Results:")
    logger.info(f"   UNet Performance:")
    logger.info(f"     Generation Time: {unet_result.total_time:.3f}s")
    logger.info(f"     Performance Score: {unet_result.performance_score:.1f}/100")
    logger.info(f"     Target Met: {'Yes' if unet_result.target_met else 'No'}")
    
    logger.info(f"   MMDiT Performance:")
    logger.info(f"     Generation Time: {mmdit_result.total_time:.3f}s")
    logger.info(f"     Performance Score: {mmdit_result.performance_score:.1f}/100")
    logger.info(f"     Target Met: {'Yes' if mmdit_result.target_met else 'No'}")
    
    logger.info(f"   Improvement: MMDiT is {improvement_factor:.1f}x faster than UNet")
    
    return {"unet_result": unet_result, "mmdit_result": mmdit_result, "improvement_factor": improvement_factor}


def demonstrate_performance_summary_and_export():
    """Demonstrate performance summary generation and export"""
    logger.info("📊 Starting Performance Summary and Export Demo")
    
    # Create validator with some test data
    validator = create_performance_validator()
    
    # Run a few quick tests to generate data
    generators = [
        ("Fast-Model", 0.02),
        ("Medium-Model", 0.05),
        ("Slow-Model", 0.12)
    ]
    
    for model_name, gen_time in generators:
        generator = MockImageGenerator(model_name, base_generation_time=gen_time)
        
        with validator.benchmark_generation(
            test_name=f"summary_test_{model_name.lower()}",
            model_name=model_name,
            architecture_type="MMDiT"
        ) as (result, _):
            generated = generator("test prompt for summary")
            result.success_rate = 100.0 if generated else 0.0
    
    # Generate performance summary
    logger.info("📈 Generating performance summary...")
    summary = validator.get_performance_summary()
    
    # Display summary
    logger.info("✅ Performance Summary:")
    logger.info(f"   Overall Status: {summary['overall_status']}")
    
    stats = summary["test_statistics"]
    logger.info(f"   Test Statistics:")
    logger.info(f"     Total Tests: {stats['total_tests']}")
    logger.info(f"     Passed Tests: {stats['passed_tests']}")
    logger.info(f"     Pass Rate: {stats['pass_rate']:.1f}%")
    logger.info(f"     Average Performance Score: {stats['average_performance_score']:.1f}")
    
    targets = summary["performance_targets"]
    logger.info(f"   Performance Targets:")
    logger.info(f"     Target Generation Time: {targets['target_generation_time']:.1f}s")
    logger.info(f"     Time Target Met: {'Yes' if targets['time_target_met'] else 'No'}")
    
    best = summary["best_performance"]
    worst = summary["worst_performance"]
    logger.info(f"   Best Performance: {best['test_name']} ({best['performance_score']:.1f}/100)")
    logger.info(f"   Worst Performance: {worst['test_name']} ({worst['performance_score']:.1f}/100)")
    
    # Display recommendations
    recommendations = summary["recommendations"]
    logger.info(f"   Recommendations:")
    for i, rec in enumerate(recommendations[:3], 1):  # Show first 3
        logger.info(f"     {i}. {rec}")
    
    # Export results
    export_path = "demo_performance_results.json"
    logger.info(f"💾 Exporting results to {export_path}...")
    validator.export_benchmark_results(export_path)
    
    logger.info(f"✅ Results exported successfully!")
    
    return summary


def run_complete_demo():
    """Run complete performance validation demonstration"""
    logger.info("🎉 Starting Complete Performance Validation Demo")
    logger.info("=" * 80)
    
    try:
        # 1. Basic performance validation
        logger.info("\n" + "=" * 80)
        basic_result = demonstrate_basic_performance_validation()
        
        # 2. End-to-end testing
        logger.info("\n" + "=" * 80)
        e2e_result = demonstrate_end_to_end_testing()
        
        # 3. GPU utilization monitoring
        logger.info("\n" + "=" * 80)
        gpu_result = demonstrate_gpu_utilization_monitoring()
        
        # 4. Multimodal benchmarking
        logger.info("\n" + "=" * 80)
        multimodal_result = demonstrate_multimodal_benchmarking()
        
        # 5. Regression test suite
        logger.info("\n" + "=" * 80)
        regression_results = demonstrate_regression_test_suite()
        
        # 6. Architecture comparison
        logger.info("\n" + "=" * 80)
        arch_comparison = demonstrate_architecture_comparison()
        
        # 7. Performance summary and export
        logger.info("\n" + "=" * 80)
        summary = demonstrate_performance_summary_and_export()
        
        # Final summary
        logger.info("\n" + "=" * 80)
        logger.info("🏁 Complete Performance Validation Demo Finished!")
        logger.info("=" * 80)
        
        logger.info("📋 Demo Summary:")
        logger.info(f"   ✅ Basic validation: {basic_result['validation_status']}")
        logger.info(f"   ✅ End-to-end test: {e2e_result.success_rate:.1f}% success rate")
        logger.info(f"   ✅ GPU monitoring: {gpu_result.success_rate:.1f}% success rate")
        logger.info(f"   ✅ Multimodal test: {multimodal_result.success_rate:.1f}% success rate")
        logger.info(f"   ✅ Regression suite: {regression_results['summary']['overall_status']}")
        logger.info(f"   ✅ Architecture comparison: {arch_comparison['improvement_factor']:.1f}x improvement")
        logger.info(f"   ✅ Performance summary: {summary['overall_status']}")
        
        logger.info("\n🎯 Key Achievements:")
        logger.info("   • Comprehensive performance validation system implemented")
        logger.info("   • GPU utilization and memory monitoring working")
        logger.info("   • Multimodal benchmarking capabilities demonstrated")
        logger.info("   • Regression testing suite operational")
        logger.info("   • Architecture comparison functionality verified")
        logger.info("   • Performance summary and export features working")
        
        logger.info("\n📁 Output Files:")
        logger.info("   • demo_performance_results.json - Complete benchmark results")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    """Run the performance validation demo"""
    print("🚀 Performance Validation and Benchmarking Demo")
    print("=" * 80)
    print("This demo showcases the comprehensive performance validation system")
    print("with multimodal support, GPU monitoring, and regression testing.")
    print("=" * 80)
    
    success = run_complete_demo()
    
    if success:
        print("\n✅ Demo completed successfully!")
        print("Check the generated demo_performance_results.json file for detailed results.")
    else:
        print("\n❌ Demo failed. Check the logs for details.")
        sys.exit(1)