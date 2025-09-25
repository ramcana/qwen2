"""
Integration tests for Performance Validation and Benchmarking System
Tests end-to-end performance validation, GPU monitoring, and multimodal benchmarking
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import time
import torch
import sys
import os
import tempfile
import json
import threading
from contextlib import contextmanager

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils.performance_validator import (
    PerformanceValidator,
    BenchmarkResult,
    BenchmarkSuite,
    GPUMonitor,
    create_performance_validator,
    create_default_benchmark_suite,
    validate_performance_improvement
)
from utils.performance_monitor import PerformanceMetrics


class TestBenchmarkResult(unittest.TestCase):
    """Test BenchmarkResult dataclass"""
    
    def test_default_benchmark_result(self):
        """Test default benchmark result values"""
        result = BenchmarkResult()
        
        # Test identification defaults
        self.assertEqual(result.test_name, "")
        self.assertEqual(result.model_name, "")
        self.assertEqual(result.architecture_type, "MMDiT")
        
        # Test performance defaults
        self.assertEqual(result.total_time, 0.0)
        self.assertEqual(result.per_step_time, 0.0)
        self.assertEqual(result.target_time, 5.0)
        self.assertFalse(result.target_met)
        self.assertEqual(result.performance_score, 0.0)
        self.assertEqual(result.improvement_factor, 1.0)
        
        # Test GPU defaults
        self.assertEqual(result.gpu_utilization_percent, 0.0)
        self.assertEqual(result.gpu_memory_used_gb, 0.0)
        self.assertEqual(result.gpu_memory_peak_gb, 0.0)
        
        # Test generation defaults
        self.assertEqual(result.num_steps, 20)
        self.assertEqual(result.resolution, (1024, 1024))
        self.assertEqual(result.batch_size, 1)
        
        # Test multimodal defaults
        self.assertFalse(result.multimodal_enabled)
        self.assertEqual(result.text_processing_time, 0.0)
        self.assertEqual(result.image_analysis_time, 0.0)
        
        # Test quality defaults
        self.assertEqual(result.success_rate, 100.0)
        self.assertEqual(result.error_count, 0)
        self.assertEqual(len(result.warnings), 0)
    
    def test_custom_benchmark_result(self):
        """Test custom benchmark result values"""
        result = BenchmarkResult(
            test_name="custom_test",
            model_name="Qwen-Image",
            total_time=3.5,
            target_met=True,
            performance_score=85.7,
            gpu_utilization_percent=92.5,
            multimodal_enabled=True,
            text_processing_time=0.5,
            resolution=(512, 512),
            num_steps=15
        )
        
        self.assertEqual(result.test_name, "custom_test")
        self.assertEqual(result.model_name, "Qwen-Image")
        self.assertEqual(result.total_time, 3.5)
        self.assertTrue(result.target_met)
        self.assertEqual(result.performance_score, 85.7)
        self.assertEqual(result.gpu_utilization_percent, 92.5)
        self.assertTrue(result.multimodal_enabled)
        self.assertEqual(result.text_processing_time, 0.5)
        self.assertEqual(result.resolution, (512, 512))
        self.assertEqual(result.num_steps, 15)


class TestBenchmarkSuite(unittest.TestCase):
    """Test BenchmarkSuite configuration"""
    
    def test_default_benchmark_suite(self):
        """Test default benchmark suite configuration"""
        suite = BenchmarkSuite()
        
        self.assertEqual(suite.name, "Default Benchmark Suite")
        self.assertEqual(suite.target_improvement_factor, 50.0)
        self.assertEqual(suite.target_generation_time, 5.0)
        
        # Test default configurations
        self.assertIn((512, 512), suite.test_resolutions)
        self.assertIn((1024, 1024), suite.test_resolutions)
        self.assertIn(20, suite.test_step_counts)
        self.assertIn(1, suite.test_batch_sizes)
        
        # Test models configuration
        self.assertTrue(len(suite.models_to_test) >= 2)
        self.assertTrue(suite.include_multimodal_tests)
        self.assertEqual(suite.qwen2vl_model, "Qwen2-VL-7B-Instruct")
    
    def test_custom_benchmark_suite(self):
        """Test custom benchmark suite configuration"""
        suite = BenchmarkSuite(
            name="Custom Test Suite",
            target_improvement_factor=100.0,
            target_generation_time=3.0,
            test_resolutions=[(256, 256), (512, 512)],
            test_step_counts=[10, 15],
            test_batch_sizes=[1],
            include_multimodal_tests=False
        )
        
        self.assertEqual(suite.name, "Custom Test Suite")
        self.assertEqual(suite.target_improvement_factor, 100.0)
        self.assertEqual(suite.target_generation_time, 3.0)
        self.assertEqual(suite.test_resolutions, [(256, 256), (512, 512)])
        self.assertEqual(suite.test_step_counts, [10, 15])
        self.assertEqual(suite.test_batch_sizes, [1])
        self.assertFalse(suite.include_multimodal_tests)


class TestGPUMonitor(unittest.TestCase):
    """Test GPU monitoring functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.monitor = GPUMonitor(sampling_interval=0.01)  # Fast sampling for tests
    
    @patch('torch.cuda.is_available')
    def test_gpu_monitor_initialization_no_gpu(self, mock_cuda_available):
        """Test GPU monitor initialization without GPU"""
        mock_cuda_available.return_value = False
        
        monitor = GPUMonitor()
        self.assertFalse(monitor.gpu_available)
        self.assertEqual(monitor.device_count, 0)
    
    @patch('torch.cuda.is_available')
    @patch('torch.cuda.device_count')
    @patch('torch.cuda.get_device_properties')
    def test_gpu_monitor_initialization_with_gpu(self, mock_get_properties, 
                                               mock_device_count, mock_cuda_available):
        """Test GPU monitor initialization with GPU"""
        mock_cuda_available.return_value = True
        mock_device_count.return_value = 1
        
        # Mock device properties
        mock_properties = Mock()
        mock_properties.total_memory = 16 * 1e9  # 16GB
        mock_get_properties.return_value = mock_properties
        
        monitor = GPUMonitor()
        self.assertTrue(monitor.gpu_available)
        self.assertEqual(monitor.device_count, 1)
    
    @patch('torch.cuda.is_available')
    def test_start_stop_monitoring_no_gpu(self, mock_cuda_available):
        """Test monitoring start/stop without GPU"""
        mock_cuda_available.return_value = False
        
        monitor = GPUMonitor()
        monitor.start_monitoring()
        self.assertFalse(monitor.monitoring)
        
        stats = monitor.stop_monitoring()
        self.assertEqual(stats, {})
    
    @patch('torch.cuda.is_available')
    @patch('torch.cuda.memory_allocated')
    @patch('torch.cuda.memory_reserved')
    def test_monitoring_with_gpu(self, mock_memory_reserved, mock_memory_allocated, 
                                mock_cuda_available):
        """Test GPU monitoring functionality"""
        mock_cuda_available.return_value = True
        mock_memory_allocated.return_value = 2 * 1e9  # 2GB
        mock_memory_reserved.return_value = 3 * 1e9   # 3GB
        
        monitor = GPUMonitor(sampling_interval=0.01)
        monitor.start_monitoring()
        
        # Let it collect some samples
        time.sleep(0.05)
        
        stats = monitor.stop_monitoring()
        
        # Verify stats were collected
        self.assertIn("avg_memory_usage_gb", stats)
        self.assertIn("peak_memory_usage_gb", stats)
        self.assertIn("sample_count", stats)
        self.assertGreater(stats["sample_count"], 0)
    
    def test_monitoring_state_management(self):
        """Test monitoring state management"""
        monitor = GPUMonitor()
        
        # Initially not monitoring
        self.assertFalse(monitor.monitoring)
        
        # Start monitoring (will be disabled if no GPU)
        monitor.start_monitoring()
        
        # Stop monitoring
        stats = monitor.stop_monitoring()
        self.assertFalse(monitor.monitoring)
        
        # Multiple stops should be safe
        stats2 = monitor.stop_monitoring()
        self.assertFalse(monitor.monitoring)


class TestPerformanceValidator(unittest.TestCase):
    """Test PerformanceValidator class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.validator = PerformanceValidator(
            target_improvement_factor=10.0,  # Lower target for testing
            target_generation_time=2.0
        )
    
    def test_validator_initialization(self):
        """Test validator initialization"""
        self.assertEqual(self.validator.target_improvement_factor, 10.0)
        self.assertEqual(self.validator.target_generation_time, 2.0)
        self.assertIsNotNone(self.validator.performance_monitor)
        self.assertIsNotNone(self.validator.gpu_monitor)
        self.assertEqual(len(self.validator.benchmark_results), 0)
        self.assertIsNone(self.validator.baseline_metrics)
        self.assertIsNotNone(self.validator.hardware_info)
    
    def test_hardware_info_collection(self):
        """Test hardware information collection"""
        info = self.validator.hardware_info
        
        # Should always have basic info
        self.assertIn("cpu_count", info)
        self.assertIn("memory_total_gb", info)
        self.assertIn("gpu_available", info)
        
        # GPU-specific info if available
        if torch.cuda.is_available():
            self.assertIn("gpu_name", info)
            self.assertIn("gpu_memory_total_gb", info)
            self.assertIn("cuda_version", info)
    
    def test_set_baseline_metrics(self):
        """Test setting baseline metrics"""
        baseline = PerformanceMetrics(
            total_generation_time=20.0,
            generation_time_per_step=1.0,
            model_name="Baseline-Model"
        )
        
        self.validator.set_baseline_metrics(baseline)
        self.assertEqual(self.validator.baseline_metrics, baseline)
    
    def test_benchmark_generation_context_manager(self):
        """Test benchmark generation context manager"""
        with self.validator.benchmark_generation(
            test_name="test_generation",
            model_name="Test-Model",
            architecture_type="MMDiT",
            num_steps=10,
            resolution=(512, 512)
        ) as (result, perf_monitor):
            
            # Verify result initialization
            self.assertEqual(result.test_name, "test_generation")
            self.assertEqual(result.model_name, "Test-Model")
            self.assertEqual(result.architecture_type, "MMDiT")
            self.assertEqual(result.num_steps, 10)
            self.assertEqual(result.resolution, (512, 512))
            
            # Simulate some work
            time.sleep(0.02)
        
        # Verify result was stored
        self.assertEqual(len(self.validator.benchmark_results), 1)
        stored_result = self.validator.benchmark_results[0]
        self.assertEqual(stored_result.test_name, "test_generation")
        self.assertGreater(stored_result.total_time, 0.01)
    
    def test_benchmark_generation_with_exception(self):
        """Test benchmark generation with exception handling"""
        try:
            with self.validator.benchmark_generation(
                test_name="failing_test",
                model_name="Test-Model"
            ) as (result, perf_monitor):
                
                # Simulate an error
                raise ValueError("Test error")
                
        except ValueError:
            pass  # Expected
        
        # Verify result was still stored with error info
        self.assertEqual(len(self.validator.benchmark_results), 1)
        stored_result = self.validator.benchmark_results[0]
        self.assertEqual(stored_result.test_name, "failing_test")
        self.assertEqual(stored_result.error_count, 1)
        self.assertEqual(stored_result.success_rate, 0.0)
        self.assertTrue(any("Generation error" in warning for warning in stored_result.warnings))
    
    def test_end_to_end_performance_test(self):
        """Test end-to-end performance testing"""
        # Mock generator function
        def mock_generator(prompt):
            time.sleep(0.01)  # Simulate generation time
            return f"generated_image_for_{prompt}"
        
        test_prompts = ["prompt1", "prompt2", "prompt3"]
        
        result = self.validator.run_end_to_end_performance_test(
            generator_function=mock_generator,
            test_prompts=test_prompts,
            model_name="E2E-Test-Model"
        )
        
        # Verify aggregated result
        self.assertEqual(result.test_name, "end_to_end_test")
        self.assertEqual(result.model_name, "E2E-Test-Model")
        self.assertEqual(result.success_rate, 100.0)
        self.assertEqual(result.error_count, 0)
        self.assertGreater(result.total_time, 0.02)  # Should be at least 3 * 0.01s
        
        # Verify individual results were stored
        self.assertEqual(len(self.validator.benchmark_results), 4)  # 3 individual + 1 aggregated
    
    def test_end_to_end_performance_test_with_failures(self):
        """Test end-to-end performance testing with some failures"""
        call_count = 0
        
        def failing_generator(prompt):
            nonlocal call_count
            call_count += 1
            if call_count == 2:  # Fail on second call
                raise RuntimeError("Generation failed")
            time.sleep(0.01)
            return f"generated_image_for_{prompt}"
        
        test_prompts = ["prompt1", "prompt2", "prompt3"]
        
        result = self.validator.run_end_to_end_performance_test(
            generator_function=failing_generator,
            test_prompts=test_prompts,
            model_name="Failing-Test-Model"
        )
        
        # Verify partial success
        self.assertEqual(result.test_name, "end_to_end_test")
        self.assertEqual(result.success_rate, 66.67)  # 2/3 success
        self.assertEqual(result.error_count, 1)
    
    def test_gpu_utilization_benchmark(self):
        """Test GPU utilization benchmark"""
        generation_count = 0
        
        def mock_generator(prompt):
            nonlocal generation_count
            generation_count += 1
            time.sleep(0.005)  # 5ms per generation
            return f"generated_image_{generation_count}"
        
        result = self.validator.run_gpu_utilization_benchmark(
            generator_function=mock_generator,
            test_prompt="test prompt",
            duration_seconds=0.1  # Short duration for testing
        )
        
        # Verify benchmark result
        self.assertEqual(result.test_name, "gpu_utilization_benchmark")
        self.assertEqual(result.model_name, "GPU_Test")
        self.assertGreater(result.total_time, 0.05)  # Should run for at least 0.1s
        self.assertGreater(generation_count, 5)  # Should generate multiple images
        
        # Check for generation rate info in warnings
        self.assertTrue(any("Generations per second" in warning for warning in result.warnings))
        self.assertTrue(any("Total generations" in warning for warning in result.warnings))
    
    def test_speed_improvement_validation(self):
        """Test speed improvement validation"""
        def slow_function(prompt):
            time.sleep(0.1)  # 100ms
            return "slow_result"
        
        def fast_function(prompt):
            time.sleep(0.01)  # 10ms
            return "fast_result"
        
        validation_result = self.validator.run_speed_improvement_validation(
            before_function=slow_function,
            after_function=fast_function,
            test_prompt="test prompt"
        )
        
        # Verify improvement calculation
        self.assertIn("before_metrics", validation_result)
        self.assertIn("after_metrics", validation_result)
        self.assertIn("improvement_factor", validation_result)
        self.assertIn("target_improvement_factor", validation_result)
        self.assertIn("validation_status", validation_result)
        
        # Should show significant improvement
        self.assertGreater(validation_result["improvement_factor"], 5.0)  # At least 5x faster
        self.assertGreater(validation_result["improvement_percent"], 400.0)  # At least 400% improvement
        self.assertGreater(validation_result["time_saved_seconds"], 0.05)  # At least 50ms saved
        
        # Check summary
        summary = validation_result["summary"]
        self.assertIn("before_time", summary)
        self.assertIn("after_time", summary)
        self.assertIn("speedup", summary)
        self.assertIn("target_achieved", summary)
    
    def test_speed_improvement_validation_with_failures(self):
        """Test speed improvement validation with function failures"""
        def failing_before_function(prompt):
            raise RuntimeError("Before function failed")
        
        def working_after_function(prompt):
            time.sleep(0.01)
            return "after_result"
        
        validation_result = self.validator.run_speed_improvement_validation(
            before_function=failing_before_function,
            after_function=working_after_function,
            test_prompt="test prompt"
        )
        
        # Verify failure handling
        before_metrics = validation_result["before_metrics"]
        self.assertEqual(before_metrics["error_count"], 1)
        self.assertEqual(before_metrics["success_rate"], 0.0)
        
        after_metrics = validation_result["after_metrics"]
        self.assertEqual(after_metrics["error_count"], 0)
        self.assertEqual(after_metrics["success_rate"], 100.0)
    
    def test_multimodal_performance_benchmark(self):
        """Test multimodal performance benchmark"""
        def mock_qwen2vl_function(text_prompt, image_path=None):
            time.sleep(0.01)  # Simulate processing time
            if image_path:
                return f"multimodal_result_for_{text_prompt}_with_{image_path}"
            else:
                return f"text_result_for_{text_prompt}"
        
        text_prompts = ["prompt1", "prompt2", "prompt3"]
        image_paths = ["image1.jpg", "image2.jpg"]  # Fewer images than prompts
        
        result = self.validator.run_multimodal_performance_benchmark(
            qwen2vl_function=mock_qwen2vl_function,
            text_prompts=text_prompts,
            image_paths=image_paths
        )
        
        # Verify multimodal result
        self.assertEqual(result.test_name, "multimodal_benchmark")
        self.assertEqual(result.model_name, "Qwen2-VL-7B-Instruct")
        self.assertEqual(result.architecture_type, "Multimodal")
        self.assertTrue(result.multimodal_enabled)
        self.assertEqual(result.success_rate, 100.0)
        self.assertGreater(result.text_processing_time, 0.0)
        self.assertGreater(result.image_analysis_time, 0.0)
    
    def test_regression_test_suite(self):
        """Test regression test suite execution"""
        def mock_generator(prompt):
            time.sleep(0.005)  # 5ms per generation
            return "generated_image"
        
        # Create a minimal test suite
        suite = BenchmarkSuite(
            name="Test Regression Suite",
            test_resolutions=[(256, 256), (512, 512)],
            test_step_counts=[5, 10],
            test_batch_sizes=[1]
        )
        
        results = self.validator.run_regression_test_suite(
            generator_function=mock_generator,
            benchmark_suite=suite
        )
        
        # Verify suite results structure
        self.assertEqual(results["suite_name"], "Test Regression Suite")
        self.assertIn("start_time", results)
        self.assertIn("test_results", results)
        self.assertIn("summary", results)
        self.assertIn("hardware_info", results)
        
        # Should have 4 tests (2 resolutions * 2 step counts * 1 batch size)
        self.assertEqual(len(results["test_results"]), 4)
        
        # Verify summary
        summary = results["summary"]
        self.assertEqual(summary["total_tests"], 4)
        self.assertIn("passed_tests", summary)
        self.assertIn("failed_tests", summary)
        self.assertIn("pass_rate", summary)
        self.assertIn("overall_status", summary)
    
    def test_export_benchmark_results(self):
        """Test benchmark results export"""
        # Add some test results
        with self.validator.benchmark_generation(
            test_name="export_test",
            model_name="Export-Model"
        ) as (result, perf_monitor):
            time.sleep(0.01)
        
        # Export to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            self.validator.export_benchmark_results(temp_path)
            
            # Verify file was created and contains expected data
            with open(temp_path, 'r') as f:
                data = json.load(f)
            
            self.assertIn("export_timestamp", data)
            self.assertIn("hardware_info", data)
            self.assertIn("target_improvement_factor", data)
            self.assertIn("target_generation_time", data)
            self.assertIn("benchmark_results", data)
            self.assertIn("summary", data)
            
            # Verify benchmark results
            self.assertEqual(len(data["benchmark_results"]), 1)
            self.assertEqual(data["benchmark_results"][0]["test_name"], "export_test")
            
            # Verify summary
            summary = data["summary"]
            self.assertEqual(summary["total_benchmarks"], 1)
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_performance_summary(self):
        """Test performance summary generation"""
        # Add some test results with different performance levels
        with self.validator.benchmark_generation(
            test_name="good_performance",
            model_name="Good-Model"
        ) as (result, perf_monitor):
            # Simulate fast generation
            time.sleep(0.01)
            result.performance_score = 95.0
            result.target_met = True
        
        with self.validator.benchmark_generation(
            test_name="poor_performance",
            model_name="Poor-Model"
        ) as (result, perf_monitor):
            # Simulate slow generation
            time.sleep(0.05)
            result.performance_score = 30.0
            result.target_met = False
        
        summary = self.validator.get_performance_summary()
        
        # Verify summary structure
        self.assertIn("overall_status", summary)
        self.assertIn("test_statistics", summary)
        self.assertIn("performance_targets", summary)
        self.assertIn("best_performance", summary)
        self.assertIn("worst_performance", summary)
        self.assertIn("hardware_info", summary)
        self.assertIn("recommendations", summary)
        
        # Verify statistics
        stats = summary["test_statistics"]
        self.assertEqual(stats["total_tests"], 2)
        self.assertEqual(stats["passed_tests"], 1)
        self.assertEqual(stats["failed_tests"], 1)
        self.assertEqual(stats["pass_rate"], 50.0)
        
        # Verify best/worst identification
        self.assertEqual(summary["best_performance"]["test_name"], "good_performance")
        self.assertEqual(summary["worst_performance"]["test_name"], "poor_performance")
    
    def test_performance_summary_no_results(self):
        """Test performance summary with no results"""
        summary = self.validator.get_performance_summary()
        self.assertEqual(summary, {"error": "No benchmark results available"})
    
    def test_clear_results(self):
        """Test clearing benchmark results"""
        # Add some results
        with self.validator.benchmark_generation(
            test_name="clear_test",
            model_name="Clear-Model"
        ) as (result, perf_monitor):
            time.sleep(0.01)
        
        # Set baseline
        baseline = PerformanceMetrics(total_generation_time=10.0)
        self.validator.set_baseline_metrics(baseline)
        
        # Verify results exist
        self.assertEqual(len(self.validator.benchmark_results), 1)
        self.assertIsNotNone(self.validator.baseline_metrics)
        
        # Clear results
        self.validator.clear_results()
        
        # Verify results are cleared
        self.assertEqual(len(self.validator.benchmark_results), 0)
        self.assertIsNone(self.validator.baseline_metrics)


class TestUtilityFunctions(unittest.TestCase):
    """Test utility functions"""
    
    def test_create_performance_validator(self):
        """Test performance validator creation utility"""
        validator = create_performance_validator(target_improvement=25.0, target_time=3.0)
        
        self.assertIsInstance(validator, PerformanceValidator)
        self.assertEqual(validator.target_improvement_factor, 25.0)
        self.assertEqual(validator.target_generation_time, 3.0)
    
    def test_create_default_benchmark_suite(self):
        """Test default benchmark suite creation"""
        suite = create_default_benchmark_suite()
        
        self.assertIsInstance(suite, BenchmarkSuite)
        self.assertEqual(suite.name, "Default MMDiT Performance Benchmark")
        self.assertEqual(suite.target_improvement_factor, 50.0)
        self.assertEqual(suite.target_generation_time, 5.0)
    
    def test_validate_performance_improvement_context_manager(self):
        """Test performance improvement validation context manager"""
        def slow_function(prompt):
            time.sleep(0.05)
            return "slow_result"
        
        def fast_function(prompt):
            time.sleep(0.01)
            return "fast_result"
        
        with validate_performance_improvement(
            before_function=slow_function,
            after_function=fast_function,
            test_prompt="test prompt",
            target_improvement=3.0  # Lower target for testing
        ) as results:
            
            # Verify results structure
            self.assertIn("improvement_factor", results)
            self.assertIn("target_improvement_factor", results)
            self.assertIn("validation_status", results)
            self.assertIn("summary", results)
            
            # Should show improvement
            self.assertGreater(results["improvement_factor"], 3.0)
            self.assertEqual(results["target_improvement_factor"], 3.0)


class TestIntegrationScenarios(unittest.TestCase):
    """Test realistic integration scenarios"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.validator = PerformanceValidator(
            target_improvement_factor=5.0,  # Realistic target for testing
            target_generation_time=1.0      # Fast target for testing
        )
    
    def test_mmdit_vs_unet_comparison_scenario(self):
        """Test MMDiT vs UNet architecture comparison scenario"""
        # Simulate UNet (slower) generation
        def unet_generator(prompt):
            time.sleep(0.08)  # 80ms - slower
            return "unet_generated_image"
        
        # Simulate MMDiT (faster) generation
        def mmdit_generator(prompt):
            time.sleep(0.02)  # 20ms - faster
            return "mmdit_generated_image"
        
        # Test UNet performance
        with self.validator.benchmark_generation(
            test_name="unet_test",
            model_name="UNet-Model",
            architecture_type="UNet",
            num_steps=20
        ) as (unet_result, _):
            unet_image = unet_generator("test prompt")
            unet_result.success_rate = 100.0 if unet_image else 0.0
        
        # Test MMDiT performance
        with self.validator.benchmark_generation(
            test_name="mmdit_test",
            model_name="MMDiT-Model",
            architecture_type="MMDiT",
            num_steps=20
        ) as (mmdit_result, _):
            mmdit_image = mmdit_generator("test prompt")
            mmdit_result.success_rate = 100.0 if mmdit_image else 0.0
        
        # Verify both tests completed
        self.assertEqual(len(self.validator.benchmark_results), 2)
        
        # Find results
        unet_result = next(r for r in self.validator.benchmark_results if r.architecture_type == "UNet")
        mmdit_result = next(r for r in self.validator.benchmark_results if r.architecture_type == "MMDiT")
        
        # Verify MMDiT is faster
        self.assertLess(mmdit_result.total_time, unet_result.total_time)
        self.assertGreater(mmdit_result.performance_score, unet_result.performance_score)
    
    def test_optimization_workflow_scenario(self):
        """Test complete optimization workflow scenario"""
        # Step 1: Establish baseline with slow implementation
        def slow_implementation(prompt):
            time.sleep(0.1)  # 100ms - very slow
            return "slow_generated_image"
        
        baseline_metrics = PerformanceMetrics(
            total_generation_time=0.1,
            generation_time_per_step=0.005,
            model_name="Slow-Implementation",
            target_met=False
        )
        self.validator.set_baseline_metrics(baseline_metrics)
        
        # Step 2: Test optimized implementation
        def optimized_implementation(prompt):
            time.sleep(0.015)  # 15ms - much faster
            return "optimized_generated_image"
        
        # Step 3: Run comprehensive validation
        validation_result = self.validator.run_speed_improvement_validation(
            before_function=slow_implementation,
            after_function=optimized_implementation,
            test_prompt="optimization test"
        )
        
        # Step 4: Verify optimization success
        self.assertGreater(validation_result["improvement_factor"], 5.0)
        self.assertEqual(validation_result["validation_status"], "PASSED")
        self.assertTrue(validation_result["summary"]["target_achieved"])
        
        # Step 5: Run end-to-end validation
        test_prompts = ["prompt1", "prompt2", "prompt3"]
        e2e_result = self.validator.run_end_to_end_performance_test(
            generator_function=optimized_implementation,
            test_prompts=test_prompts,
            model_name="Optimized-Model"
        )
        
        # Verify end-to-end success
        self.assertEqual(e2e_result.success_rate, 100.0)
        self.assertEqual(e2e_result.error_count, 0)
        self.assertLess(e2e_result.total_time, 0.1)  # Should be faster than baseline
    
    def test_regression_prevention_scenario(self):
        """Test regression prevention scenario"""
        # Simulate a series of implementations with varying performance
        implementations = [
            ("baseline", lambda p: (time.sleep(0.05), "baseline")[1]),      # 50ms
            ("optimized_v1", lambda p: (time.sleep(0.02), "v1")[1]),        # 20ms - good
            ("optimized_v2", lambda p: (time.sleep(0.08), "v2")[1]),        # 80ms - regression!
            ("optimized_v3", lambda p: (time.sleep(0.015), "v3")[1]),       # 15ms - best
        ]
        
        results = []
        for name, impl in implementations:
            with self.validator.benchmark_generation(
                test_name=f"regression_test_{name}",
                model_name=name,
                architecture_type="MMDiT"
            ) as (result, _):
                generated = impl("test prompt")
                result.success_rate = 100.0 if generated else 0.0
                results.append(result)
        
        # Verify we can detect the regression in v2
        v1_result = next(r for r in self.validator.benchmark_results if "v1" in r.model_name)
        v2_result = next(r for r in self.validator.benchmark_results if "v2" in r.model_name)
        v3_result = next(r for r in self.validator.benchmark_results if "v3" in r.model_name)
        
        # v2 should be slower than v1 (regression)
        self.assertGreater(v2_result.total_time, v1_result.total_time)
        self.assertLess(v2_result.performance_score, v1_result.performance_score)
        
        # v3 should be the fastest
        self.assertLess(v3_result.total_time, v1_result.total_time)
        self.assertLess(v3_result.total_time, v2_result.total_time)
        self.assertGreater(v3_result.performance_score, v1_result.performance_score)


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)