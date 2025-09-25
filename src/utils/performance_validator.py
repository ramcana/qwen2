"""
Performance Validation and Benchmarking System with Multimodal Support
Provides comprehensive end-to-end performance testing, GPU monitoring, and benchmarking
for MMDiT architecture and Qwen2-VL integration
"""

import time
import logging
import psutil
import torch
import threading
import json
import os
import gc
from typing import Dict, Any, List, Optional, Tuple, Callable
from dataclasses import dataclass, field, asdict
from contextlib import contextmanager
from datetime import datetime
import numpy as np
from pathlib import Path

# Import existing performance monitor
from .performance_monitor import PerformanceMonitor, PerformanceMetrics

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Data class for storing benchmark results"""
    # Test identification
    test_name: str = ""
    model_name: str = ""
    architecture_type: str = "MMDiT"
    test_timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    # Performance metrics
    total_time: float = 0.0
    per_step_time: float = 0.0
    target_time: float = 5.0
    target_met: bool = False
    performance_score: float = 0.0
    improvement_factor: float = 1.0
    
    # GPU metrics
    gpu_utilization_percent: float = 0.0
    gpu_memory_used_gb: float = 0.0
    gpu_memory_peak_gb: float = 0.0
    gpu_memory_efficiency: float = 0.0
    
    # Generation parameters
    num_steps: int = 20
    resolution: Tuple[int, int] = (1024, 1024)
    batch_size: int = 1
    
    # Multimodal metrics (for Qwen2-VL)
    multimodal_enabled: bool = False
    text_processing_time: float = 0.0
    image_analysis_time: float = 0.0
    
    # Quality metrics
    success_rate: float = 100.0
    error_count: int = 0
    warnings: List[str] = field(default_factory=list)
    
    # Hardware info
    gpu_name: str = ""
    driver_version: str = ""
    cuda_version: str = ""


@dataclass
class BenchmarkSuite:
    """Configuration for a complete benchmark suite"""
    name: str = "Default Benchmark Suite"
    target_improvement_factor: float = 50.0  # 50x improvement target
    target_generation_time: float = 5.0
    
    # Test configurations
    test_resolutions: List[Tuple[int, int]] = field(default_factory=lambda: [
        (512, 512), (768, 768), (1024, 1024), (1280, 1280)
    ])
    test_step_counts: List[int] = field(default_factory=lambda: [10, 20, 30])
    test_batch_sizes: List[int] = field(default_factory=lambda: [1, 2])
    
    # Models to test
    models_to_test: List[Dict[str, str]] = field(default_factory=lambda: [
        {"name": "Qwen-Image", "architecture": "MMDiT"},
        {"name": "Qwen-Image-Edit", "architecture": "MMDiT"},
    ])
    
    # Multimodal tests
    include_multimodal_tests: bool = True
    qwen2vl_model: str = "Qwen2-VL-7B-Instruct"


class GPUMonitor:
    """Real-time GPU utilization and memory monitoring"""
    
    def __init__(self, sampling_interval: float = 0.1):
        """
        Initialize GPU monitor
        
        Args:
            sampling_interval: Sampling interval in seconds
        """
        self.sampling_interval = sampling_interval
        self.monitoring = False
        self.monitor_thread = None
        
        # Metrics storage
        self.gpu_utilization_samples = []
        self.memory_usage_samples = []
        self.timestamps = []
        
        # Peak tracking
        self.peak_memory_usage = 0.0
        self.peak_utilization = 0.0
        
        # GPU availability check
        self.gpu_available = torch.cuda.is_available()
        if self.gpu_available:
            self.device_count = torch.cuda.device_count()
            self.device_properties = torch.cuda.get_device_properties(0)
        
        # Try to import pynvml for detailed GPU monitoring
        self.pynvml_available = False
        try:
            import pynvml
            pynvml.nvmlInit()
            self.nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self.pynvml_available = True
            logger.info("NVIDIA ML library available for detailed GPU monitoring")
        except ImportError:
            logger.warning("pynvml not available - limited GPU monitoring")
    
    def start_monitoring(self):
        """Start GPU monitoring in background thread"""
        if not self.gpu_available:
            logger.warning("GPU not available - monitoring disabled")
            return
        
        if self.monitoring:
            logger.warning("GPU monitoring already running")
            return
        
        self.monitoring = True
        self.gpu_utilization_samples.clear()
        self.memory_usage_samples.clear()
        self.timestamps.clear()
        self.peak_memory_usage = 0.0
        self.peak_utilization = 0.0
        
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        
        logger.info(f"GPU monitoring started (sampling every {self.sampling_interval}s)")
    
    def stop_monitoring(self) -> Dict[str, float]:
        """
        Stop GPU monitoring and return summary statistics
        
        Returns:
            Dictionary with monitoring statistics
        """
        if not self.monitoring:
            return {}
        
        self.monitoring = False
        
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=1.0)
        
        # Calculate statistics
        stats = self._calculate_monitoring_stats()
        
        logger.info(f"GPU monitoring stopped - collected {len(self.gpu_utilization_samples)} samples")
        return stats
    
    def _monitor_loop(self):
        """Background monitoring loop"""
        while self.monitoring:
            try:
                timestamp = time.time()
                
                # Get PyTorch memory info
                memory_allocated = torch.cuda.memory_allocated() / 1e9  # GB
                memory_reserved = torch.cuda.memory_reserved() / 1e9    # GB
                
                # Get detailed GPU info if available
                gpu_utilization = 0.0
                if self.pynvml_available:
                    try:
                        import pynvml
                        util_info = pynvml.nvmlDeviceGetUtilizationRates(self.nvml_handle)
                        gpu_utilization = util_info.gpu
                    except Exception as e:
                        logger.debug(f"Failed to get GPU utilization: {e}")
                
                # Store samples
                self.timestamps.append(timestamp)
                self.memory_usage_samples.append(memory_allocated)
                self.gpu_utilization_samples.append(gpu_utilization)
                
                # Update peaks
                self.peak_memory_usage = max(self.peak_memory_usage, memory_allocated)
                self.peak_utilization = max(self.peak_utilization, gpu_utilization)
                
                time.sleep(self.sampling_interval)
                
            except Exception as e:
                logger.error(f"GPU monitoring error: {e}")
                time.sleep(self.sampling_interval)
    
    def _calculate_monitoring_stats(self) -> Dict[str, float]:
        """Calculate monitoring statistics"""
        if not self.memory_usage_samples:
            return {}
        
        memory_samples = np.array(self.memory_usage_samples)
        utilization_samples = np.array(self.gpu_utilization_samples)
        
        stats = {
            "avg_memory_usage_gb": float(np.mean(memory_samples)),
            "peak_memory_usage_gb": float(self.peak_memory_usage),
            "min_memory_usage_gb": float(np.min(memory_samples)),
            "avg_gpu_utilization_percent": float(np.mean(utilization_samples)),
            "peak_gpu_utilization_percent": float(self.peak_utilization),
            "min_gpu_utilization_percent": float(np.min(utilization_samples)),
            "memory_efficiency": float(np.mean(memory_samples) / self.device_properties.total_memory * 1e9 * 100),
            "sample_count": len(memory_samples),
            "monitoring_duration": float(self.timestamps[-1] - self.timestamps[0]) if len(self.timestamps) > 1 else 0.0
        }
        
        return stats


class PerformanceValidator:
    """
    Comprehensive performance validation and benchmarking system
    Supports MMDiT architecture and multimodal capabilities
    """
    
    def __init__(self, target_improvement_factor: float = 50.0, target_generation_time: float = 5.0):
        """
        Initialize performance validator
        
        Args:
            target_improvement_factor: Target improvement factor (e.g., 50x faster)
            target_generation_time: Target generation time in seconds
        """
        self.target_improvement_factor = target_improvement_factor
        self.target_generation_time = target_generation_time
        
        # Initialize components
        self.performance_monitor = PerformanceMonitor(target_generation_time)
        self.gpu_monitor = GPUMonitor()
        
        # Results storage
        self.benchmark_results: List[BenchmarkResult] = []
        self.baseline_metrics: Optional[PerformanceMetrics] = None
        
        # Hardware info
        self.hardware_info = self._collect_hardware_info()
        
        logger.info(f"PerformanceValidator initialized - target: {target_improvement_factor}x improvement, {target_generation_time}s generation time")
    
    def _collect_hardware_info(self) -> Dict[str, Any]:
        """Collect hardware information"""
        info = {
            "cpu_count": psutil.cpu_count(),
            "memory_total_gb": psutil.virtual_memory().total / 1e9,
            "gpu_available": torch.cuda.is_available(),
        }
        
        if torch.cuda.is_available():
            info.update({
                "gpu_name": torch.cuda.get_device_name(),
                "gpu_memory_total_gb": torch.cuda.get_device_properties(0).total_memory / 1e9,
                "cuda_version": torch.version.cuda,
                "pytorch_version": torch.__version__,
            })
            
            try:
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                info["driver_version"] = pynvml.nvmlSystemGetDriverVersion().decode()
            except ImportError:
                info["driver_version"] = "unknown"
        
        return info
    
    @contextmanager
    def benchmark_generation(self, test_name: str, model_name: str = "", 
                           architecture_type: str = "MMDiT", num_steps: int = 20,
                           resolution: Tuple[int, int] = (1024, 1024),
                           multimodal_enabled: bool = False):
        """
        Context manager for benchmarking generation performance
        
        Args:
            test_name: Name of the test
            model_name: Model being tested
            architecture_type: Architecture type (MMDiT, UNet)
            num_steps: Number of inference steps
            resolution: Image resolution
            multimodal_enabled: Whether multimodal features are enabled
        """
        # Initialize benchmark result
        result = BenchmarkResult(
            test_name=test_name,
            model_name=model_name,
            architecture_type=architecture_type,
            num_steps=num_steps,
            resolution=resolution,
            target_time=self.target_generation_time,
            multimodal_enabled=multimodal_enabled,
            gpu_name=self.hardware_info.get("gpu_name", ""),
            driver_version=self.hardware_info.get("driver_version", ""),
            cuda_version=self.hardware_info.get("cuda_version", "")
        )
        
        # Start monitoring
        self.gpu_monitor.start_monitoring()
        
        # Start performance monitoring
        with self.performance_monitor.monitor_generation(
            model_name=model_name,
            architecture_type=architecture_type,
            num_steps=num_steps,
            resolution=resolution
        ) as perf_monitor:
            
            try:
                yield result, perf_monitor
                
            except Exception as e:
                result.error_count += 1
                result.warnings.append(f"Generation error: {str(e)}")
                result.success_rate = 0.0
                logger.error(f"Benchmark generation failed: {e}")
                raise
                
            finally:
                # Stop monitoring and collect results
                gpu_stats = self.gpu_monitor.stop_monitoring()
                
                # Update result with performance metrics
                metrics = perf_monitor.get_current_metrics()
                result.total_time = metrics.total_generation_time
                result.per_step_time = metrics.generation_time_per_step
                result.target_met = metrics.target_met
                result.performance_score = metrics.performance_score
                
                # Update with GPU monitoring results
                if gpu_stats:
                    result.gpu_utilization_percent = gpu_stats.get("avg_gpu_utilization_percent", 0.0)
                    result.gpu_memory_used_gb = gpu_stats.get("avg_memory_usage_gb", 0.0)
                    result.gpu_memory_peak_gb = gpu_stats.get("peak_memory_usage_gb", 0.0)
                    result.gpu_memory_efficiency = gpu_stats.get("memory_efficiency", 0.0)
                
                # Calculate improvement factor if baseline exists
                if self.baseline_metrics:
                    if result.total_time > 0:
                        result.improvement_factor = self.baseline_metrics.total_generation_time / result.total_time
                
                # Store result
                self.benchmark_results.append(result)
                
                # Log result
                self._log_benchmark_result(result)
    
    def set_baseline_metrics(self, metrics: PerformanceMetrics):
        """
        Set baseline metrics for improvement calculations
        
        Args:
            metrics: Baseline performance metrics
        """
        self.baseline_metrics = metrics
        logger.info(f"Baseline metrics set - {metrics.total_generation_time:.3f}s generation time")
    
    def run_end_to_end_performance_test(self, generator_function: Callable, 
                                      test_prompts: List[str],
                                      model_name: str = "Qwen-Image") -> BenchmarkResult:
        """
        Run comprehensive end-to-end performance test
        
        Args:
            generator_function: Function that generates images
            test_prompts: List of prompts to test
            model_name: Name of the model being tested
            
        Returns:
            Aggregated benchmark result
        """
        logger.info(f"🚀 Starting end-to-end performance test with {len(test_prompts)} prompts")
        
        individual_results = []
        total_errors = 0
        
        for i, prompt in enumerate(test_prompts):
            test_name = f"e2e_test_{i+1}"
            
            try:
                with self.benchmark_generation(
                    test_name=test_name,
                    model_name=model_name,
                    architecture_type="MMDiT",
                    num_steps=20,
                    resolution=(1024, 1024)
                ) as (result, perf_monitor):
                    
                    # Run generation
                    start_time = time.time()
                    generated_image = generator_function(prompt)
                    end_time = time.time()
                    
                    # Verify generation succeeded
                    if generated_image is None:
                        raise ValueError("Generation returned None")
                    
                    result.success_rate = 100.0
                    individual_results.append(result)
                    
                    logger.info(f"✅ Test {i+1}/{len(test_prompts)} completed in {end_time - start_time:.3f}s")
                    
            except Exception as e:
                total_errors += 1
                logger.error(f"❌ Test {i+1}/{len(test_prompts)} failed: {e}")
        
        # Calculate aggregated results
        if individual_results:
            aggregated_result = self._aggregate_benchmark_results(individual_results, "end_to_end_test")
            aggregated_result.error_count = total_errors
            aggregated_result.success_rate = (len(individual_results) / len(test_prompts)) * 100
            
            logger.info(f"🏁 End-to-end test completed - {len(individual_results)}/{len(test_prompts)} successful")
            return aggregated_result
        else:
            # All tests failed
            failed_result = BenchmarkResult(
                test_name="end_to_end_test",
                model_name=model_name,
                error_count=total_errors,
                success_rate=0.0
            )
            return failed_result
    
    def run_gpu_utilization_benchmark(self, generator_function: Callable,
                                    test_prompt: str = "A beautiful landscape",
                                    duration_seconds: float = 30.0) -> BenchmarkResult:
        """
        Run GPU utilization and memory usage benchmark
        
        Args:
            generator_function: Function that generates images
            test_prompt: Prompt to use for testing
            duration_seconds: Duration to run the test
            
        Returns:
            Benchmark result with GPU metrics
        """
        logger.info(f"🔥 Starting GPU utilization benchmark for {duration_seconds}s")
        
        with self.benchmark_generation(
            test_name="gpu_utilization_benchmark",
            model_name="GPU_Test",
            architecture_type="MMDiT",
            num_steps=20,
            resolution=(1024, 1024)
        ) as (result, perf_monitor):
            
            start_time = time.time()
            generation_count = 0
            
            while time.time() - start_time < duration_seconds:
                try:
                    generated_image = generator_function(test_prompt)
                    generation_count += 1
                    
                    # Brief pause to allow monitoring
                    time.sleep(0.1)
                    
                except Exception as e:
                    result.error_count += 1
                    result.warnings.append(f"Generation {generation_count + 1} failed: {str(e)}")
            
            actual_duration = time.time() - start_time
            result.success_rate = (generation_count / max(1, generation_count + result.error_count)) * 100
            
            logger.info(f"🏁 GPU benchmark completed - {generation_count} generations in {actual_duration:.1f}s")
            
            # Add custom metrics
            result.warnings.append(f"Generations per second: {generation_count / actual_duration:.2f}")
            result.warnings.append(f"Total generations: {generation_count}")
            
            return result
    
    def run_speed_improvement_validation(self, before_function: Callable, after_function: Callable,
                                       test_prompt: str = "A beautiful landscape") -> Dict[str, Any]:
        """
        Validate speed improvement between two implementations
        
        Args:
            before_function: Original implementation
            after_function: Optimized implementation
            test_prompt: Test prompt
            
        Returns:
            Improvement validation results
        """
        logger.info("⚡ Running speed improvement validation")
        
        # Test "before" implementation
        logger.info("Testing original implementation...")
        with self.benchmark_generation(
            test_name="before_optimization",
            model_name="Original",
            architecture_type="MMDiT"
        ) as (before_result, _):
            try:
                before_image = before_function(test_prompt)
                before_result.success_rate = 100.0 if before_image is not None else 0.0
            except Exception as e:
                before_result.error_count = 1
                before_result.success_rate = 0.0
                before_result.warnings.append(f"Before test failed: {str(e)}")
        
        # Test "after" implementation
        logger.info("Testing optimized implementation...")
        with self.benchmark_generation(
            test_name="after_optimization",
            model_name="Optimized",
            architecture_type="MMDiT"
        ) as (after_result, _):
            try:
                after_image = after_function(test_prompt)
                after_result.success_rate = 100.0 if after_image is not None else 0.0
            except Exception as e:
                after_result.error_count = 1
                after_result.success_rate = 0.0
                after_result.warnings.append(f"After test failed: {str(e)}")
        
        # Calculate improvement
        improvement_factor = 1.0
        if after_result.total_time > 0 and before_result.total_time > 0:
            improvement_factor = before_result.total_time / after_result.total_time
        
        # Validate against target
        target_met = improvement_factor >= self.target_improvement_factor
        
        validation_result = {
            "before_metrics": asdict(before_result),
            "after_metrics": asdict(after_result),
            "improvement_factor": improvement_factor,
            "target_improvement_factor": self.target_improvement_factor,
            "target_met": target_met,
            "improvement_percent": (improvement_factor - 1) * 100,
            "time_saved_seconds": before_result.total_time - after_result.total_time,
            "validation_status": "PASSED" if target_met else "FAILED",
            "summary": {
                "before_time": before_result.total_time,
                "after_time": after_result.total_time,
                "speedup": f"{improvement_factor:.1f}x faster",
                "target_achieved": target_met
            }
        }
        
        # Log results
        logger.info(f"📊 Speed Improvement Validation Results:")
        logger.info(f"   Before: {before_result.total_time:.3f}s")
        logger.info(f"   After: {after_result.total_time:.3f}s")
        logger.info(f"   Improvement: {improvement_factor:.1f}x faster")
        logger.info(f"   Target: {self.target_improvement_factor:.1f}x")
        logger.info(f"   Status: {'✅ PASSED' if target_met else '❌ FAILED'}")
        
        return validation_result
    
    def run_multimodal_performance_benchmark(self, qwen2vl_function: Callable,
                                           text_prompts: List[str],
                                           image_paths: List[str] = None) -> BenchmarkResult:
        """
        Run performance benchmark for multimodal (Qwen2-VL) capabilities
        
        Args:
            qwen2vl_function: Function that processes text and images
            text_prompts: List of text prompts
            image_paths: Optional list of image paths for analysis
            
        Returns:
            Multimodal benchmark result
        """
        logger.info(f"🎭 Starting multimodal performance benchmark with {len(text_prompts)} prompts")
        
        with self.benchmark_generation(
            test_name="multimodal_benchmark",
            model_name="Qwen2-VL-7B-Instruct",
            architecture_type="Multimodal",
            multimodal_enabled=True
        ) as (result, perf_monitor):
            
            total_text_time = 0.0
            total_image_time = 0.0
            successful_processes = 0
            
            for i, prompt in enumerate(text_prompts):
                try:
                    # Time text processing
                    text_start = time.time()
                    
                    # Process with or without image
                    if image_paths and i < len(image_paths):
                        processed_result = qwen2vl_function(prompt, image_paths[i])
                        image_time = time.time() - text_start
                        total_image_time += image_time
                    else:
                        processed_result = qwen2vl_function(prompt)
                        text_time = time.time() - text_start
                        total_text_time += text_time
                    
                    if processed_result is not None:
                        successful_processes += 1
                    
                except Exception as e:
                    result.error_count += 1
                    result.warnings.append(f"Multimodal processing {i+1} failed: {str(e)}")
            
            # Update multimodal-specific metrics
            result.text_processing_time = total_text_time / max(1, len(text_prompts))
            result.image_analysis_time = total_image_time / max(1, len(image_paths) if image_paths else 1)
            result.success_rate = (successful_processes / len(text_prompts)) * 100
            
            logger.info(f"🏁 Multimodal benchmark completed - {successful_processes}/{len(text_prompts)} successful")
            
            return result
    
    def run_regression_test_suite(self, generator_function: Callable,
                                benchmark_suite: BenchmarkSuite) -> Dict[str, Any]:
        """
        Run comprehensive regression test suite
        
        Args:
            generator_function: Function that generates images
            benchmark_suite: Configuration for the test suite
            
        Returns:
            Complete regression test results
        """
        logger.info(f"🧪 Starting regression test suite: {benchmark_suite.name}")
        
        suite_results = {
            "suite_name": benchmark_suite.name,
            "start_time": datetime.now().isoformat(),
            "target_improvement_factor": benchmark_suite.target_improvement_factor,
            "target_generation_time": benchmark_suite.target_generation_time,
            "test_results": [],
            "summary": {},
            "hardware_info": self.hardware_info
        }
        
        test_count = 0
        passed_tests = 0
        
        # Test different resolutions
        for resolution in benchmark_suite.test_resolutions:
            for steps in benchmark_suite.test_step_counts:
                for batch_size in benchmark_suite.test_batch_sizes:
                    test_name = f"regression_test_{resolution[0]}x{resolution[1]}_{steps}steps_batch{batch_size}"
                    test_count += 1
                    
                    try:
                        with self.benchmark_generation(
                            test_name=test_name,
                            model_name="Regression_Test",
                            architecture_type="MMDiT",
                            num_steps=steps,
                            resolution=resolution
                        ) as (result, perf_monitor):
                            
                            # Generate test image
                            test_prompt = f"A test image at {resolution[0]}x{resolution[1]} resolution"
                            generated_image = generator_function(test_prompt)
                            
                            if generated_image is not None:
                                result.success_rate = 100.0
                                if result.target_met:
                                    passed_tests += 1
                            else:
                                result.success_rate = 0.0
                                result.error_count = 1
                            
                            suite_results["test_results"].append(asdict(result))
                            
                    except Exception as e:
                        logger.error(f"Regression test {test_name} failed: {e}")
                        failed_result = BenchmarkResult(
                            test_name=test_name,
                            error_count=1,
                            success_rate=0.0,
                            warnings=[f"Test failed: {str(e)}"]
                        )
                        suite_results["test_results"].append(asdict(failed_result))
        
        # Calculate suite summary
        suite_results["summary"] = {
            "total_tests": test_count,
            "passed_tests": passed_tests,
            "failed_tests": test_count - passed_tests,
            "pass_rate": (passed_tests / test_count) * 100 if test_count > 0 else 0,
            "overall_status": "PASSED" if passed_tests == test_count else "FAILED",
            "end_time": datetime.now().isoformat()
        }
        
        logger.info(f"🏁 Regression test suite completed - {passed_tests}/{test_count} tests passed")
        
        return suite_results
    
    def _aggregate_benchmark_results(self, results: List[BenchmarkResult], 
                                   aggregated_name: str) -> BenchmarkResult:
        """
        Aggregate multiple benchmark results into a single result
        
        Args:
            results: List of benchmark results to aggregate
            aggregated_name: Name for the aggregated result
            
        Returns:
            Aggregated benchmark result
        """
        if not results:
            return BenchmarkResult(test_name=aggregated_name)
        
        # Calculate averages
        total_time = sum(r.total_time for r in results) / len(results)
        per_step_time = sum(r.per_step_time for r in results) / len(results)
        performance_score = sum(r.performance_score for r in results) / len(results)
        gpu_utilization = sum(r.gpu_utilization_percent for r in results) / len(results)
        gpu_memory_used = sum(r.gpu_memory_used_gb for r in results) / len(results)
        gpu_memory_peak = max(r.gpu_memory_peak_gb for r in results)
        
        # Aggregate other metrics
        total_errors = sum(r.error_count for r in results)
        success_rate = sum(r.success_rate for r in results) / len(results)
        target_met = sum(1 for r in results if r.target_met) / len(results) >= 0.8  # 80% threshold
        
        # Collect all warnings
        all_warnings = []
        for r in results:
            all_warnings.extend(r.warnings)
        
        # Create aggregated result
        aggregated = BenchmarkResult(
            test_name=aggregated_name,
            model_name=results[0].model_name,
            architecture_type=results[0].architecture_type,
            total_time=total_time,
            per_step_time=per_step_time,
            target_time=results[0].target_time,
            target_met=target_met,
            performance_score=performance_score,
            gpu_utilization_percent=gpu_utilization,
            gpu_memory_used_gb=gpu_memory_used,
            gpu_memory_peak_gb=gpu_memory_peak,
            success_rate=success_rate,
            error_count=total_errors,
            warnings=all_warnings[:10],  # Limit to first 10 warnings
            gpu_name=results[0].gpu_name,
            driver_version=results[0].driver_version,
            cuda_version=results[0].cuda_version
        )
        
        # Add aggregation info to warnings
        aggregated.warnings.insert(0, f"Aggregated from {len(results)} individual tests")
        
        return aggregated
    
    def _log_benchmark_result(self, result: BenchmarkResult):
        """Log benchmark result"""
        status = "✅ PASSED" if result.target_met else "❌ FAILED"
        logger.info(f"📊 Benchmark Result [{result.test_name}] - {status}")
        logger.info(f"   Model: {result.model_name} ({result.architecture_type})")
        logger.info(f"   Time: {result.total_time:.3f}s (target: {result.target_time:.1f}s)")
        logger.info(f"   Performance Score: {result.performance_score:.1f}/100")
        logger.info(f"   Success Rate: {result.success_rate:.1f}%")
        
        if result.gpu_utilization_percent > 0:
            logger.info(f"   GPU Utilization: {result.gpu_utilization_percent:.1f}%")
            logger.info(f"   GPU Memory: {result.gpu_memory_used_gb:.2f}GB (peak: {result.gpu_memory_peak_gb:.2f}GB)")
        
        if result.multimodal_enabled:
            logger.info(f"   Text Processing: {result.text_processing_time:.3f}s")
            logger.info(f"   Image Analysis: {result.image_analysis_time:.3f}s")
        
        if result.error_count > 0:
            logger.warning(f"   Errors: {result.error_count}")
        
        if result.warnings:
            logger.info(f"   Warnings: {len(result.warnings)}")
    
    def export_benchmark_results(self, filepath: str):
        """
        Export all benchmark results to JSON file
        
        Args:
            filepath: Path to save results
        """
        try:
            export_data = {
                "export_timestamp": datetime.now().isoformat(),
                "hardware_info": self.hardware_info,
                "target_improvement_factor": self.target_improvement_factor,
                "target_generation_time": self.target_generation_time,
                "baseline_metrics": asdict(self.baseline_metrics) if self.baseline_metrics else None,
                "benchmark_results": [asdict(result) for result in self.benchmark_results],
                "summary": {
                    "total_benchmarks": len(self.benchmark_results),
                    "passed_benchmarks": sum(1 for r in self.benchmark_results if r.target_met),
                    "average_performance_score": sum(r.performance_score for r in self.benchmark_results) / len(self.benchmark_results) if self.benchmark_results else 0,
                    "average_improvement_factor": sum(r.improvement_factor for r in self.benchmark_results) / len(self.benchmark_results) if self.benchmark_results else 1
                }
            }
            
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            logger.info(f"Benchmark results exported to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to export benchmark results: {e}")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        if not self.benchmark_results:
            return {"error": "No benchmark results available"}
        
        # Calculate summary statistics
        total_tests = len(self.benchmark_results)
        passed_tests = sum(1 for r in self.benchmark_results if r.target_met)
        avg_performance_score = sum(r.performance_score for r in self.benchmark_results) / total_tests
        avg_improvement_factor = sum(r.improvement_factor for r in self.benchmark_results) / total_tests
        
        # Find best and worst results
        best_result = max(self.benchmark_results, key=lambda r: r.performance_score)
        worst_result = min(self.benchmark_results, key=lambda r: r.performance_score)
        
        summary = {
            "overall_status": "PASSED" if passed_tests == total_tests else "FAILED",
            "test_statistics": {
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "failed_tests": total_tests - passed_tests,
                "pass_rate": (passed_tests / total_tests) * 100,
                "average_performance_score": avg_performance_score,
                "average_improvement_factor": avg_improvement_factor
            },
            "performance_targets": {
                "target_improvement_factor": self.target_improvement_factor,
                "target_generation_time": self.target_generation_time,
                "improvement_target_met": avg_improvement_factor >= self.target_improvement_factor,
                "time_target_met": passed_tests / total_tests >= 0.8  # 80% of tests should pass
            },
            "best_performance": {
                "test_name": best_result.test_name,
                "performance_score": best_result.performance_score,
                "total_time": best_result.total_time,
                "improvement_factor": best_result.improvement_factor
            },
            "worst_performance": {
                "test_name": worst_result.test_name,
                "performance_score": worst_result.performance_score,
                "total_time": worst_result.total_time,
                "improvement_factor": worst_result.improvement_factor
            },
            "hardware_info": self.hardware_info,
            "recommendations": self._generate_performance_recommendations()
        }
        
        return summary
    
    def _generate_performance_recommendations(self) -> List[str]:
        """Generate performance improvement recommendations"""
        recommendations = []
        
        if not self.benchmark_results:
            return ["No benchmark data available for recommendations"]
        
        # Analyze results for recommendations
        avg_gpu_utilization = sum(r.gpu_utilization_percent for r in self.benchmark_results) / len(self.benchmark_results)
        avg_memory_efficiency = sum(r.gpu_memory_efficiency for r in self.benchmark_results) / len(self.benchmark_results)
        failed_tests = [r for r in self.benchmark_results if not r.target_met]
        
        # GPU utilization recommendations
        if avg_gpu_utilization < 70:
            recommendations.append("Low GPU utilization detected - consider optimizing pipeline for better GPU usage")
        
        # Memory efficiency recommendations
        if avg_memory_efficiency < 50:
            recommendations.append("Low memory efficiency - consider adjusting batch size or model precision")
        
        # Architecture-specific recommendations
        mmdit_tests = [r for r in self.benchmark_results if r.architecture_type == "MMDiT"]
        if mmdit_tests:
            avg_mmdit_time = sum(r.per_step_time for r in mmdit_tests) / len(mmdit_tests)
            if avg_mmdit_time > 1.0:
                recommendations.append("MMDiT step time is high - verify AttnProcessor2_0 is not used")
        
        # Error-based recommendations
        if failed_tests:
            recommendations.append(f"{len(failed_tests)} tests failed - review error logs for optimization opportunities")
        
        # General recommendations
        if not recommendations:
            recommendations.append("Performance looks good - consider testing with higher resolutions or batch sizes")
        
        return recommendations
    
    def clear_results(self):
        """Clear all benchmark results"""
        self.benchmark_results.clear()
        self.baseline_metrics = None
        logger.info("Benchmark results cleared")


# Utility functions for easy integration

def create_performance_validator(target_improvement: float = 50.0, 
                               target_time: float = 5.0) -> PerformanceValidator:
    """
    Create a performance validator with specified targets
    
    Args:
        target_improvement: Target improvement factor (e.g., 50x)
        target_time: Target generation time in seconds
        
    Returns:
        Configured PerformanceValidator instance
    """
    validator = PerformanceValidator(target_improvement, target_time)
    logger.info(f"Performance validator created - {target_improvement}x improvement, {target_time}s target")
    return validator


def create_default_benchmark_suite() -> BenchmarkSuite:
    """
    Create a default benchmark suite configuration
    
    Returns:
        Default BenchmarkSuite configuration
    """
    return BenchmarkSuite(
        name="Default MMDiT Performance Benchmark",
        target_improvement_factor=50.0,
        target_generation_time=5.0
    )


@contextmanager
def validate_performance_improvement(before_function: Callable, after_function: Callable,
                                   test_prompt: str = "A beautiful landscape",
                                   target_improvement: float = 50.0):
    """
    Convenience context manager for validating performance improvements
    
    Args:
        before_function: Original implementation
        after_function: Optimized implementation
        test_prompt: Test prompt
        target_improvement: Target improvement factor
        
    Yields:
        Validation results dictionary
    """
    validator = PerformanceValidator(target_improvement_factor=target_improvement)
    
    try:
        results = validator.run_speed_improvement_validation(
            before_function, after_function, test_prompt
        )
        yield results
        
    finally:
        # Export results if needed
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        export_path = f"performance_validation_{timestamp}.json"
        validator.export_benchmark_results(export_path)